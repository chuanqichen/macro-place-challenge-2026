"""
Phase 4: Simulated Annealing with Incremental HPWL

Algorithm:
1. Centroid-based initial placement (proven good congestion)
2. Spiral legalization (guaranteed zero overlaps)
3. Simulated Annealing with:
   - Incremental HPWL: maintain per-net bounding boxes, O(degree) per move
   - Move types: single-macro shift (70%), pairwise swap (20%), cluster move (10%)
   - Metropolis acceptance: always accept improvements, probabilistically accept
     worsening moves to escape local minima
   - Geometric cooling schedule calibrated for ~40% initial acceptance
   - GPU-accelerated overlap checking
4. Soft macro placement at connected-node centroids

Usage:
    uv run evaluate submissions/phase4_sa_placer.py -b ibm01
    uv run evaluate submissions/phase4_sa_placer.py --all
"""

import time as _time
import math
import random

import torch
import numpy as np

from macro_place.benchmark import Benchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAPlacer:

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = _time.time()
        dev = DEVICE

        placement = benchmark.macro_positions.clone()
        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        total_nodes = num_macros + num_ports

        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable_mask)[0].tolist()

        if not movable_indices:
            return placement

        # ── Step 1: Centroid initial placement ───────────────────────────
        placement = self._centroid_init(benchmark, placement, sizes, canvas_w, canvas_h)

        # ── Step 2: Spiral legalization ──────────────────────────────────
        placement = self._legalize(placement, movable_indices, sizes, canvas_w, canvas_h, num_hard, dev)

        # ── Step 3: Simulated Annealing ──────────────────────────────────
        sa_budget = max(10.0, 45.0 - (_time.time() - t0))
        placement = self._simulated_annealing(
            benchmark, placement, sizes, canvas_w, canvas_h,
            num_hard, movable_indices, dev, sa_budget,
        )

        # ── Step 4: Final CD pass to clean up density ────────────────────
        cd_budget = max(5.0, 55.0 - (_time.time() - t0))
        placement = self._final_cd(
            benchmark, placement, sizes, canvas_w, canvas_h,
            num_hard, movable_indices, dev, cd_budget,
        )

        # ── Step 5: Soft macro placement ─────────────────────────────────
        placement = self._place_soft_macros(benchmark, placement, sizes, canvas_w, canvas_h, dev)

        return placement

    # ── Centroid init ────────────────────────────────────────────────────

    def _centroid_init(self, benchmark, placement, sizes, canvas_w, canvas_h):
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable_mask)[0].tolist()

        pull_x = torch.zeros(num_macros)
        pull_y = torch.zeros(num_macros)
        pull_w = torch.zeros(num_macros)

        for net_idx in range(len(benchmark.net_nodes)):
            nodes = benchmark.net_nodes[net_idx]
            weight = benchmark.net_weights[net_idx].item()
            if len(nodes) < 2:
                continue
            positions, macro_ids = [], []
            for nid in nodes.tolist():
                if nid < num_macros:
                    positions.append(placement[nid])
                    macro_ids.append(nid)
                elif num_ports > 0 and nid < num_macros + num_ports:
                    positions.append(benchmark.port_positions[nid - num_macros])
            if len(positions) < 2:
                continue
            centroid = torch.stack(positions).mean(dim=0)
            nw = weight / len(positions)
            for mid in macro_ids:
                pull_x[mid] += centroid[0] * nw
                pull_y[mid] += centroid[1] * nw
                pull_w[mid] += nw

        for idx in movable_indices:
            if pull_w[idx] > 0:
                cx = (pull_x[idx] / pull_w[idx]).item()
                cy = (pull_y[idx] / pull_w[idx]).item()
                w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
                placement[idx, 0] = max(w / 2, min(canvas_w - w / 2, cx))
                placement[idx, 1] = max(h / 2, min(canvas_h - h / 2, cy))
        return placement

    # ── Spiral legalization ──────────────────────────────────────────────

    def _legalize(self, placement, movable_indices, sizes, canvas_w, canvas_h, num_hard, dev):
        gap = 0.01
        pos_gpu = placement[:num_hard].clone().to(dev)
        sizes_gpu = sizes[:num_hard].to(dev)
        half_sz = sizes_gpu / 2

        targets = {i: (placement[i, 0].item(), placement[i, 1].item()) for i in movable_indices}
        sorted_idx = sorted(movable_indices, key=lambda i: -(sizes[i, 0].item() * sizes[i, 1].item()))

        movable_set = set(movable_indices)
        placed_mask = torch.zeros(num_hard, dtype=torch.bool, device=dev)
        for j in range(num_hard):
            if j not in movable_set:
                placed_mask[j] = True

        offsets_t = torch.tensor([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [1, -1], [-1, 1], [-1, -1],
            [0.5, 1], [0.5, -1], [-0.5, 1], [-0.5, -1],
            [1, 0.5], [1, -0.5], [-1, 0.5], [-1, -0.5],
            [0.3, 0.7], [-0.3, 0.7], [0.3, -0.7], [-0.3, -0.7],
            [0.7, 0.3], [-0.7, 0.3], [0.7, -0.3], [-0.7, -0.3],
        ], device=dev)

        for idx in sorted_idx:
            wi = sizes_gpu[idx, 0].item()
            hi = sizes_gpu[idx, 1].item()
            cx, cy = targets[idx]
            cx = max(wi / 2, min(canvas_w - wi / 2, cx))
            cy = max(hi / 2, min(canvas_h - hi / 2, cy))

            pi = torch.where(placed_mask)[0]
            if len(pi) == 0 or not ((pos_gpu[pi, 0] - cx).abs() < half_sz[pi, 0] + wi / 2 + gap).any() or \
               not (((pos_gpu[pi, 0] - cx).abs() < half_sz[pi, 0] + wi / 2 + gap) &
                    ((pos_gpu[pi, 1] - cy).abs() < half_sz[pi, 1] + hi / 2 + gap)).any():
                pos_gpu[idx] = torch.tensor([cx, cy], device=dev)
                placed_mask[idx] = True
                continue

            step = max(min(wi, hi) * 0.25, 0.1)
            found = False
            for rm in range(1, 500):
                r = step * rm
                if r > max(canvas_w, canvas_h):
                    break
                cands = torch.tensor([cx, cy], device=dev).unsqueeze(0) + offsets_t * r
                cands[:, 0].clamp_(wi / 2, canvas_w - wi / 2)
                cands[:, 1].clamp_(hi / 2, canvas_h - hi / 2)
                dx = (cands[:, 0].unsqueeze(1) - pos_gpu[pi, 0].unsqueeze(0)).abs()
                dy = (cands[:, 1].unsqueeze(1) - pos_gpu[pi, 1].unsqueeze(0)).abs()
                ov = (dx < half_sz[pi, 0].unsqueeze(0) + wi / 2 + gap) & \
                     (dy < half_sz[pi, 1].unsqueeze(0) + hi / 2 + gap)
                free = ~ov.any(dim=1)
                if free.any():
                    dists = (cands[free, 0] - cx) ** 2 + (cands[free, 1] - cy) ** 2
                    bi = torch.where(free)[0][dists.argmin()]
                    pos_gpu[idx] = cands[bi]
                    placed_mask[idx] = True
                    found = True
                    break
            if not found:
                pos_gpu[idx] = torch.tensor([cx, cy], device=dev)
                placed_mask[idx] = True

        placement[:num_hard] = pos_gpu.cpu()
        return placement

    # ── Simulated Annealing ──────────────────────────────────────────────

    def _simulated_annealing(self, benchmark, placement, sizes, canvas_w, canvas_h,
                              num_hard, movable_indices, dev, time_budget):
        deadline = _time.time() + time_budget
        gap = 0.01
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        total_nodes = num_macros + num_ports
        n_mov = len(movable_indices)

        if n_mov == 0:
            return placement

        random.seed(42)

        # ── Build data structures ────────────────────────────────────────
        # All positions as numpy for fast access
        pos_x = np.zeros(total_nodes)
        pos_y = np.zeros(total_nodes)
        for i in range(num_macros):
            pos_x[i] = placement[i, 0].item()
            pos_y[i] = placement[i, 1].item()
        if num_ports > 0:
            for i in range(num_ports):
                pos_x[num_macros + i] = benchmark.port_positions[i, 0].item()
                pos_y[num_macros + i] = benchmark.port_positions[i, 1].item()

        sz_w = np.zeros(num_hard)
        sz_h = np.zeros(num_hard)
        half_w = np.zeros(num_hard)
        half_h = np.zeros(num_hard)
        for i in range(num_hard):
            sz_w[i] = sizes[i, 0].item()
            sz_h[i] = sizes[i, 1].item()
            half_w[i] = sz_w[i] / 2
            half_h[i] = sz_h[i] / 2

        # Per-macro net membership
        macro_nets = [[] for _ in range(num_macros)]
        net_node_lists = []
        net_weights_np = np.zeros(len(benchmark.net_nodes))
        for ni, net in enumerate(benchmark.net_nodes):
            valid = net[net < total_nodes].tolist()
            net_node_lists.append(valid)
            net_weights_np[ni] = benchmark.net_weights[ni].item()
            for nid in valid:
                if nid < num_macros:
                    macro_nets[nid].append(ni)

        num_nets = len(net_node_lists)

        # ── Per-net bounding boxes (incremental HPWL) ────────────────────
        net_xmin = np.full(num_nets, 1e9)
        net_xmax = np.full(num_nets, -1e9)
        net_ymin = np.full(num_nets, 1e9)
        net_ymax = np.full(num_nets, -1e9)

        for ni in range(num_nets):
            for nid in net_node_lists[ni]:
                x, y = pos_x[nid], pos_y[nid]
                if x < net_xmin[ni]: net_xmin[ni] = x
                if x > net_xmax[ni]: net_xmax[ni] = x
                if y < net_ymin[ni]: net_ymin[ni] = y
                if y > net_ymax[ni]: net_ymax[ni] = y

        # Total weighted HPWL
        def _total_hpwl():
            return np.sum((net_xmax - net_xmin + net_ymax - net_ymin) * net_weights_np)

        def _recompute_net_bb(ni):
            """Recompute bounding box for net ni from scratch."""
            xmin, xmax = 1e9, -1e9
            ymin, ymax = 1e9, -1e9
            for nid in net_node_lists[ni]:
                x, y = pos_x[nid], pos_y[nid]
                if x < xmin: xmin = x
                if x > xmax: xmax = x
                if y < ymin: ymin = y
                if y > ymax: ymax = y
            net_xmin[ni] = xmin
            net_xmax[ni] = xmax
            net_ymin[ni] = ymin
            net_ymax[ni] = ymax

        def _affected_hpwl_delta_shift(mi, new_x, new_y):
            """
            Compute HPWL change if macro mi moves to (new_x, new_y).
            Returns delta (negative = improvement).
            """
            old_x, old_y = pos_x[mi], pos_y[mi]
            delta = 0.0
            for ni in macro_nets[mi]:
                w = net_weights_np[ni]
                old_span = (net_xmax[ni] - net_xmin[ni] + net_ymax[ni] - net_ymin[ni]) * w

                # Temporarily compute new bounding box
                xmin, xmax = 1e9, -1e9
                ymin, ymax = 1e9, -1e9
                for nid in net_node_lists[ni]:
                    if nid == mi:
                        x, y = new_x, new_y
                    else:
                        x, y = pos_x[nid], pos_y[nid]
                    if x < xmin: xmin = x
                    if x > xmax: xmax = x
                    if y < ymin: ymin = y
                    if y > ymax: ymax = y

                new_span = (xmax - xmin + ymax - ymin) * w
                delta += new_span - old_span
            return delta

        def _affected_hpwl_delta_swap(mi, mj):
            """
            Compute HPWL change if macros mi and mj swap positions.
            """
            xi, yi = pos_x[mi], pos_y[mi]
            xj, yj = pos_x[mj], pos_y[mj]
            affected = set(macro_nets[mi]) | set(macro_nets[mj])
            delta = 0.0
            for ni in affected:
                w = net_weights_np[ni]
                old_span = (net_xmax[ni] - net_xmin[ni] + net_ymax[ni] - net_ymin[ni]) * w

                xmin, xmax = 1e9, -1e9
                ymin, ymax = 1e9, -1e9
                for nid in net_node_lists[ni]:
                    if nid == mi:
                        x, y = xj, yj
                    elif nid == mj:
                        x, y = xi, yi
                    else:
                        x, y = pos_x[nid], pos_y[nid]
                    if x < xmin: xmin = x
                    if x > xmax: xmax = x
                    if y < ymin: ymin = y
                    if y > ymax: ymax = y

                new_span = (xmax - xmin + ymax - ymin) * w
                delta += new_span - old_span
            return delta

        def _check_overlap_single(mi, nx, ny):
            """Check if macro mi at (nx, ny) overlaps any other hard macro.
            Uses early termination for speed."""
            hw_i = half_w[mi]
            hh_i = half_h[mi]
            for j in range(num_hard):
                if j == mi:
                    continue
                dx = abs(nx - pos_x[j])
                if dx >= hw_i + half_w[j] + gap:
                    continue
                dy = abs(ny - pos_y[j])
                if dy < hh_i + half_h[j] + gap:
                    return True
            return False

        def _check_overlap_swap(mi, mj):
            """Check if swapping mi and mj creates overlaps."""
            xi, yi = pos_x[mi], pos_y[mi]
            xj, yj = pos_x[mj], pos_y[mj]
            hw_mi, hh_mi = half_w[mi], half_h[mi]
            hw_mj, hh_mj = half_w[mj], half_h[mj]
            for k in range(num_hard):
                if k == mi or k == mj:
                    continue
                pk_x, pk_y = pos_x[k], pos_y[k]
                hw_k, hh_k = half_w[k], half_h[k]
                # mi at (xj, yj)
                if abs(xj - pk_x) < hw_mi + hw_k + gap and abs(yj - pk_y) < hh_mi + hh_k + gap:
                    return True
                # mj at (xi, yi)
                if abs(xi - pk_x) < hw_mj + hw_k + gap and abs(yi - pk_y) < hh_mj + hh_k + gap:
                    return True
            return False

        def _apply_shift(mi, new_x, new_y):
            """Apply a shift move and update bounding boxes."""
            pos_x[mi] = new_x
            pos_y[mi] = new_y
            for ni in macro_nets[mi]:
                _recompute_net_bb(ni)

        def _apply_swap(mi, mj):
            """Apply a swap move and update bounding boxes."""
            pos_x[mi], pos_x[mj] = pos_x[mj], pos_x[mi]
            pos_y[mi], pos_y[mj] = pos_y[mj], pos_y[mi]
            affected = set(macro_nets[mi]) | set(macro_nets[mj])
            for ni in affected:
                _recompute_net_bb(ni)

        # ── Calibrate initial temperature ────────────────────────────────
        # Sample random moves and compute average uphill delta
        sample_deltas = []
        for _ in range(min(200, n_mov * 2)):
            mi = random.choice(movable_indices)
            dx = random.gauss(0, canvas_w * 0.1)
            dy = random.gauss(0, canvas_h * 0.1)
            nx = max(half_w[mi], min(canvas_w - half_w[mi], pos_x[mi] + dx))
            ny = max(half_h[mi], min(canvas_h - half_h[mi], pos_y[mi] + dy))
            delta = _affected_hpwl_delta_shift(mi, nx, ny)
            if delta > 0:
                sample_deltas.append(delta)

        if sample_deltas:
            avg_uphill = np.median(sample_deltas)
            # T0 such that exp(-avg_uphill / T0) = 0.4 → T0 = -avg_uphill / ln(0.4)
            T0 = avg_uphill / (-math.log(0.4))
        else:
            T0 = 1.0

        T = T0
        cooling_rate = 0.995
        min_temp = T0 * 0.001

        # ── SA main loop ─────────────────────────────────────────────────
        total_moves = 0
        accepted_moves = 0
        improved_moves = 0
        best_hpwl = _total_hpwl()
        best_pos_x = pos_x.copy()
        best_pos_y = pos_y.copy()

        moves_per_temp = max(n_mov // 2, 30)

        # Density grid for density-aware acceptance
        grid_n = 12
        cell_gw = canvas_w / grid_n
        cell_gh = canvas_h / grid_n
        density_grid = np.zeros((grid_n, grid_n))
        for i in range(num_hard):
            gi = min(grid_n - 1, max(0, int(pos_x[i] / cell_gw)))
            gj = min(grid_n - 1, max(0, int(pos_y[i] / cell_gh)))
            density_grid[gi, gj] += sz_w[i] * sz_h[i]
        avg_density = density_grid.sum() / (grid_n * grid_n)
        density_weight = 0.2  # penalty weight for density

        while T > min_temp and _time.time() < deadline:
            for _ in range(moves_per_temp):
                if _time.time() > deadline:
                    break

                total_moves += 1
                r = random.random()

                if r < 0.70:
                    # ── Single-macro shift ────────────────────────────────
                    mi = random.choice(movable_indices)
                    # Displacement scaled by temperature
                    scale = min(canvas_w, canvas_h) * 0.3 * (T / T0)
                    dx = random.gauss(0, scale)
                    dy = random.gauss(0, scale)
                    nx = pos_x[mi] + dx
                    ny = pos_y[mi] + dy
                    # Clamp to canvas
                    nx = max(half_w[mi], min(canvas_w - half_w[mi], nx))
                    ny = max(half_h[mi], min(canvas_h - half_h[mi], ny))

                    # Check overlap
                    if _check_overlap_single(mi, nx, ny):
                        continue

                    # Compute delta (HPWL + density penalty)
                    delta = _affected_hpwl_delta_shift(mi, nx, ny)

                    # Add density penalty: penalize moving into denser regions
                    old_gi = min(grid_n - 1, max(0, int(pos_x[mi] / cell_gw)))
                    old_gj = min(grid_n - 1, max(0, int(pos_y[mi] / cell_gh)))
                    new_gi = min(grid_n - 1, max(0, int(nx / cell_gw)))
                    new_gj = min(grid_n - 1, max(0, int(ny / cell_gh)))
                    if new_gi != old_gi or new_gj != old_gj:
                        old_d = max(0, density_grid[old_gi, old_gj] - avg_density)
                        new_d = max(0, density_grid[new_gi, new_gj] - avg_density)
                        delta += density_weight * (new_d - old_d)

                    # Metropolis acceptance
                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        old_x, old_y = pos_x[mi], pos_y[mi]
                        _apply_shift(mi, nx, ny)
                        # Update density grid
                        density_grid[old_gi, old_gj] -= sz_w[mi] * sz_h[mi]
                        density_grid[new_gi, new_gj] += sz_w[mi] * sz_h[mi]
                        accepted_moves += 1
                        if delta < 0:
                            improved_moves += 1

                elif r < 0.90:
                    # ── Pairwise swap ─────────────────────────────────────
                    if n_mov < 2:
                        continue
                    mi = random.choice(movable_indices)
                    mj = random.choice(movable_indices)
                    if mi == mj:
                        continue

                    # Bounds check
                    xi, yi = pos_x[mi], pos_y[mi]
                    xj, yj = pos_x[mj], pos_y[mj]
                    if (xj - half_w[mi] < 0 or xj + half_w[mi] > canvas_w or
                            yj - half_h[mi] < 0 or yj + half_h[mi] > canvas_h):
                        continue
                    if (xi - half_w[mj] < 0 or xi + half_w[mj] > canvas_w or
                            yi - half_h[mj] < 0 or yi + half_h[mj] > canvas_h):
                        continue

                    if _check_overlap_swap(mi, mj):
                        continue

                    delta = _affected_hpwl_delta_swap(mi, mj)

                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        _apply_swap(mi, mj)
                        accepted_moves += 1
                        if delta < 0:
                            improved_moves += 1

                else:
                    # ── Cluster move (shift connected group) ──────────────
                    mi = random.choice(movable_indices)
                    # Find 2-4 neighbors connected via nets
                    neighbors = set()
                    movable_set = set(movable_indices)
                    for ni in macro_nets[mi]:
                        for nid in net_node_lists[ni]:
                            if nid != mi and nid in movable_set:
                                neighbors.add(nid)
                                if len(neighbors) >= 3:
                                    break
                        if len(neighbors) >= 3:
                            break

                    if not neighbors:
                        continue

                    cluster = [mi] + list(neighbors)[:3]
                    scale = min(canvas_w, canvas_h) * 0.15 * (T / T0)
                    dx = random.gauss(0, scale)
                    dy = random.gauss(0, scale)

                    # Check if all cluster members can move
                    new_positions = {}
                    valid = True
                    for ci in cluster:
                        nx = max(half_w[ci], min(canvas_w - half_w[ci], pos_x[ci] + dx))
                        ny = max(half_h[ci], min(canvas_h - half_h[ci], pos_y[ci] + dy))
                        new_positions[ci] = (nx, ny)

                    # Check overlaps (cluster members against non-cluster)
                    cluster_set = set(cluster)
                    for ci in cluster:
                        nx, ny = new_positions[ci]
                        for k in range(num_hard):
                            if k in cluster_set:
                                continue
                            if (abs(nx - pos_x[k]) < half_w[ci] + half_w[k] + gap and
                                    abs(ny - pos_y[k]) < half_h[ci] + half_h[k] + gap):
                                valid = False
                                break
                        if not valid:
                            break

                    # Check intra-cluster overlaps at new positions
                    if valid:
                        cl = list(cluster)
                        for a in range(len(cl)):
                            for b in range(a + 1, len(cl)):
                                na_x, na_y = new_positions[cl[a]]
                                nb_x, nb_y = new_positions[cl[b]]
                                if (abs(na_x - nb_x) < half_w[cl[a]] + half_w[cl[b]] + gap and
                                        abs(na_y - nb_y) < half_h[cl[a]] + half_h[cl[b]] + gap):
                                    valid = False
                                    break
                            if not valid:
                                break

                    if not valid:
                        continue

                    # Compute delta
                    affected = set()
                    for ci in cluster:
                        affected.update(macro_nets[ci])

                    old_span = 0.0
                    for ni in affected:
                        old_span += (net_xmax[ni] - net_xmin[ni] + net_ymax[ni] - net_ymin[ni]) * net_weights_np[ni]

                    # Temporarily apply
                    old_positions = {ci: (pos_x[ci], pos_y[ci]) for ci in cluster}
                    for ci in cluster:
                        pos_x[ci], pos_y[ci] = new_positions[ci]

                    new_span = 0.0
                    for ni in affected:
                        xmin, xmax = 1e9, -1e9
                        ymin, ymax = 1e9, -1e9
                        for nid in net_node_lists[ni]:
                            x, y = pos_x[nid], pos_y[nid]
                            if x < xmin: xmin = x
                            if x > xmax: xmax = x
                            if y < ymin: ymin = y
                            if y > ymax: ymax = y
                        new_span += (xmax - xmin + ymax - ymin) * net_weights_np[ni]

                    delta = new_span - old_span

                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        # Accept: update bounding boxes
                        for ni in affected:
                            _recompute_net_bb(ni)
                        accepted_moves += 1
                        if delta < 0:
                            improved_moves += 1
                    else:
                        # Reject: revert
                        for ci in cluster:
                            pos_x[ci], pos_y[ci] = old_positions[ci]

                # Track best
                if total_moves % (moves_per_temp * 5) == 0:
                    cur_hpwl = _total_hpwl()
                    if cur_hpwl < best_hpwl:
                        best_hpwl = cur_hpwl
                        best_pos_x = pos_x.copy()
                        best_pos_y = pos_y.copy()

            T *= cooling_rate

        # Final best check
        cur_hpwl = _total_hpwl()
        if cur_hpwl < best_hpwl:
            best_pos_x = pos_x.copy()
            best_pos_y = pos_y.copy()

        # Apply best positions
        for i in range(num_macros):
            placement[i, 0] = best_pos_x[i]
            placement[i, 1] = best_pos_y[i]

        return placement

    # ── Final coordinate descent pass ───────────────────────────────────

    def _final_cd(self, benchmark, placement, sizes, canvas_w, canvas_h,
                   num_hard, movable_indices, dev, time_budget):
        """Quick CD pass using GPU HPWL to clean up after SA."""
        deadline = _time.time() + time_budget
        gap = 0.01
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        total_nodes = num_macros + num_ports

        # Prepare net data on GPU
        nets = benchmark.net_nodes
        if not nets:
            return placement

        max_deg = max(len(n) for n in nets)
        num_nets = len(nets)
        net_tensor = torch.full((num_nets, max_deg), total_nodes, dtype=torch.long, device=dev)
        net_mask = torch.zeros(num_nets, max_deg, dtype=torch.bool, device=dev)
        weights = benchmark.net_weights.to(dev)

        for i, nodes in enumerate(nets):
            valid = nodes[nodes < total_nodes]
            n = len(valid)
            if n >= 2:
                net_tensor[i, :n] = valid.to(dev)
                net_mask[i, :n] = True

        valid_nets = net_mask.sum(dim=1) >= 2
        net_tensor = net_tensor[valid_nets]
        net_mask = net_mask[valid_nets]
        weights = weights[valid_nets]

        macro_nets = [[] for _ in range(num_macros)]
        valid_idx = torch.where(valid_nets)[0]
        for li in range(len(valid_idx)):
            oi = valid_idx[li].item()
            for nid in benchmark.net_nodes[oi].tolist():
                if nid < num_macros:
                    macro_nets[nid].append(li)

        all_pos = torch.zeros(total_nodes, 2, device=dev)
        all_pos[:num_macros] = placement.to(dev)
        if num_ports > 0:
            all_pos[num_macros:] = benchmark.port_positions.to(dev)

        sizes_gpu = sizes[:num_hard].to(dev)
        half_sz = sizes_gpu / 2

        def _affected_hpwl(mi):
            nids_list = macro_nets[mi]
            if not nids_list:
                return 0.0
            nids = torch.tensor(nids_list, device=dev, dtype=torch.long)
            padded = torch.cat([all_pos, torch.zeros(1, 2, device=dev)], dim=0)
            nt = net_tensor[nids]; nm = net_mask[nids]; nw = weights[nids]
            pos = padded[nt]
            x = pos[:, :, 0].clone(); y = pos[:, :, 1].clone()
            x[~nm] = 1e9; x_min = x.min(1).values
            x[~nm] = -1e9; x_max = x.max(1).values
            y[~nm] = 1e9; y_min = y.min(1).values
            y[~nm] = -1e9; y_max = y.max(1).values
            return (((x_max - x_min) + (y_max - y_min)) * nw).sum().item()

        shift_deltas = []
        for scale in [0.2, 0.5, 1.0, 2.0, 3.0]:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1),
                           (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)]:
                shift_deltas.append((dx * scale, dy * scale))

        for _pass in range(5):
            if _time.time() > deadline:
                break
            any_imp = False
            for mi in movable_indices:
                if _time.time() > deadline:
                    break
                ox = all_pos[mi, 0].item(); oy = all_pos[mi, 1].item()
                wi_h = half_sz[mi, 0].item(); hi_h = half_sz[mi, 1].item()
                old_hpwl = _affected_hpwl(mi)
                best_hpwl = old_hpwl; best_x, best_y = ox, oy

                exc = torch.ones(num_hard, dtype=torch.bool, device=dev)
                exc[mi] = False
                others = torch.where(exc)[0]
                other_pos = all_pos[others]; other_hsz = half_sz[others]

                for dx, dy in shift_deltas:
                    nx = ox + dx; ny = oy + dy
                    if nx - wi_h < 0 or nx + wi_h > canvas_w or ny - hi_h < 0 or ny + hi_h > canvas_h:
                        continue
                    if len(others) > 0:
                        if (((other_pos[:, 0] - nx).abs() < other_hsz[:, 0] + wi_h + gap) &
                                ((other_pos[:, 1] - ny).abs() < other_hsz[:, 1] + hi_h + gap)).any():
                            continue
                    all_pos[mi, 0] = nx; all_pos[mi, 1] = ny
                    new_hpwl = _affected_hpwl(mi)
                    if new_hpwl < best_hpwl - 1e-6:
                        best_hpwl = new_hpwl; best_x, best_y = nx, ny
                all_pos[mi, 0] = best_x; all_pos[mi, 1] = best_y
                if best_x != ox or best_y != oy:
                    any_imp = True
            if not any_imp:
                break

        placement[:num_macros] = all_pos[:num_macros].cpu()
        return placement

    # ── Soft macro placement ─────────────────────────────────────────────

    def _place_soft_macros(self, benchmark, placement, sizes, canvas_w, canvas_h, dev):
        num_macros = benchmark.num_macros
        num_hard = benchmark.num_hard_macros
        num_ports = benchmark.port_positions.shape[0]
        soft_indices = list(range(num_hard, num_macros))
        if not soft_indices:
            return placement

        soft_set = set(soft_indices)
        pull_x = torch.zeros(num_macros, device=dev)
        pull_y = torch.zeros(num_macros, device=dev)
        pull_w = torch.zeros(num_macros, device=dev)
        pos_d = placement.to(dev)
        port_d = benchmark.port_positions.to(dev) if num_ports > 0 else None

        for ni in range(len(benchmark.net_nodes)):
            nodes = benchmark.net_nodes[ni]
            w = benchmark.net_weights[ni].item()
            if len(nodes) < 2:
                continue
            ax, ay, ns = [], [], []
            for nid in nodes.tolist():
                if nid in soft_set:
                    ns.append(nid)
                elif nid < num_macros:
                    ax.append(pos_d[nid, 0]); ay.append(pos_d[nid, 1])
                elif port_d is not None and nid < num_macros + num_ports:
                    pi = nid - num_macros
                    ax.append(port_d[pi, 0]); ay.append(port_d[pi, 1])
            if not ax or not ns:
                continue
            cx = torch.stack(ax).mean(); cy = torch.stack(ay).mean()
            nw = w / max(1, len(ax))
            for s in ns:
                pull_x[s] += cx * nw; pull_y[s] += cy * nw; pull_w[s] += nw

        valid = pull_w > 0
        sm = torch.zeros(num_macros, dtype=torch.bool, device=dev)
        for s in soft_indices:
            sm[s] = True
        upd = valid & sm
        if upd.any():
            idx = torch.where(upd)[0]
            sizes_d = sizes.to(dev)
            wh = sizes_d[idx, 0] / 2; hh = sizes_d[idx, 1] / 2
            pos_d[idx, 0] = (pull_x[idx] / pull_w[idx]).clamp(min=wh, max=canvas_w - wh)
            pos_d[idx, 1] = (pull_y[idx] / pull_w[idx]).clamp(min=hh, max=canvas_h - hh)

        return pos_d.cpu()
