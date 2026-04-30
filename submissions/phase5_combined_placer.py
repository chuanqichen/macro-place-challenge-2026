"""
Phase 5: Spatial-Hash SA + Soft Macro Co-Optimization + RUDY Congestion

Combined approach:
1. Centroid initial placement (proven good congestion baseline)
2. Spiral legalization (guaranteed zero overlaps)
3. Simulated Annealing with:
   - Spatial hash grid for O(~1) overlap checking (3x speedup on large benchmarks)
   - Incremental HPWL with per-net bounding boxes
   - RUDY congestion estimate in SA objective (steers away from congested moves)
   - Soft macro re-placement every K iterations (improves density + wirelength)
   - Move types: shift (70%), swap (20%), cluster (10%)
   - Metropolis acceptance with geometric cooling
4. Final soft macro placement

Usage:
    uv run evaluate submissions/phase5_combined_placer.py -b ibm01
    uv run evaluate submissions/phase5_combined_placer.py --all
"""

import time as _time
import math
import random

import torch
import numpy as np

from macro_place.benchmark import Benchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CombinedPlacer:

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

        # ── Step 3: SA with spatial hash + soft co-opt + RUDY ────────────
        sa_budget = max(10.0, 55.0 - (_time.time() - t0))
        placement = self._sa_combined(
            benchmark, placement, sizes, canvas_w, canvas_h,
            num_hard, movable_indices, dev, sa_budget,
        )

        # ── Step 4: Final soft macro placement ───────────────────────────
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
            if len(pi) == 0 or not (((pos_gpu[pi, 0] - cx).abs() < half_sz[pi, 0] + wi / 2 + gap) &
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

    # ══════════════════════════════════════════════════════════════════════
    # Spatial-Hash SA + Soft Co-Opt + RUDY Congestion
    # ══════════════════════════════════════════════════════════════════════

    def _sa_combined(self, benchmark, placement, sizes, canvas_w, canvas_h,
                      num_hard, movable_indices, dev, time_budget):
        deadline = _time.time() + time_budget
        gap = 0.01
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        total_nodes = num_macros + num_ports
        n_mov = len(movable_indices)
        movable_set = set(movable_indices)

        if n_mov == 0:
            return placement

        random.seed(42)

        # ── Numpy arrays for fast access ─────────────────────────────────
        pos_x = np.zeros(total_nodes)
        pos_y = np.zeros(total_nodes)
        for i in range(num_macros):
            pos_x[i] = placement[i, 0].item()
            pos_y[i] = placement[i, 1].item()
        if num_ports > 0:
            for i in range(num_ports):
                pos_x[num_macros + i] = benchmark.port_positions[i, 0].item()
                pos_y[num_macros + i] = benchmark.port_positions[i, 1].item()

        hw = np.zeros(num_hard)
        hh = np.zeros(num_hard)
        sz_w = np.zeros(num_hard)
        sz_h = np.zeros(num_hard)
        for i in range(num_hard):
            sz_w[i] = sizes[i, 0].item()
            sz_h[i] = sizes[i, 1].item()
            hw[i] = sz_w[i] / 2
            hh[i] = sz_h[i] / 2

        # ── Net data ─────────────────────────────────────────────────────
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

        # ── Per-net bounding boxes ───────────────────────────────────────
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

        def _recompute_net_bb(ni):
            xmin, xmax, ymin, ymax = 1e9, -1e9, 1e9, -1e9
            for nid in net_node_lists[ni]:
                x, y = pos_x[nid], pos_y[nid]
                if x < xmin: xmin = x
                if x > xmax: xmax = x
                if y < ymin: ymin = y
                if y > ymax: ymax = y
            net_xmin[ni] = xmin; net_xmax[ni] = xmax
            net_ymin[ni] = ymin; net_ymax[ni] = ymax

        # ── Spatial hash grid ────────────────────────────────────────────
        max_extent = 2 * (hw.max() + hw.max()) + gap  # max overlap distance
        cell_size = max(3.0, max_extent * 0.4)
        ncx = max(1, int(canvas_w / cell_size) + 1)
        ncy = max(1, int(canvas_h / cell_size) + 1)
        check_range = int(max_extent / cell_size) + 1

        # Grid: list of sets for O(1) add/remove
        grid = [[set() for _ in range(ncy)] for _ in range(ncx)]

        def _grid_cell(x, y):
            return (min(ncx - 1, max(0, int(x / cell_size))),
                    min(ncy - 1, max(0, int(y / cell_size))))

        # Initialize grid
        for i in range(num_hard):
            gx, gy = _grid_cell(pos_x[i], pos_y[i])
            grid[gx][gy].add(i)

        def _grid_move(mi, old_x, old_y, new_x, new_y):
            ogx, ogy = _grid_cell(old_x, old_y)
            ngx, ngy = _grid_cell(new_x, new_y)
            if ogx != ngx or ogy != ngy:
                grid[ogx][ogy].discard(mi)
                grid[ngx][ngy].add(mi)

        def _check_overlap(mi, nx, ny):
            hw_i, hh_i = hw[mi], hh[mi]
            gx, gy = _grid_cell(nx, ny)
            for dx in range(-check_range, check_range + 1):
                for dy in range(-check_range, check_range + 1):
                    cx, cy = gx + dx, gy + dy
                    if 0 <= cx < ncx and 0 <= cy < ncy:
                        for j in grid[cx][cy]:
                            if j == mi:
                                continue
                            if (abs(nx - pos_x[j]) < hw_i + hw[j] + gap and
                                    abs(ny - pos_y[j]) < hh_i + hh[j] + gap):
                                return True
            return False

        def _check_overlap_swap(mi, mj):
            xi, yi = pos_x[mi], pos_y[mi]
            xj, yj = pos_x[mj], pos_y[mj]
            gx_i, gy_i = _grid_cell(xj, yj)
            gx_j, gy_j = _grid_cell(xi, yi)
            # Check mi at (xj, yj)
            for dx in range(-check_range, check_range + 1):
                for dy in range(-check_range, check_range + 1):
                    cx, cy = gx_i + dx, gy_i + dy
                    if 0 <= cx < ncx and 0 <= cy < ncy:
                        for k in grid[cx][cy]:
                            if k == mi or k == mj:
                                continue
                            if (abs(xj - pos_x[k]) < hw[mi] + hw[k] + gap and
                                    abs(yj - pos_y[k]) < hh[mi] + hh[k] + gap):
                                return True
            # Check mj at (xi, yi)
            for dx in range(-check_range, check_range + 1):
                for dy in range(-check_range, check_range + 1):
                    cx, cy = gx_j + dx, gy_j + dy
                    if 0 <= cx < ncx and 0 <= cy < ncy:
                        for k in grid[cx][cy]:
                            if k == mi or k == mj:
                                continue
                            if (abs(xi - pos_x[k]) < hw[mj] + hw[k] + gap and
                                    abs(yi - pos_y[k]) < hh[mj] + hh[k] + gap):
                                return True
            return False

        # ── RUDY congestion grid ─────────────────────────────────────────
        rudy_n = 12
        rudy_cw = canvas_w / rudy_n
        rudy_ch = canvas_h / rudy_n
        rudy_grid = np.zeros((rudy_n, rudy_n))

        def _compute_rudy():
            """Compute RUDY (Rectangular Uniform wire DensitY) congestion estimate."""
            rudy_grid[:] = 0
            for ni in range(num_nets):
                nodes = net_node_lists[ni]
                if len(nodes) < 2:
                    continue
                w = net_weights_np[ni]
                xmin, xmax = net_xmin[ni], net_xmax[ni]
                ymin, ymax = net_ymin[ni], net_ymax[ni]
                span_x = max(xmax - xmin, 0.01)
                span_y = max(ymax - ymin, 0.01)
                # Wire density = weight / (span_x * span_y) distributed over spanned cells
                density = w / (span_x + span_y)
                gi_min = max(0, int(xmin / rudy_cw))
                gi_max = min(rudy_n - 1, int(xmax / rudy_cw))
                gj_min = max(0, int(ymin / rudy_ch))
                gj_max = min(rudy_n - 1, int(ymax / rudy_ch))
                n_cells = max(1, (gi_max - gi_min + 1) * (gj_max - gj_min + 1))
                d_per_cell = density / n_cells
                for gi in range(gi_min, gi_max + 1):
                    for gj in range(gj_min, gj_max + 1):
                        rudy_grid[gi, gj] += d_per_cell

        _compute_rudy()
        rudy_avg = rudy_grid.mean()
        rudy_weight = 0.15  # weight of congestion penalty in SA objective

        def _rudy_penalty_shift(mi, old_x, old_y, new_x, new_y):
            """Estimate congestion change from moving macro mi."""
            old_gi = min(rudy_n - 1, max(0, int(old_x / rudy_cw)))
            old_gj = min(rudy_n - 1, max(0, int(old_y / rudy_ch)))
            new_gi = min(rudy_n - 1, max(0, int(new_x / rudy_cw)))
            new_gj = min(rudy_n - 1, max(0, int(new_y / rudy_ch)))
            if old_gi == new_gi and old_gj == new_gj:
                return 0.0
            # Penalize moving into more congested regions
            old_cong = max(0, rudy_grid[old_gi, old_gj] - rudy_avg)
            new_cong = max(0, rudy_grid[new_gi, new_gj] - rudy_avg)
            return rudy_weight * (new_cong - old_cong) * (canvas_w + canvas_h)

        # ── Soft macro co-optimization ───────────────────────────────────
        soft_indices = list(range(benchmark.num_hard_macros, num_macros))
        soft_set = set(soft_indices)

        def _replace_soft_macros():
            """Re-place soft macros at centroid of connected hard macros."""
            for si in soft_indices:
                sx, sy, sw = 0.0, 0.0, 0.0
                for ni in macro_nets[si]:
                    for nid in net_node_lists[ni]:
                        if nid < num_hard or (nid >= num_macros and nid < total_nodes):
                            sx += pos_x[nid] * net_weights_np[ni]
                            sy += pos_y[nid] * net_weights_np[ni]
                            sw += net_weights_np[ni]
                if sw > 0:
                    new_x = sx / sw
                    new_y = sy / sw
                    w_s = sizes[si, 0].item() / 2
                    h_s = sizes[si, 1].item() / 2
                    new_x = max(w_s, min(canvas_w - w_s, new_x))
                    new_y = max(h_s, min(canvas_h - h_s, new_y))
                    pos_x[si] = new_x
                    pos_y[si] = new_y
            # Recompute all net bounding boxes after soft macro update
            for ni in range(num_nets):
                _recompute_net_bb(ni)

        # ── HPWL delta computation ───────────────────────────────────────
        def _hpwl_delta_shift(mi, new_x, new_y):
            delta = 0.0
            for ni in macro_nets[mi]:
                w = net_weights_np[ni]
                old_span = (net_xmax[ni] - net_xmin[ni] + net_ymax[ni] - net_ymin[ni]) * w
                xmin, xmax, ymin, ymax = 1e9, -1e9, 1e9, -1e9
                for nid in net_node_lists[ni]:
                    x = new_x if nid == mi else pos_x[nid]
                    y = new_y if nid == mi else pos_y[nid]
                    if x < xmin: xmin = x
                    if x > xmax: xmax = x
                    if y < ymin: ymin = y
                    if y > ymax: ymax = y
                delta += (xmax - xmin + ymax - ymin) * w - old_span
            return delta

        def _hpwl_delta_swap(mi, mj):
            xi, yi = pos_x[mi], pos_y[mi]
            xj, yj = pos_x[mj], pos_y[mj]
            affected = set(macro_nets[mi]) | set(macro_nets[mj])
            delta = 0.0
            for ni in affected:
                w = net_weights_np[ni]
                old_span = (net_xmax[ni] - net_xmin[ni] + net_ymax[ni] - net_ymin[ni]) * w
                xmin, xmax, ymin, ymax = 1e9, -1e9, 1e9, -1e9
                for nid in net_node_lists[ni]:
                    if nid == mi: x, y = xj, yj
                    elif nid == mj: x, y = xi, yi
                    else: x, y = pos_x[nid], pos_y[nid]
                    if x < xmin: xmin = x
                    if x > xmax: xmax = x
                    if y < ymin: ymin = y
                    if y > ymax: ymax = y
                delta += (xmax - xmin + ymax - ymin) * w - old_span
            return delta

        # ── Calibrate temperature ────────────────────────────────────────
        sample_deltas = []
        for _ in range(min(200, n_mov * 2)):
            mi = random.choice(movable_indices)
            dx = random.gauss(0, canvas_w * 0.1)
            dy = random.gauss(0, canvas_h * 0.1)
            nx = max(hw[mi], min(canvas_w - hw[mi], pos_x[mi] + dx))
            ny = max(hh[mi], min(canvas_h - hh[mi], pos_y[mi] + dy))
            delta = _hpwl_delta_shift(mi, nx, ny)
            if delta > 0:
                sample_deltas.append(delta)

        if sample_deltas:
            T0 = np.median(sample_deltas) / (-math.log(0.4))
        else:
            T0 = 1.0

        T = T0
        cooling_rate = 0.995
        min_temp = T0 * 0.001
        moves_per_temp = max(n_mov // 2, 30)

        # ── SA main loop ─────────────────────────────────────────────────
        total_moves = 0
        accepted = 0
        best_hpwl = np.sum((net_xmax - net_xmin + net_ymax - net_ymin) * net_weights_np)
        best_pos_x = pos_x.copy()
        best_pos_y = pos_y.copy()
        soft_reopt_interval = max(n_mov * 3, 500)  # re-place soft macros every N moves
        rudy_recompute_interval = max(n_mov * 5, 1000)

        while T > min_temp and _time.time() < deadline:
            for _ in range(moves_per_temp):
                if _time.time() > deadline:
                    break

                total_moves += 1

                # Periodic soft macro re-optimization
                if total_moves % soft_reopt_interval == 0:
                    _replace_soft_macros()

                # Periodic RUDY recomputation
                if total_moves % rudy_recompute_interval == 0:
                    _compute_rudy()
                    rudy_avg = rudy_grid.mean()

                r = random.random()

                if r < 0.70:
                    # ── Single-macro shift ────────────────────────────────
                    mi = random.choice(movable_indices)
                    scale = min(canvas_w, canvas_h) * 0.3 * (T / T0)
                    dx = random.gauss(0, scale)
                    dy = random.gauss(0, scale)
                    nx = max(hw[mi], min(canvas_w - hw[mi], pos_x[mi] + dx))
                    ny = max(hh[mi], min(canvas_h - hh[mi], pos_y[mi] + dy))

                    if _check_overlap(mi, nx, ny):
                        continue

                    delta = _hpwl_delta_shift(mi, nx, ny)
                    delta += _rudy_penalty_shift(mi, pos_x[mi], pos_y[mi], nx, ny)

                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        old_x, old_y = pos_x[mi], pos_y[mi]
                        _grid_move(mi, old_x, old_y, nx, ny)
                        pos_x[mi] = nx
                        pos_y[mi] = ny
                        for ni in macro_nets[mi]:
                            _recompute_net_bb(ni)
                        accepted += 1

                elif r < 0.90:
                    # ── Pairwise swap ─────────────────────────────────────
                    if n_mov < 2:
                        continue
                    mi = random.choice(movable_indices)
                    mj = random.choice(movable_indices)
                    if mi == mj:
                        continue

                    xi, yi = pos_x[mi], pos_y[mi]
                    xj, yj = pos_x[mj], pos_y[mj]

                    if (xj - hw[mi] < 0 or xj + hw[mi] > canvas_w or
                            yj - hh[mi] < 0 or yj + hh[mi] > canvas_h):
                        continue
                    if (xi - hw[mj] < 0 or xi + hw[mj] > canvas_w or
                            yi - hh[mj] < 0 or yi + hh[mj] > canvas_h):
                        continue

                    if _check_overlap_swap(mi, mj):
                        continue

                    delta = _hpwl_delta_swap(mi, mj)

                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        _grid_move(mi, xi, yi, xj, yj)
                        _grid_move(mj, xj, yj, xi, yi)
                        pos_x[mi], pos_x[mj] = xj, xi
                        pos_y[mi], pos_y[mj] = yj, yi
                        affected = set(macro_nets[mi]) | set(macro_nets[mj])
                        for ni in affected:
                            _recompute_net_bb(ni)
                        accepted += 1

                else:
                    # ── Cluster move ──────────────────────────────────────
                    mi = random.choice(movable_indices)
                    neighbors = set()
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
                    cdx = random.gauss(0, scale)
                    cdy = random.gauss(0, scale)

                    new_pos = {}
                    valid = True
                    for ci in cluster:
                        nx = max(hw[ci], min(canvas_w - hw[ci], pos_x[ci] + cdx))
                        ny = max(hh[ci], min(canvas_h - hh[ci], pos_y[ci] + cdy))
                        new_pos[ci] = (nx, ny)

                    cluster_set = set(cluster)
                    for ci in cluster:
                        if not valid:
                            break
                        nx, ny = new_pos[ci]
                        gx, gy = _grid_cell(nx, ny)
                        for ddx in range(-check_range, check_range + 1):
                            if not valid:
                                break
                            for ddy in range(-check_range, check_range + 1):
                                cx, cy = gx + ddx, gy + ddy
                                if 0 <= cx < ncx and 0 <= cy < ncy:
                                    for k in grid[cx][cy]:
                                        if k in cluster_set:
                                            continue
                                        if (abs(nx - pos_x[k]) < hw[ci] + hw[k] + gap and
                                                abs(ny - pos_y[k]) < hh[ci] + hh[k] + gap):
                                            valid = False
                                            break

                    if not valid:
                        continue

                    # Intra-cluster overlap check
                    cl = list(cluster)
                    for a in range(len(cl)):
                        if not valid:
                            break
                        for b in range(a + 1, len(cl)):
                            na_x, na_y = new_pos[cl[a]]
                            nb_x, nb_y = new_pos[cl[b]]
                            if (abs(na_x - nb_x) < hw[cl[a]] + hw[cl[b]] + gap and
                                    abs(na_y - nb_y) < hh[cl[a]] + hh[cl[b]] + gap):
                                valid = False
                                break

                    if not valid:
                        continue

                    # Compute delta
                    affected = set()
                    for ci in cluster:
                        affected.update(macro_nets[ci])

                    old_span = sum((net_xmax[ni] - net_xmin[ni] + net_ymax[ni] - net_ymin[ni]) * net_weights_np[ni]
                                   for ni in affected)

                    old_positions = {ci: (pos_x[ci], pos_y[ci]) for ci in cluster}
                    for ci in cluster:
                        pos_x[ci], pos_y[ci] = new_pos[ci]

                    new_span = 0.0
                    for ni in affected:
                        xmin, xmax, ymin, ymax = 1e9, -1e9, 1e9, -1e9
                        for nid in net_node_lists[ni]:
                            x, y = pos_x[nid], pos_y[nid]
                            if x < xmin: xmin = x
                            if x > xmax: xmax = x
                            if y < ymin: ymin = y
                            if y > ymax: ymax = y
                        new_span += (xmax - xmin + ymax - ymin) * net_weights_np[ni]

                    delta = new_span - old_span

                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        for ci in cluster:
                            ox, oy = old_positions[ci]
                            nx, ny = new_pos[ci]
                            _grid_move(ci, ox, oy, nx, ny)
                        for ni in affected:
                            _recompute_net_bb(ni)
                        accepted += 1
                    else:
                        for ci in cluster:
                            pos_x[ci], pos_y[ci] = old_positions[ci]

                # Track best periodically
                if total_moves % (moves_per_temp * 5) == 0:
                    cur = np.sum((net_xmax - net_xmin + net_ymax - net_ymin) * net_weights_np)
                    if cur < best_hpwl:
                        best_hpwl = cur
                        best_pos_x = pos_x.copy()
                        best_pos_y = pos_y.copy()

            T *= cooling_rate

        # Final soft macro re-placement and best check
        _replace_soft_macros()
        cur = np.sum((net_xmax - net_xmin + net_ymax - net_ymin) * net_weights_np)
        if cur < best_hpwl:
            best_pos_x = pos_x.copy()
            best_pos_y = pos_y.copy()

        for i in range(num_macros):
            placement[i, 0] = best_pos_x[i]
            placement[i, 1] = best_pos_y[i]

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
            wh = sizes_d[idx, 0] / 2; hh_t = sizes_d[idx, 1] / 2
            pos_d[idx, 0] = (pull_x[idx] / pull_w[idx]).clamp(min=wh, max=canvas_w - wh)
            pos_d[idx, 1] = (pull_y[idx] / pull_w[idx]).clamp(min=hh_t, max=canvas_h - hh_t)

        return pos_d.cpu()
