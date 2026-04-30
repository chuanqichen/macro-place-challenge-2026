"""
Phase 1: Spectral Placer — Quadratic Placement + GPU-Accelerated Refinement

Algorithm:
1. Build weighted Laplacian from netlist (clique/star model)
2. Solve quadratic placement with strong spreading (density-aware anchors)
3. GPU-accelerated greedy legalization (largest-first, minimum displacement)
4. GPU-accelerated iterative swap + shift refinement
5. Vectorized soft macro placement at connected-node centroids

Key insight: use aggressive spreading in the analytical phase so that
legalization displacement is small, preserving wirelength quality.

Usage:
    uv run evaluate submissions/phase1_spectral_placer.py -b ibm01
    uv run evaluate submissions/phase1_spectral_placer.py --all
"""

import time as _time

import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from macro_place.benchmark import Benchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpectralPlacer:

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = _time.time()
        placement = benchmark.macro_positions.clone()
        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]

        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable_mask)[0].tolist()
        movable_set = set(movable_indices)
        num_movable = len(movable_indices)

        if num_movable == 0:
            return placement

        idx_to_dense = {orig: dense for dense, orig in enumerate(movable_indices)}

        # ── Step 1: Build base Laplacian ─────────────────────────────────
        row_e, col_e, val_e = [], [], []
        rhs_x = np.zeros(num_movable)
        rhs_y = np.zeros(num_movable)

        def _get_pos(nid):
            if nid < num_macros:
                return placement[nid, 0].item(), placement[nid, 1].item()
            pi = nid - num_macros
            return (benchmark.port_positions[pi, 0].item(),
                    benchmark.port_positions[pi, 1].item())

        for net_idx in range(len(benchmark.net_nodes)):
            nodes = benchmark.net_nodes[net_idx].tolist()
            nw = benchmark.net_weights[net_idx].item()
            if len(nodes) < 2:
                continue
            deg = len(nodes)
            w = nw / (deg - 1)

            mov_in = [n for n in nodes if n in movable_set]
            fix_in = [n for n in nodes if n not in movable_set]

            # Clique model (exact for small nets, good enough for larger)
            for ip in range(len(mov_in)):
                di = idx_to_dense[mov_in[ip]]
                for jp in range(ip + 1, len(mov_in)):
                    dj = idx_to_dense[mov_in[jp]]
                    row_e.extend([di, dj, di, dj])
                    col_e.extend([dj, di, di, dj])
                    val_e.extend([-w, -w, w, w])
                for nf in fix_in:
                    fx, fy = _get_pos(nf)
                    row_e.append(di); col_e.append(di); val_e.append(w)
                    rhs_x[di] += w * fx
                    rhs_y[di] += w * fy

        if not row_e:
            return placement

        A_base = sparse.coo_matrix(
            (val_e, (row_e, col_e)), shape=(num_movable, num_movable)
        ).tocsr()

        # Regularization
        reg = 1e-6
        A_base += sparse.eye(num_movable) * reg
        rhs_x += reg * canvas_w / 2
        rhs_y += reg * canvas_h / 2

        # ── Step 2: Solve with progressive spreading ─────────────────────
        # Start with pure wirelength, then add increasingly strong spreading
        # to push macros apart before legalization.

        # Initial solve (pure wirelength)
        opt_x = spsolve(A_base, rhs_x)
        opt_y = spsolve(A_base, rhs_y)
        self._apply(placement, movable_indices, opt_x, opt_y, sizes, canvas_w, canvas_h)

        # Progressive spreading: 8 rounds with increasing force
        grid_n = 8
        for it in range(8):
            alpha = 0.02 * (1.5 ** it)  # stronger spreading: 0.02, 0.03, 0.045, ...
            A_s = A_base.copy()
            rx_s = rhs_x.copy()
            ry_s = rhs_y.copy()

            # Compute density on grid
            cell_w = canvas_w / grid_n
            cell_h = canvas_h / grid_n
            density = np.zeros((grid_n, grid_n))
            for idx in movable_indices:
                gi = min(grid_n - 1, max(0, int(placement[idx, 0].item() / cell_w)))
                gj = min(grid_n - 1, max(0, int(placement[idx, 1].item() / cell_h)))
                density[gi, gj] += sizes[idx, 0].item() * sizes[idx, 1].item()

            avg_d = max(density.sum() / (grid_n * grid_n), 1e-9)

            # For each macro in a dense cell, add a force pulling it toward
            # the nearest under-utilized cell
            sparse_centers = []
            for ti in range(grid_n):
                for tj in range(grid_n):
                    if density[ti, tj] < avg_d * 0.8:
                        sparse_centers.append(
                            (cell_w * (ti + 0.5), cell_h * (tj + 0.5),
                             avg_d - density[ti, tj])
                        )

            for i, idx in enumerate(movable_indices):
                cx = placement[idx, 0].item()
                cy = placement[idx, 1].item()
                gi = min(grid_n - 1, max(0, int(cx / cell_w)))
                gj = min(grid_n - 1, max(0, int(cy / cell_h)))
                ld = density[gi, gj]

                if ld <= avg_d * 1.1:
                    continue  # Not dense enough to push

                # Find best target: nearest sparse cell weighted by capacity
                best_score = -1
                tx, ty = cx, cy
                for scx, scy, cap in sparse_centers:
                    dist = ((scx - cx) ** 2 + (scy - cy) ** 2) ** 0.5
                    score = cap / max(dist, 0.1)
                    if score > best_score:
                        best_score = score
                        tx, ty = scx, scy

                force = alpha * (ld / avg_d)
                A_s[i, i] += force
                rx_s[i] += force * tx
                ry_s[i] += force * ty

            opt_x = spsolve(A_s, rx_s)
            opt_y = spsolve(A_s, ry_s)
            self._apply(placement, movable_indices, opt_x, opt_y, sizes, canvas_w, canvas_h)

        # ── Step 3: GPU-accelerated legalization ─────────────────────────
        placement = self._legalize_gpu(
            placement, movable_indices, sizes, canvas_w, canvas_h, num_hard
        )

        # ── Step 4: GPU-accelerated swap + shift refinement ──────────────
        time_left = max(5.0, 30.0 - (_time.time() - t0))
        placement = self._refine_gpu(
            benchmark, placement, movable_indices, sizes,
            canvas_w, canvas_h, num_hard, time_left,
        )

        # ── Step 5: Soft macro placement ─────────────────────────────────
        placement = self._place_soft_macros(benchmark, placement, sizes, canvas_w, canvas_h)

        return placement

    @staticmethod
    def _apply(placement, movable_indices, opt_x, opt_y, sizes, cw, ch):
        for i, idx in enumerate(movable_indices):
            wh = sizes[idx, 0].item() / 2
            hh = sizes[idx, 1].item() / 2
            placement[idx, 0] = max(wh, min(cw - wh, opt_x[i]))
            placement[idx, 1] = max(hh, min(ch - hh, opt_y[i]))

    # ── GPU legalization ─────────────────────────────────────────────────

    def _legalize_gpu(self, placement, movable_indices, sizes, canvas_w, canvas_h, num_hard):
        gap = 0.01
        dev = DEVICE

        pos_gpu = placement[:num_hard].clone().to(dev)
        sizes_gpu = sizes[:num_hard].to(dev)
        half_sz = sizes_gpu / 2

        targets = {i: (placement[i, 0].item(), placement[i, 1].item()) for i in movable_indices}

        # Sort by area descending
        sorted_idx = sorted(movable_indices, key=lambda i: -(sizes[i, 0].item() * sizes[i, 1].item()))

        movable_set = set(movable_indices)
        placed_mask = torch.zeros(num_hard, dtype=torch.bool, device=dev)
        for j in range(num_hard):
            if j not in movable_set:
                placed_mask[j] = True

        # Precompute spiral offsets (shared across all macros)
        offsets_list = []
        for dx_f, dy_f in [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (0.5, 1), (0.5, -1), (-0.5, 1), (-0.5, -1),
            (1, 0.5), (1, -0.5), (-1, 0.5), (-1, -0.5),
            (0.3, 0.7), (-0.3, 0.7), (0.3, -0.7), (-0.3, -0.7),
            (0.7, 0.3), (-0.7, 0.3), (0.7, -0.3), (-0.7, -0.3),
        ]:
            offsets_list.append([dx_f, dy_f])
        offsets_t = torch.tensor(offsets_list, device=dev)  # [24, 2]

        for idx in sorted_idx:
            wi = sizes_gpu[idx, 0].item()
            hi = sizes_gpu[idx, 1].item()
            cx, cy = targets[idx]
            cx = max(wi / 2, min(canvas_w - wi / 2, cx))
            cy = max(hi / 2, min(canvas_h - hi / 2, cy))

            if not self._has_overlap(pos_gpu, half_sz, placed_mask, cx, cy, wi, hi, gap):
                pos_gpu[idx] = torch.tensor([cx, cy], device=dev)
                placed_mask[idx] = True
                continue

            # Spiral search with batch overlap check
            step = max(min(wi, hi) * 0.25, 0.1)
            found = False
            for rm in range(1, 500):
                r = step * rm
                if r > max(canvas_w, canvas_h):
                    break

                cands = torch.tensor([cx, cy], device=dev).unsqueeze(0) + offsets_t * r  # [24, 2]
                cands[:, 0].clamp_(wi / 2, canvas_w - wi / 2)
                cands[:, 1].clamp_(hi / 2, canvas_h - hi / 2)

                free = self._batch_check(pos_gpu, half_sz, placed_mask, cands, wi, hi, gap)
                if free.any():
                    dists = (cands[free, 0] - cx) ** 2 + (cands[free, 1] - cy) ** 2
                    best = dists.argmin()
                    bi = torch.where(free)[0][best]
                    pos_gpu[idx] = cands[bi]
                    placed_mask[idx] = True
                    found = True
                    break

            if not found:
                pos_gpu[idx] = torch.tensor([cx, cy], device=dev)
                placed_mask[idx] = True

        # Write back
        placement[:num_hard] = pos_gpu.cpu()
        return placement

    @staticmethod
    def _has_overlap(pos, hsz, mask, cx, cy, wi, hi, gap):
        if not mask.any():
            return False
        pi = torch.where(mask)[0]
        dx = (pos[pi, 0] - cx).abs()
        dy = (pos[pi, 1] - cy).abs()
        return ((dx < hsz[pi, 0] + wi / 2 + gap) & (dy < hsz[pi, 1] + hi / 2 + gap)).any().item()

    @staticmethod
    def _batch_check(pos, hsz, mask, cands, wi, hi, gap):
        """cands: [C, 2]. Returns [C] bool — True if free."""
        if not mask.any():
            return torch.ones(cands.shape[0], dtype=torch.bool, device=cands.device)
        pi = torch.where(mask)[0]
        px = pos[pi, 0]  # [P]
        py = pos[pi, 1]
        phw = hsz[pi, 0]
        phh = hsz[pi, 1]
        dx = (cands[:, 0].unsqueeze(1) - px.unsqueeze(0)).abs()  # [C, P]
        dy = (cands[:, 1].unsqueeze(1) - py.unsqueeze(0)).abs()
        ov = (dx < phw.unsqueeze(0) + wi / 2 + gap) & (dy < phh.unsqueeze(0) + hi / 2 + gap)
        return ~ov.any(dim=1)

    # ── GPU swap + single-macro shift refinement ─────────────────────────

    def _refine_gpu(self, benchmark, placement, movable_indices, sizes,
                    canvas_w, canvas_h, num_hard, time_budget):
        dev = DEVICE
        gap = 0.01
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        deadline = _time.time() + time_budget

        total_nodes = num_macros + num_ports
        all_pos = torch.zeros(total_nodes, 2, device=dev)
        all_pos[:num_macros] = placement.to(dev)
        if num_ports > 0:
            all_pos[num_macros:] = benchmark.port_positions.to(dev)

        sizes_gpu = sizes[:num_hard].to(dev)
        half_sz = sizes_gpu / 2

        # Per-macro net membership
        macro_nets = [[] for _ in range(num_macros)]
        for ni in range(len(benchmark.net_nodes)):
            for nid in benchmark.net_nodes[ni].tolist():
                if nid < num_macros:
                    macro_nets[nid].append(ni)

        def _affected_hpwl(mi, mj=-1):
            nets = set(macro_nets[mi])
            if mj >= 0:
                nets |= set(macro_nets[mj])
            total = 0.0
            for ni in nets:
                nodes = benchmark.net_nodes[ni]
                valid = nodes[nodes < total_nodes].to(dev)
                if len(valid) < 2:
                    continue
                p = all_pos[valid]
                total += (p.max(0).values - p.min(0).values).sum().item()
            return total, nets

        mov = movable_indices
        n_mov = len(mov)

        # ── Phase A: Pairwise swaps ──────────────────────────────────────
        for _pass in range(2):
            if _time.time() > deadline:
                break
            any_swap = False
            for ip in range(n_mov):
                if _time.time() > deadline:
                    break
                mi = mov[ip]
                xi, yi = all_pos[mi, 0].item(), all_pos[mi, 1].item()
                wi_h, hi_h = half_sz[mi, 0].item(), half_sz[mi, 1].item()

                for jp in range(ip + 1, n_mov):
                    mj = mov[jp]
                    xj, yj = all_pos[mj, 0].item(), all_pos[mj, 1].item()

                    if abs(xi - xj) > canvas_w * 0.25 or abs(yi - yj) > canvas_h * 0.25:
                        continue

                    wj_h, hj_h = half_sz[mj, 0].item(), half_sz[mj, 1].item()

                    # Bounds check
                    if (xj - wi_h < -0.001 or xj + wi_h > canvas_w + 0.001 or
                            yj - hi_h < -0.001 or yj + hi_h > canvas_h + 0.001):
                        continue
                    if (xi - wj_h < -0.001 or xi + wj_h > canvas_w + 0.001 or
                            yi - hj_h < -0.001 or yi + hj_h > canvas_h + 0.001):
                        continue

                    # Vectorized overlap check
                    exc = torch.ones(num_hard, dtype=torch.bool, device=dev)
                    exc[mi] = False; exc[mj] = False
                    others = torch.where(exc)[0]

                    if len(others) > 0:
                        po = all_pos[others]
                        hs = half_sz[others]
                        # mi at (xj, yj)
                        if ((po[:, 0] - xj).abs() < hs[:, 0] + wi_h + gap).any() and \
                           (((po[:, 0] - xj).abs() < hs[:, 0] + wi_h + gap) &
                            ((po[:, 1] - yj).abs() < hs[:, 1] + hi_h + gap)).any():
                            continue
                        # mj at (xi, yi)
                        if (((po[:, 0] - xi).abs() < hs[:, 0] + wj_h + gap) &
                            ((po[:, 1] - yi).abs() < hs[:, 1] + hj_h + gap)).any():
                            continue

                    old_h, nets = _affected_hpwl(mi, mj)
                    all_pos[mi, 0], all_pos[mj, 0] = xj, xi
                    all_pos[mi, 1], all_pos[mj, 1] = yj, yi
                    new_h, _ = _affected_hpwl(mi, mj)

                    if new_h < old_h - 1e-6:
                        any_swap = True
                        xi, yi = xj, yj  # update for continued inner loop
                    else:
                        all_pos[mi, 0], all_pos[mj, 0] = xi, xj
                        all_pos[mi, 1], all_pos[mj, 1] = yi, yj

            if not any_swap:
                break

        # ── Phase B: Single-macro shifts ─────────────────────────────────
        # Try small displacements for each macro to reduce its net HPWL
        shift_offsets = []
        for d in [0.5, 1.0, 2.0]:
            for dx, dy in [(d, 0), (-d, 0), (0, d), (0, -d),
                           (d, d), (d, -d), (-d, d), (-d, -d)]:
                shift_offsets.append((dx, dy))

        for _pass in range(2):
            if _time.time() > deadline:
                break
            any_shift = False
            for mi in mov:
                if _time.time() > deadline:
                    break
                ox = all_pos[mi, 0].item()
                oy = all_pos[mi, 1].item()
                wi_h = half_sz[mi, 0].item()
                hi_h = half_sz[mi, 1].item()

                old_h, _ = _affected_hpwl(mi)
                best_h = old_h
                best_x, best_y = ox, oy

                for dx, dy in shift_offsets:
                    nx = ox + dx
                    ny = oy + dy
                    if nx - wi_h < 0 or nx + wi_h > canvas_w:
                        continue
                    if ny - hi_h < 0 or ny + hi_h > canvas_h:
                        continue

                    # Quick overlap check
                    exc = torch.ones(num_hard, dtype=torch.bool, device=dev)
                    exc[mi] = False
                    others = torch.where(exc)[0]
                    if len(others) > 0:
                        po = all_pos[others]
                        hs = half_sz[others]
                        if (((po[:, 0] - nx).abs() < hs[:, 0] + wi_h + gap) &
                                ((po[:, 1] - ny).abs() < hs[:, 1] + hi_h + gap)).any():
                            continue

                    all_pos[mi, 0] = nx
                    all_pos[mi, 1] = ny
                    new_h, _ = _affected_hpwl(mi)
                    if new_h < best_h - 1e-6:
                        best_h = new_h
                        best_x, best_y = nx, ny

                if best_x != ox or best_y != oy:
                    all_pos[mi, 0] = best_x
                    all_pos[mi, 1] = best_y
                    any_shift = True
                else:
                    all_pos[mi, 0] = ox
                    all_pos[mi, 1] = oy

            if not any_shift:
                break

        placement[:num_macros] = all_pos[:num_macros].cpu()
        return placement

    # ── Soft macro placement ─────────────────────────────────────────────

    def _place_soft_macros(self, benchmark, placement, sizes, canvas_w, canvas_h):
        num_macros = benchmark.num_macros
        num_hard = benchmark.num_hard_macros
        num_ports = benchmark.port_positions.shape[0]
        soft_indices = list(range(num_hard, num_macros))
        if not soft_indices:
            return placement

        soft_set = set(soft_indices)
        dev = DEVICE
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

            cx = torch.stack(ax).mean()
            cy = torch.stack(ay).mean()
            nw = w / max(1, len(ax))
            for s in ns:
                pull_x[s] += cx * nw
                pull_y[s] += cy * nw
                pull_w[s] += nw

        valid = pull_w > 0
        sm = torch.zeros(num_macros, dtype=torch.bool, device=dev)
        for s in soft_indices:
            sm[s] = True
        upd = valid & sm

        if upd.any():
            idx = torch.where(upd)[0]
            sizes_d = sizes.to(dev)
            wh = sizes_d[idx, 0] / 2
            hh = sizes_d[idx, 1] / 2
            pos_d[idx, 0] = (pull_x[idx] / pull_w[idx]).clamp(min=wh, max=canvas_w - wh)
            pos_d[idx, 1] = (pull_y[idx] / pull_w[idx]).clamp(min=hh, max=canvas_h - hh)

        placement = pos_d.cpu()
        return placement
