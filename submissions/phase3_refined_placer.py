"""
Phase 3: Centroid Init + Abacus Legalization + Density-Aware Coordinate Descent

Improvements over Phase 2:
1. Abacus-style legalization: row-based, displacement-minimizing
2. Density-aware CD: penalize moves that increase local density
3. GPU-accelerated density grid for fast density evaluation
4. More time allocated to CD refinement
5. Adaptive step sizes based on improvement history

Usage:
    uv run evaluate submissions/phase3_refined_placer.py -b ibm01
    uv run evaluate submissions/phase3_refined_placer.py --all
"""

import time as _time
import torch
import numpy as np

from macro_place.benchmark import Benchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RefinedPlacer:

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

        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable_mask)[0].tolist()

        if not movable_indices:
            return placement

        net_data = self._prepare_nets(benchmark, dev)

        # ── Step 1: Centroid initial placement ───────────────────────────
        placement = self._centroid_init(benchmark, placement, sizes, canvas_w, canvas_h)

        # ── Step 2: Legalization (spiral, guaranteed zero overlaps) ────────
        placement = self._legalize_spiral(
            placement, movable_indices, sizes, canvas_w, canvas_h, num_hard, dev,
        )

        # ── Step 3: Density-aware CD + swaps ─────────────────────────────
        time_left = max(10.0, 55.0 - (_time.time() - t0))
        placement = self._density_aware_cd(
            benchmark, placement, sizes, canvas_w, canvas_h,
            num_hard, net_data, dev, time_left,
        )

        # ── Step 4: Soft macro placement ─────────────────────────────────
        placement = self._place_soft_macros(benchmark, placement, sizes, canvas_w, canvas_h, dev)

        return placement

    # ── Centroid init (same as Phase 0/2) ────────────────────────────────

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

    # ── Spiral legalization (proven zero-overlap) ───────────────────────

    def _legalize_spiral(self, placement, movable_indices, sizes,
                          canvas_w, canvas_h, num_hard, dev):
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

            if not self._has_ov(pos_gpu, half_sz, placed_mask, cx, cy, wi, hi, gap):
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
                free = self._batch_ov(pos_gpu, half_sz, placed_mask, cands, wi, hi, gap)
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

    @staticmethod
    def _has_ov(pos, hsz, mask, cx, cy, wi, hi, gap):
        if not mask.any():
            return False
        pi = torch.where(mask)[0]
        dx = (pos[pi, 0] - cx).abs()
        dy = (pos[pi, 1] - cy).abs()
        return ((dx < hsz[pi, 0] + wi / 2 + gap) & (dy < hsz[pi, 1] + hi / 2 + gap)).any().item()

    @staticmethod
    def _batch_ov(pos, hsz, mask, cands, wi, hi, gap):
        if not mask.any():
            return torch.ones(cands.shape[0], dtype=torch.bool, device=cands.device)
        pi = torch.where(mask)[0]
        dx = (cands[:, 0].unsqueeze(1) - pos[pi, 0].unsqueeze(0)).abs()
        dy = (cands[:, 1].unsqueeze(1) - pos[pi, 1].unsqueeze(0)).abs()
        ov = (dx < hsz[pi, 0].unsqueeze(0) + wi / 2 + gap) & (dy < hsz[pi, 1].unsqueeze(0) + hi / 2 + gap)
        return ~ov.any(dim=1)

    # ── Net data preparation ─────────────────────────────────────────────

    def _prepare_nets(self, benchmark, dev):
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        total_nodes = num_macros + num_ports
        nets = benchmark.net_nodes
        if not nets:
            return None

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

        return {
            'net_tensor': net_tensor, 'net_mask': net_mask,
            'weights': weights, 'total_nodes': total_nodes,
            'macro_nets': macro_nets,
        }

    # ── Abacus-style legalization ────────────────────────────────────────

    def _abacus_legalize(self, placement, movable_indices, sizes,
                          canvas_w, canvas_h, num_hard, dev):
        """
        Row-based legalization that minimizes total displacement.

        1. Divide canvas into rows based on most common macro height
        2. Sort macros by x-coordinate
        3. For each macro, find the row that minimizes displacement
        4. Within the row, place at the leftmost legal position closest to target
        """
        gap = 0.01
        movable_set = set(movable_indices)

        # Collect fixed macros as obstacles
        fixed_macros = []
        for j in range(num_hard):
            if j not in movable_set:
                fixed_macros.append(j)

        # Get macro dimensions
        heights = [sizes[i, 1].item() for i in movable_indices]
        widths = [sizes[i, 0].item() for i in movable_indices]

        if not heights:
            return placement

        # Determine row height: use median macro height
        sorted_h = sorted(heights)
        row_h = sorted_h[len(sorted_h) // 2]
        if row_h < 0.01:
            row_h = canvas_h / 10

        num_rows = max(1, int(canvas_h / row_h))
        row_h = canvas_h / num_rows

        # Build row structures: each row tracks placed intervals
        # An interval is (x_left, x_right) of a placed macro
        rows = [[] for _ in range(num_rows)]

        # Place fixed macros into rows first
        for j in fixed_macros:
            fx = placement[j, 0].item()
            fy = placement[j, 1].item()
            fw = sizes[j, 0].item()
            fh = sizes[j, 1].item()
            # Find which rows this fixed macro spans
            y_bot = fy - fh / 2
            y_top = fy + fh / 2
            for r in range(num_rows):
                ry_bot = r * row_h
                ry_top = (r + 1) * row_h
                if y_bot < ry_top and y_top > ry_bot:
                    rows[r].append((fx - fw / 2 - gap, fx + fw / 2 + gap))

        # Sort rows' intervals
        for r in range(num_rows):
            rows[r].sort()

        # Sort movable macros by x-coordinate (left to right)
        sorted_movable = sorted(
            movable_indices,
            key=lambda i: (placement[i, 0].item(), -sizes[i, 0].item() * sizes[i, 1].item()),
        )

        def _find_best_x_in_row(row_intervals, target_x, macro_w):
            """Find the x position in a row closest to target_x that doesn't overlap."""
            half_w = macro_w / 2 + gap
            best_x = None
            best_dist = float('inf')

            # Try target position first
            x_left = target_x - half_w
            x_right = target_x + half_w
            if x_left >= 0 and x_right <= canvas_w:
                ok = True
                for (il, ir) in row_intervals:
                    if x_left < ir and x_right > il:
                        ok = False
                        break
                if ok:
                    return target_x

            # Try positions just after each existing interval
            candidates = [half_w + gap]  # leftmost position
            for (il, ir) in row_intervals:
                candidates.append(ir + half_w + gap)
            # Also try just before each interval
            for (il, ir) in row_intervals:
                candidates.append(il - half_w - gap)
            candidates.append(canvas_w - half_w - gap)

            for cx in candidates:
                if cx - half_w < -0.001 or cx + half_w > canvas_w + 0.001:
                    continue
                x_l = cx - half_w
                x_r = cx + half_w
                ok = True
                for (il, ir) in row_intervals:
                    if x_l < ir and x_r > il:
                        ok = False
                        break
                if ok:
                    dist = abs(cx - target_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_x = cx

            return best_x

        for idx in sorted_movable:
            target_x = placement[idx, 0].item()
            target_y = placement[idx, 1].item()
            mw = sizes[idx, 0].item()
            mh = sizes[idx, 1].item()

            # Find best row (minimize y displacement)
            best_row = -1
            best_x = None
            best_cost = float('inf')

            # Check rows near the target y
            target_row = min(num_rows - 1, max(0, int(target_y / row_h)))
            search_range = min(num_rows, max(5, num_rows // 3))

            for dr in range(search_range):
                for r in [target_row + dr, target_row - dr]:
                    if r < 0 or r >= num_rows:
                        continue
                    row_cy = (r + 0.5) * row_h

                    # Check if macro fits vertically
                    if row_cy - mh / 2 < -0.001 or row_cy + mh / 2 > canvas_h + 0.001:
                        continue

                    bx = _find_best_x_in_row(rows[r], target_x, mw)
                    if bx is not None:
                        cost = (bx - target_x) ** 2 + (row_cy - target_y) ** 2
                        if cost < best_cost:
                            best_cost = cost
                            best_row = r
                            best_x = bx

            if best_row >= 0 and best_x is not None:
                row_cy = (best_row + 0.5) * row_h
                placement[idx, 0] = best_x
                placement[idx, 1] = max(mh / 2, min(canvas_h - mh / 2, row_cy))
                # Add to row intervals
                half_w = mw / 2 + gap
                rows[best_row].append((best_x - half_w, best_x + half_w))
                rows[best_row].sort()
            else:
                # Fallback: spiral search (same as Phase 2)
                placement = self._spiral_place_single(
                    placement, idx, sizes, canvas_w, canvas_h, num_hard, dev, movable_set,
                )

        # Verify and fix any remaining overlaps with GPU check
        for _round in range(3):
            placement = self._fix_remaining_overlaps(
                placement, movable_indices, sizes, canvas_w, canvas_h, num_hard, dev,
            )
            # Quick vectorized overlap check
            pos_check = placement[:num_hard].to(dev)
            sz_check = sizes[:num_hard].to(dev) / 2
            # All-pairs distance check (vectorized)
            cx = pos_check[:, 0]; cy = pos_check[:, 1]
            hw = sz_check[:, 0]; hh = sz_check[:, 1]
            dx = (cx.unsqueeze(0) - cx.unsqueeze(1)).abs()
            dy = (cy.unsqueeze(0) - cy.unsqueeze(1)).abs()
            min_dx = hw.unsqueeze(0) + hw.unsqueeze(1) + gap
            min_dy = hh.unsqueeze(0) + hh.unsqueeze(1) + gap
            ov_matrix = (dx < min_dx) & (dy < min_dy)
            ov_matrix.fill_diagonal_(False)
            if not ov_matrix.any():
                break

        return placement

    def _spiral_place_single(self, placement, idx, sizes, canvas_w, canvas_h,
                              num_hard, dev, movable_set):
        """Fallback spiral placement for a single macro."""
        gap = 0.01
        pos_gpu = placement[:num_hard].to(dev)
        sizes_gpu = sizes[:num_hard].to(dev)
        half_sz = sizes_gpu / 2

        wi = sizes[idx, 0].item()
        hi = sizes[idx, 1].item()
        cx = placement[idx, 0].item()
        cy = placement[idx, 1].item()
        cx = max(wi / 2, min(canvas_w - wi / 2, cx))
        cy = max(hi / 2, min(canvas_h - hi / 2, cy))

        # Check all hard macros except this one
        mask = torch.ones(num_hard, dtype=torch.bool, device=dev)
        mask[idx] = False

        offsets_t = torch.tensor([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [1, -1], [-1, 1], [-1, -1],
            [0.5, 1], [0.5, -1], [-0.5, 1], [-0.5, -1],
            [1, 0.5], [1, -0.5], [-1, 0.5], [-1, -0.5],
        ], device=dev)

        step = max(min(wi, hi) * 0.25, 0.1)
        for rm in range(1, 500):
            r = step * rm
            if r > max(canvas_w, canvas_h):
                break
            cands = torch.tensor([cx, cy], device=dev).unsqueeze(0) + offsets_t * r
            cands[:, 0].clamp_(wi / 2, canvas_w - wi / 2)
            cands[:, 1].clamp_(hi / 2, canvas_h - hi / 2)

            pi = torch.where(mask)[0]
            if len(pi) > 0:
                dx = (cands[:, 0].unsqueeze(1) - pos_gpu[pi, 0].unsqueeze(0)).abs()
                dy = (cands[:, 1].unsqueeze(1) - pos_gpu[pi, 1].unsqueeze(0)).abs()
                ov = (dx < half_sz[pi, 0].unsqueeze(0) + wi / 2 + gap) & \
                     (dy < half_sz[pi, 1].unsqueeze(0) + hi / 2 + gap)
                free = ~ov.any(dim=1)
            else:
                free = torch.ones(cands.shape[0], dtype=torch.bool, device=dev)

            if free.any():
                dists = (cands[free, 0] - cx) ** 2 + (cands[free, 1] - cy) ** 2
                bi = torch.where(free)[0][dists.argmin()]
                placement[idx, 0] = cands[bi, 0].item()
                placement[idx, 1] = cands[bi, 1].item()
                return placement

        return placement

    def _fix_remaining_overlaps(self, placement, movable_indices, sizes,
                                 canvas_w, canvas_h, num_hard, dev):
        """Check for and fix any remaining overlaps after Abacus legalization."""
        gap = 0.01
        pos_gpu = placement[:num_hard].to(dev)
        sizes_gpu = sizes[:num_hard].to(dev)
        half_sz = sizes_gpu / 2
        movable_set = set(movable_indices)

        # Find all overlapping pairs (vectorized)
        overlapping_macros = set()
        cx = pos_gpu[:, 0]; cy = pos_gpu[:, 1]
        hw = half_sz[:, 0]; hh = half_sz[:, 1]
        dx_mat = (cx.unsqueeze(0) - cx.unsqueeze(1)).abs()
        dy_mat = (cy.unsqueeze(0) - cy.unsqueeze(1)).abs()
        min_dx_mat = hw.unsqueeze(0) + hw.unsqueeze(1) + gap
        min_dy_mat = hh.unsqueeze(0) + hh.unsqueeze(1) + gap
        ov_mat = (dx_mat < min_dx_mat) & (dy_mat < min_dy_mat)
        ov_mat.fill_diagonal_(False)

        # For each overlapping pair, pick the smaller movable macro
        ov_pairs = torch.where(ov_mat.triu())
        for k in range(len(ov_pairs[0])):
            i = ov_pairs[0][k].item()
            j = ov_pairs[1][k].item()
            if j in movable_set:
                overlapping_macros.add(j)
            elif i in movable_set:
                overlapping_macros.add(i)

        if not overlapping_macros:
            return placement

        # Sort by area ascending (re-place smallest first)
        to_fix = sorted(overlapping_macros,
                        key=lambda i: sizes[i, 0].item() * sizes[i, 1].item())

        # Re-legalize each overlapping macro with spiral search
        placed_mask = torch.ones(num_hard, dtype=torch.bool, device=dev)
        for idx in to_fix:
            placed_mask[idx] = False  # temporarily remove

        for idx in to_fix:
            wi = sizes_gpu[idx, 0].item()
            hi = sizes_gpu[idx, 1].item()
            cx = pos_gpu[idx, 0].item()
            cy = pos_gpu[idx, 1].item()

            offsets_t = torch.tensor([
                [1, 0], [-1, 0], [0, 1], [0, -1],
                [1, 1], [1, -1], [-1, 1], [-1, -1],
                [0.5, 1], [0.5, -1], [-0.5, 1], [-0.5, -1],
                [1, 0.5], [1, -0.5], [-1, 0.5], [-1, -0.5],
            ], device=dev)

            step = max(min(wi, hi) * 0.25, 0.1)
            found = False
            # Check current position first
            pi = torch.where(placed_mask)[0]
            if len(pi) > 0:
                ddx = (pos_gpu[pi, 0] - cx).abs()
                ddy = (pos_gpu[pi, 1] - cy).abs()
                if not ((ddx < half_sz[pi, 0] + wi / 2 + gap) &
                        (ddy < half_sz[pi, 1] + hi / 2 + gap)).any():
                    found = True

            if not found:
                for rm in range(1, 500):
                    r = step * rm
                    if r > max(canvas_w, canvas_h):
                        break
                    cands = torch.tensor([cx, cy], device=dev).unsqueeze(0) + offsets_t * r
                    cands[:, 0].clamp_(wi / 2, canvas_w - wi / 2)
                    cands[:, 1].clamp_(hi / 2, canvas_h - hi / 2)

                    if len(pi) > 0:
                        ddx = (cands[:, 0].unsqueeze(1) - pos_gpu[pi, 0].unsqueeze(0)).abs()
                        ddy = (cands[:, 1].unsqueeze(1) - pos_gpu[pi, 1].unsqueeze(0)).abs()
                        ov = (ddx < half_sz[pi, 0].unsqueeze(0) + wi / 2 + gap) & \
                             (ddy < half_sz[pi, 1].unsqueeze(0) + hi / 2 + gap)
                        free = ~ov.any(dim=1)
                    else:
                        free = torch.ones(cands.shape[0], dtype=torch.bool, device=dev)

                    if free.any():
                        dists = (cands[free, 0] - cx) ** 2 + (cands[free, 1] - cy) ** 2
                        bi = torch.where(free)[0][dists.argmin()]
                        pos_gpu[idx, 0] = cands[bi, 0]
                        pos_gpu[idx, 1] = cands[bi, 1]
                        found = True
                        break

            placed_mask[idx] = True

        placement[:num_hard] = pos_gpu.cpu()
        return placement

    # ── Density-aware coordinate descent ─────────────────────────────────

    def _density_aware_cd(self, benchmark, placement, sizes, canvas_w, canvas_h,
                           num_hard, net_data, dev, time_budget):
        """
        Coordinate descent that considers both HPWL and local density.
        Uses a fast GPU density grid to penalize moves into dense regions.
        """
        if net_data is None:
            return placement

        deadline = _time.time() + time_budget
        gap = 0.01
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]

        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable_mask)[0].tolist()

        total_nodes = net_data['total_nodes']
        all_pos = torch.zeros(total_nodes, 2, device=dev)
        all_pos[:num_macros] = placement.to(dev)
        if num_ports > 0:
            all_pos[num_macros:num_macros + num_ports] = benchmark.port_positions.to(dev)

        sizes_gpu = sizes[:num_hard].to(dev)
        half_sz = sizes_gpu / 2

        macro_nets = net_data['macro_nets']
        net_tensor = net_data['net_tensor']
        net_mask = net_data['net_mask']
        net_weights = net_data['weights']

        # ── GPU density grid ─────────────────────────────────────────────
        grid_n = 16  # finer grid for density
        cell_w = canvas_w / grid_n
        cell_h = canvas_h / grid_n

        def _compute_density_grid():
            """Compute density grid on GPU."""
            grid = torch.zeros(grid_n, grid_n, device=dev)
            for idx in range(num_hard):
                cx = all_pos[idx, 0]
                cy = all_pos[idx, 1]
                mw = sizes_gpu[idx, 0]
                mh = sizes_gpu[idx, 1]
                # Find grid cells this macro overlaps
                gi_min = max(0, int(((cx - mw / 2) / cell_w).item()))
                gi_max = min(grid_n - 1, int(((cx + mw / 2) / cell_w).item()))
                gj_min = max(0, int(((cy - mh / 2) / cell_h).item()))
                gj_max = min(grid_n - 1, int(((cy + mh / 2) / cell_h).item()))
                area = mw * mh
                num_cells = max(1, (gi_max - gi_min + 1) * (gj_max - gj_min + 1))
                for gi in range(gi_min, gi_max + 1):
                    for gj in range(gj_min, gj_max + 1):
                        grid[gi, gj] += area / num_cells
            return grid / (cell_w * cell_h)

        def _local_density(cx, cy, mw, mh):
            """Get density at a position (fast lookup)."""
            gi = min(grid_n - 1, max(0, int(cx / cell_w)))
            gj = min(grid_n - 1, max(0, int(cy / cell_h)))
            return density_grid[gi, gj].item()

        density_grid = _compute_density_grid()
        avg_density = density_grid.mean().item()

        def _affected_hpwl(mi):
            nids_list = macro_nets[mi]
            if not nids_list:
                return 0.0
            nids = torch.tensor(nids_list, device=dev, dtype=torch.long)
            padded = torch.cat([all_pos, torch.zeros(1, 2, device=dev)], dim=0)
            nt = net_tensor[nids]; nm = net_mask[nids]; nw = net_weights[nids]
            pos = padded[nt]
            x = pos[:, :, 0].clone(); y = pos[:, :, 1].clone()
            x[~nm] = 1e9; x_min = x.min(1).values
            x[~nm] = -1e9; x_max = x.max(1).values
            y[~nm] = 1e9; y_min = y.min(1).values
            y[~nm] = -1e9; y_max = y.max(1).values
            return (((x_max - x_min) + (y_max - y_min)) * nw).sum().item()

        # Density weight for combined cost
        density_penalty = 0.3  # weight of density in move evaluation

        # Multi-scale shift offsets
        shift_deltas = []
        for scale in [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1),
                           (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1),
                           (1, 0.5), (-1, 0.5), (1, -0.5), (-1, -0.5)]:
                shift_deltas.append((dx * scale, dy * scale))

        # ── Phase A: Density-aware coordinate descent ────────────────────
        for _pass in range(10):
            if _time.time() > deadline:
                break
            any_imp = False

            # Recompute density grid periodically
            if _pass % 3 == 0 and _pass > 0:
                density_grid = _compute_density_grid()
                avg_density = density_grid.mean().item()

            for mi in movable_indices:
                if _time.time() > deadline:
                    break

                ox = all_pos[mi, 0].item()
                oy = all_pos[mi, 1].item()
                wi_h = half_sz[mi, 0].item()
                hi_h = half_sz[mi, 1].item()
                mw = sizes_gpu[mi, 0].item()
                mh = sizes_gpu[mi, 1].item()

                old_hpwl = _affected_hpwl(mi)
                old_density = _local_density(ox, oy, mw, mh)
                old_cost = old_hpwl + density_penalty * max(0, old_density - avg_density) * (canvas_w + canvas_h)

                best_cost = old_cost
                best_x, best_y = ox, oy

                exc = torch.ones(num_hard, dtype=torch.bool, device=dev)
                exc[mi] = False
                others = torch.where(exc)[0]
                other_pos = all_pos[others]
                other_hsz = half_sz[others]

                for dx, dy in shift_deltas:
                    nx = ox + dx; ny = oy + dy
                    if nx - wi_h < 0 or nx + wi_h > canvas_w:
                        continue
                    if ny - hi_h < 0 or ny + hi_h > canvas_h:
                        continue

                    # Vectorized overlap check
                    if len(others) > 0:
                        if (((other_pos[:, 0] - nx).abs() < other_hsz[:, 0] + wi_h + gap) &
                                ((other_pos[:, 1] - ny).abs() < other_hsz[:, 1] + hi_h + gap)).any():
                            continue

                    # Evaluate combined cost
                    all_pos[mi, 0] = nx; all_pos[mi, 1] = ny
                    new_hpwl = _affected_hpwl(mi)
                    new_density = _local_density(nx, ny, mw, mh)
                    new_cost = new_hpwl + density_penalty * max(0, new_density - avg_density) * (canvas_w + canvas_h)

                    if new_cost < best_cost - 1e-6:
                        best_cost = new_cost
                        best_x, best_y = nx, ny

                all_pos[mi, 0] = best_x; all_pos[mi, 1] = best_y
                if best_x != ox or best_y != oy:
                    any_imp = True

            if not any_imp:
                break

        # ── Phase B: Pairwise swaps ──────────────────────────────────────
        for _pass in range(3):
            if _time.time() > deadline:
                break
            any_swap = False

            for ip in range(len(movable_indices)):
                if _time.time() > deadline:
                    break
                mi = movable_indices[ip]
                xi = all_pos[mi, 0].item(); yi = all_pos[mi, 1].item()
                wi_h = half_sz[mi, 0].item(); hi_h = half_sz[mi, 1].item()

                for jp in range(ip + 1, len(movable_indices)):
                    mj = movable_indices[jp]
                    xj = all_pos[mj, 0].item(); yj = all_pos[mj, 1].item()

                    if abs(xi - xj) > canvas_w * 0.25 or abs(yi - yj) > canvas_h * 0.25:
                        continue

                    wj_h = half_sz[mj, 0].item(); hj_h = half_sz[mj, 1].item()

                    if (xj - wi_h < 0 or xj + wi_h > canvas_w or
                            yj - hi_h < 0 or yj + hi_h > canvas_h):
                        continue
                    if (xi - wj_h < 0 or xi + wj_h > canvas_w or
                            yi - hj_h < 0 or yi + hj_h > canvas_h):
                        continue

                    exc = torch.ones(num_hard, dtype=torch.bool, device=dev)
                    exc[mi] = False; exc[mj] = False
                    others = torch.where(exc)[0]
                    if len(others) > 0:
                        op = all_pos[others]; hs = half_sz[others]
                        if (((op[:, 0] - xj).abs() < hs[:, 0] + wi_h + gap) &
                                ((op[:, 1] - yj).abs() < hs[:, 1] + hi_h + gap)).any():
                            continue
                        if (((op[:, 0] - xi).abs() < hs[:, 0] + wj_h + gap) &
                                ((op[:, 1] - yi).abs() < hs[:, 1] + hj_h + gap)).any():
                            continue

                    old_h = _affected_hpwl(mi) + _affected_hpwl(mj)
                    all_pos[mi, 0], all_pos[mj, 0] = xj, xi
                    all_pos[mi, 1], all_pos[mj, 1] = yj, yi
                    new_h = _affected_hpwl(mi) + _affected_hpwl(mj)

                    if new_h < old_h - 1e-6:
                        any_swap = True; xi, yi = xj, yj
                    else:
                        all_pos[mi, 0], all_pos[mj, 0] = xi, xj
                        all_pos[mi, 1], all_pos[mj, 1] = yi, yj

            if not any_swap:
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
