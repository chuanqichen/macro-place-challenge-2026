"""
Phase 2: Hybrid Analytical Placer + Coordinate Descent Refinement

Strategy:
1. Quadratic initial placement (Laplacian solve) for wirelength-optimal positions
2. Strong density-aware spreading to reduce congestion
3. GPU-accelerated greedy legalization
4. GPU-accelerated coordinate descent: for each macro, try many nearby
   positions and keep the one minimizing HPWL of affected nets
5. Pairwise swap refinement
6. Soft macro placement at connected-node centroids

Key insight from Phase 0/1: congestion is the bottleneck. This version
uses stronger spreading and allocates more time to coordinate descent
refinement which directly optimizes placement quality post-legalization.

Usage:
    uv run evaluate submissions/phase2_analytical_placer.py -b ibm01
    uv run evaluate submissions/phase2_analytical_placer.py --all
"""

import time as _time

import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from macro_place.benchmark import Benchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnalyticalPlacer:

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
        movable_set = set(movable_indices)
        num_movable = len(movable_indices)

        if num_movable == 0:
            return placement

        # ── Prepare net data on GPU ──────────────────────────────────────
        net_data = self._prepare_nets(benchmark, dev)

        # ── Step 1: Centroid-based initial placement (Phase 0 approach) ───
        # This gives better congestion than quadratic placement
        placement = self._centroid_init(benchmark, placement, sizes, canvas_w, canvas_h)

        # ── Step 2: GPU-accelerated legalization ─────────────────────────
        placement = self._legalize(
            placement, movable_indices, sizes, canvas_w, canvas_h, num_hard, dev,
        )

        # ── Step 3: Coordinate descent + swap refinement ─────────────────
        time_left = max(10.0, 55.0 - (_time.time() - t0))
        placement = self._coordinate_descent(
            benchmark, placement, sizes, canvas_w, canvas_h,
            num_hard, net_data, dev, time_left,
        )

        # ── Step 4: Soft macro placement ─────────────────────────────────
        placement = self._place_soft_macros(benchmark, placement, sizes, canvas_w, canvas_h, dev)

        return placement

    # ── Centroid-based initial placement ─────────────────────────────────

    def _centroid_init(self, benchmark, placement, sizes, canvas_w, canvas_h):
        """Place each macro at weighted centroid of its net neighbors."""
        num_macros = benchmark.num_macros
        num_hard = benchmark.num_hard_macros
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

            net_positions = []
            net_macro_indices = []
            for nid in nodes.tolist():
                if nid < num_macros:
                    net_positions.append(placement[nid])
                    net_macro_indices.append(nid)
                elif num_ports > 0 and nid < num_macros + num_ports:
                    pi = nid - num_macros
                    net_positions.append(benchmark.port_positions[pi])

            if len(net_positions) < 2:
                continue

            pos_stack = torch.stack(net_positions)
            centroid = pos_stack.mean(dim=0)
            net_w = weight / len(net_positions)

            for mid in net_macro_indices:
                pull_x[mid] += centroid[0] * net_w
                pull_y[mid] += centroid[1] * net_w
                pull_w[mid] += net_w

        for idx in movable_indices:
            if pull_w[idx] > 0:
                cx = (pull_x[idx] / pull_w[idx]).item()
                cy = (pull_y[idx] / pull_w[idx]).item()
                w = sizes[idx, 0].item(); h = sizes[idx, 1].item()
                placement[idx, 0] = max(w / 2, min(canvas_w - w / 2, cx))
                placement[idx, 1] = max(h / 2, min(canvas_h - h / 2, cy))

        return placement

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

    # ── Quadratic placement with spreading ───────────────────────────────

    def _quadratic_with_spreading(self, benchmark, placement, sizes,
                                   canvas_w, canvas_h, movable_indices, movable_set):
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        num_movable = len(movable_indices)
        idx_to_dense = {orig: dense for dense, orig in enumerate(movable_indices)}

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
                    rhs_x[di] += w * fx; rhs_y[di] += w * fy

        if not row_e:
            return placement

        A_base = sparse.coo_matrix(
            (val_e, (row_e, col_e)), shape=(num_movable, num_movable)
        ).tocsr()
        reg = 1e-6
        A_base += sparse.eye(num_movable) * reg
        rhs_x += reg * canvas_w / 2
        rhs_y += reg * canvas_h / 2

        # Initial solve
        opt_x = spsolve(A_base, rhs_x)
        opt_y = spsolve(A_base, rhs_y)
        for i, idx in enumerate(movable_indices):
            wh = sizes[idx, 0].item() / 2; hh = sizes[idx, 1].item() / 2
            placement[idx, 0] = max(wh, min(canvas_w - wh, opt_x[i]))
            placement[idx, 1] = max(hh, min(canvas_h - hh, opt_y[i]))

        # Progressive spreading: 10 rounds with strong forces
        grid_n = 8
        for it in range(10):
            alpha = 0.03 * (1.5 ** it)  # aggressive spreading
            cell_w = canvas_w / grid_n
            cell_h = canvas_h / grid_n
            density = np.zeros((grid_n, grid_n))

            for idx in movable_indices:
                gi = min(grid_n - 1, max(0, int(placement[idx, 0].item() / cell_w)))
                gj = min(grid_n - 1, max(0, int(placement[idx, 1].item() / cell_h)))
                density[gi, gj] += sizes[idx, 0].item() * sizes[idx, 1].item()

            avg_d = max(density.sum() / (grid_n * grid_n), 1e-9)

            sparse_cells = [(cell_w * (ti + 0.5), cell_h * (tj + 0.5), avg_d - density[ti, tj])
                            for ti in range(grid_n) for tj in range(grid_n)
                            if density[ti, tj] < avg_d * 0.8]

            A_s = A_base.copy()
            rx_s, ry_s = rhs_x.copy(), rhs_y.copy()

            for i, idx in enumerate(movable_indices):
                cx = placement[idx, 0].item()
                cy = placement[idx, 1].item()
                gi = min(grid_n - 1, max(0, int(cx / cell_w)))
                gj = min(grid_n - 1, max(0, int(cy / cell_h)))
                ld = density[gi, gj]

                if ld <= avg_d * 1.05:
                    continue

                best_score = -1
                tx, ty = cx, cy
                for scx, scy, cap in sparse_cells:
                    dist = max(((scx - cx) ** 2 + (scy - cy) ** 2) ** 0.5, 0.1)
                    score = cap / dist
                    if score > best_score:
                        best_score = score; tx, ty = scx, scy

                force = alpha * (ld / avg_d)
                A_s[i, i] += force
                rx_s[i] += force * tx; ry_s[i] += force * ty

            opt_x = spsolve(A_s, rx_s)
            opt_y = spsolve(A_s, ry_s)
            for i, idx in enumerate(movable_indices):
                wh = sizes[idx, 0].item() / 2; hh = sizes[idx, 1].item() / 2
                placement[idx, 0] = max(wh, min(canvas_w - wh, opt_x[i]))
                placement[idx, 1] = max(hh, min(canvas_h - hh, opt_y[i]))

        return placement

    # ── GPU legalization ─────────────────────────────────────────────────

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

    # ── Coordinate descent + swap refinement ─────────────────────────────

    def _coordinate_descent(self, benchmark, placement, sizes, canvas_w, canvas_h,
                             num_hard, net_data, dev, time_budget):
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

        # Multi-scale shift offsets
        shift_deltas = []
        for scale in [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1),
                           (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1),
                           (1, 0.5), (-1, 0.5), (1, -0.5), (-1, -0.5)]:
                shift_deltas.append((dx * scale, dy * scale))

        # ── Phase A: Coordinate descent ──────────────────────────────────
        for _pass in range(8):
            if _time.time() > deadline:
                break
            any_imp = False

            for mi in movable_indices:
                if _time.time() > deadline:
                    break

                ox = all_pos[mi, 0].item()
                oy = all_pos[mi, 1].item()
                wi_h = half_sz[mi, 0].item()
                hi_h = half_sz[mi, 1].item()

                old_hpwl = _affected_hpwl(mi)
                best_hpwl = old_hpwl
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

                    if len(others) > 0:
                        if (((other_pos[:, 0] - nx).abs() < other_hsz[:, 0] + wi_h + gap) &
                                ((other_pos[:, 1] - ny).abs() < other_hsz[:, 1] + hi_h + gap)).any():
                            continue

                    all_pos[mi, 0] = nx; all_pos[mi, 1] = ny
                    new_hpwl = _affected_hpwl(mi)
                    if new_hpwl < best_hpwl - 1e-6:
                        best_hpwl = new_hpwl
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

                    if abs(xi - xj) > canvas_w * 0.2 or abs(yi - yj) > canvas_h * 0.2:
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
