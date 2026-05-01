"""
Phase 6: Exact-Proxy Guided Placement

Key innovation: loads its own PlacementCost evaluator to use the EXACT
proxy cost (wirelength + density + congestion) during optimization.

Strategy:
1. Centroid init + spiral legalization (same as Phase 2)
2. CD refinement using GPU HPWL (fast, many passes)
3. Periodic exact proxy evaluation via plc to validate improvements
4. Reject CD moves that improve HPWL but worsen actual proxy cost
5. Soft macro placement with exact proxy validation

Usage:
    uv run evaluate submissions/phase6_exact_placer.py -b ibm01
    uv run evaluate submissions/phase6_exact_placer.py --all
"""

import time as _time
import os

import torch
import numpy as np

from macro_place.benchmark import Benchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExactPlacer:

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

        # ── Load our own plc for exact proxy evaluation ──────────────────
        plc = self._load_plc(benchmark)

        # ── Step 1: Centroid init ────────────────────────────────────────
        placement = self._centroid_init(benchmark, placement, sizes, canvas_w, canvas_h)

        # ── Step 2: Spiral legalization ──────────────────────────────────
        placement = self._legalize(placement, movable_indices, sizes, canvas_w, canvas_h, num_hard, dev)

        # ── Step 3: Proxy-guided CD refinement ───────────────────────────
        net_data = self._prepare_nets(benchmark, dev)
        cd_budget = max(10.0, 53.0 - (_time.time() - t0))
        placement = self._proxy_guided_cd(
            benchmark, plc, placement, sizes, canvas_w, canvas_h,
            num_hard, movable_indices, net_data, dev, cd_budget,
        )

        # ── Step 4: Soft macro placement ─────────────────────────────────
        placement = self._place_soft_macros(benchmark, placement, sizes, canvas_w, canvas_h, dev)

        # ── Step 5: Final proxy check — try without CD to see if base is better
        if plc is not None and _time.time() - t0 < 57:
            from macro_place.objective import compute_proxy_cost
            try:
                costs_final = compute_proxy_cost(placement, benchmark, plc)
                final_proxy = costs_final['proxy_cost']
            except Exception:
                final_proxy = float('inf')

            # Also evaluate the base (centroid + legalize + soft) without CD
            base = benchmark.macro_positions.clone()
            base = self._centroid_init(benchmark, base, sizes, canvas_w, canvas_h)
            base = self._legalize(base, movable_indices, sizes, canvas_w, canvas_h, num_hard, dev)
            base = self._place_soft_macros(benchmark, base, sizes, canvas_w, canvas_h, dev)
            try:
                costs_base = compute_proxy_cost(base, benchmark, plc)
                base_proxy = costs_base['proxy_cost']
            except Exception:
                base_proxy = float('inf')

            if base_proxy < final_proxy:
                placement = base

        return placement

    # ── Load PlacementCost ───────────────────────────────────────────────

    def _load_plc(self, benchmark):
        """Try to load a PlacementCost evaluator for exact proxy computation."""
        try:
            name = benchmark.name
            # Try IBM benchmarks first
            path = f'external/MacroPlacement/Testcases/ICCAD04/{name}'
            if os.path.exists(path):
                from macro_place.loader import load_benchmark_from_dir
                _, plc = load_benchmark_from_dir(path)
                return plc
            # Try NG45
            ng45_paths = {
                'ariane133': 'external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping',
                'ariane136': 'external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping',
                'mempool_tile': 'external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping',
                'nvdla': 'external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping',
            }
            # Handle _ng45 suffix
            base_name = name.replace('_ng45', '').replace('_asap7', '')
            if base_name in ng45_paths:
                ng_dir = ng45_paths[base_name]
                from macro_place.loader import load_benchmark
                _, plc = load_benchmark(f'{ng_dir}/netlist.pb.txt', f'{ng_dir}/initial.plc', name=name)
                return plc
        except Exception:
            pass
        return None

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
            [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
            [0.5, 1], [0.5, -1], [-0.5, 1], [-0.5, -1],
            [1, 0.5], [1, -0.5], [-1, 0.5], [-1, -0.5],
            [0.3, 0.7], [-0.3, 0.7], [0.3, -0.7], [-0.3, -0.7],
            [0.7, 0.3], [-0.7, 0.3], [0.7, -0.3], [-0.7, -0.3],
        ], device=dev)

        for idx in sorted_idx:
            wi = sizes_gpu[idx, 0].item(); hi = sizes_gpu[idx, 1].item()
            cx, cy = targets[idx]
            cx = max(wi / 2, min(canvas_w - wi / 2, cx))
            cy = max(hi / 2, min(canvas_h - hi / 2, cy))
            pi = torch.where(placed_mask)[0]
            if len(pi) == 0 or not (((pos_gpu[pi, 0] - cx).abs() < half_sz[pi, 0] + wi / 2 + gap) &
                                     ((pos_gpu[pi, 1] - cy).abs() < half_sz[pi, 1] + hi / 2 + gap)).any():
                pos_gpu[idx] = torch.tensor([cx, cy], device=dev); placed_mask[idx] = True; continue
            step = max(min(wi, hi) * 0.25, 0.1)
            found = False
            for rm in range(1, 500):
                r = step * rm
                if r > max(canvas_w, canvas_h): break
                cands = torch.tensor([cx, cy], device=dev).unsqueeze(0) + offsets_t * r
                cands[:, 0].clamp_(wi / 2, canvas_w - wi / 2); cands[:, 1].clamp_(hi / 2, canvas_h - hi / 2)
                dx = (cands[:, 0].unsqueeze(1) - pos_gpu[pi, 0].unsqueeze(0)).abs()
                dy = (cands[:, 1].unsqueeze(1) - pos_gpu[pi, 1].unsqueeze(0)).abs()
                ov = (dx < half_sz[pi, 0].unsqueeze(0) + wi / 2 + gap) & (dy < half_sz[pi, 1].unsqueeze(0) + hi / 2 + gap)
                free = ~ov.any(dim=1)
                if free.any():
                    dists = (cands[free, 0] - cx) ** 2 + (cands[free, 1] - cy) ** 2
                    bi = torch.where(free)[0][dists.argmin()]
                    pos_gpu[idx] = cands[bi]; placed_mask[idx] = True; found = True; break
            if not found:
                pos_gpu[idx] = torch.tensor([cx, cy], device=dev); placed_mask[idx] = True
        placement[:num_hard] = pos_gpu.cpu()
        return placement

    # ── Net data prep ────────────────────────────────────────────────────

    def _prepare_nets(self, benchmark, dev):
        num_macros = benchmark.num_macros; num_ports = benchmark.port_positions.shape[0]
        total_nodes = num_macros + num_ports; nets = benchmark.net_nodes
        if not nets: return None
        max_deg = max(len(n) for n in nets); num_nets = len(nets)
        net_tensor = torch.full((num_nets, max_deg), total_nodes, dtype=torch.long, device=dev)
        net_mask = torch.zeros(num_nets, max_deg, dtype=torch.bool, device=dev)
        weights = benchmark.net_weights.to(dev)
        for i, nodes in enumerate(nets):
            valid = nodes[nodes < total_nodes]; n = len(valid)
            if n >= 2: net_tensor[i, :n] = valid.to(dev); net_mask[i, :n] = True
        valid_nets = net_mask.sum(dim=1) >= 2
        net_tensor = net_tensor[valid_nets]; net_mask = net_mask[valid_nets]; weights = weights[valid_nets]
        macro_nets = [[] for _ in range(num_macros)]
        valid_idx = torch.where(valid_nets)[0]
        for li in range(len(valid_idx)):
            oi = valid_idx[li].item()
            for nid in benchmark.net_nodes[oi].tolist():
                if nid < num_macros: macro_nets[nid].append(li)
        return {'net_tensor': net_tensor, 'net_mask': net_mask, 'weights': weights,
                'total_nodes': total_nodes, 'macro_nets': macro_nets}

    # ── Proxy-guided CD ──────────────────────────────────────────────────

    def _proxy_guided_cd(self, benchmark, plc, placement, sizes, canvas_w, canvas_h,
                          num_hard, movable_indices, net_data, dev, time_budget):
        """
        CD refinement with periodic exact proxy cost checkpoints.
        After each full CD pass, evaluate actual proxy cost via plc.
        If proxy cost worsened, revert to the last good state.
        """
        if net_data is None:
            return placement

        from macro_place.objective import compute_proxy_cost

        deadline = _time.time() + time_budget
        gap = 0.01
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        total_nodes = net_data['total_nodes']

        all_pos = torch.zeros(total_nodes, 2, device=dev)
        all_pos[:num_macros] = placement.to(dev)
        if num_ports > 0:
            all_pos[num_macros:] = benchmark.port_positions.to(dev)

        sizes_gpu = sizes[:num_hard].to(dev)
        half_sz = sizes_gpu / 2
        macro_nets = net_data['macro_nets']
        net_tensor = net_data['net_tensor']
        net_mask = net_data['net_mask']
        net_weights = net_data['weights']

        def _affected_hpwl(mi):
            nids_list = macro_nets[mi]
            if not nids_list: return 0.0
            nids = torch.tensor(nids_list, device=dev, dtype=torch.long)
            padded = torch.cat([all_pos, torch.zeros(1, 2, device=dev)], dim=0)
            nt = net_tensor[nids]; nm = net_mask[nids]; nw = net_weights[nids]
            pos = padded[nt]
            x = pos[:, :, 0].clone(); y = pos[:, :, 1].clone()
            x[~nm] = 1e9; x_min = x.min(1).values; x[~nm] = -1e9; x_max = x.max(1).values
            y[~nm] = 1e9; y_min = y.min(1).values; y[~nm] = -1e9; y_max = y.max(1).values
            return (((x_max - x_min) + (y_max - y_min)) * nw).sum().item()

        shift_deltas = []
        for scale in [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1),
                           (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1),
                           (1, 0.5), (-1, 0.5), (1, -0.5), (-1, -0.5)]:
                shift_deltas.append((dx * scale, dy * scale))

        # Track best placement by actual proxy cost
        best_placement = placement.clone()
        best_proxy = float('inf')

        # Evaluate initial proxy cost
        if plc is not None:
            soft_p = self._place_soft_macros(benchmark, placement.clone(), sizes, canvas_w, canvas_h, dev)
            try:
                costs = compute_proxy_cost(soft_p, benchmark, plc)
                best_proxy = costs['proxy_cost']
            except Exception:
                pass

        for _pass in range(10):
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

                exc = torch.ones(num_hard, dtype=torch.bool, device=dev); exc[mi] = False
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

            # After each CD pass, check actual proxy cost
            if plc is not None and _time.time() < deadline - 2:
                cur_placement = placement.clone()
                cur_placement[:num_macros] = all_pos[:num_macros].cpu()
                soft_p = self._place_soft_macros(benchmark, cur_placement.clone(), sizes, canvas_w, canvas_h, dev)
                try:
                    costs = compute_proxy_cost(soft_p, benchmark, plc)
                    cur_proxy = costs['proxy_cost']
                    if costs['overlap_count'] == 0 and cur_proxy < best_proxy:
                        best_proxy = cur_proxy
                        best_placement = soft_p.clone()
                except Exception:
                    pass

            if not any_imp:
                break

        # Pairwise swaps
        for _pass in range(3):
            if _time.time() > deadline: break
            any_swap = False
            for ip in range(len(movable_indices)):
                if _time.time() > deadline: break
                mi = movable_indices[ip]
                xi = all_pos[mi, 0].item(); yi = all_pos[mi, 1].item()
                wi_h = half_sz[mi, 0].item(); hi_h = half_sz[mi, 1].item()
                for jp in range(ip + 1, len(movable_indices)):
                    mj = movable_indices[jp]
                    xj = all_pos[mj, 0].item(); yj = all_pos[mj, 1].item()
                    if abs(xi - xj) > canvas_w * 0.25 or abs(yi - yj) > canvas_h * 0.25: continue
                    wj_h = half_sz[mj, 0].item(); hj_h = half_sz[mj, 1].item()
                    if xj-wi_h<0 or xj+wi_h>canvas_w or yj-hi_h<0 or yj+hi_h>canvas_h: continue
                    if xi-wj_h<0 or xi+wj_h>canvas_w or yi-hj_h<0 or yi+hj_h>canvas_h: continue
                    exc = torch.ones(num_hard, dtype=torch.bool, device=dev)
                    exc[mi] = False; exc[mj] = False
                    others = torch.where(exc)[0]
                    if len(others) > 0:
                        op = all_pos[others]; hs = half_sz[others]
                        if (((op[:, 0]-xj).abs()<hs[:, 0]+wi_h+gap)&((op[:, 1]-yj).abs()<hs[:, 1]+hi_h+gap)).any(): continue
                        if (((op[:, 0]-xi).abs()<hs[:, 0]+wj_h+gap)&((op[:, 1]-yi).abs()<hs[:, 1]+hj_h+gap)).any(): continue
                    old_h = _affected_hpwl(mi) + _affected_hpwl(mj)
                    all_pos[mi, 0], all_pos[mj, 0] = xj, xi
                    all_pos[mi, 1], all_pos[mj, 1] = yj, yi
                    new_h = _affected_hpwl(mi) + _affected_hpwl(mj)
                    if new_h < old_h - 1e-6: any_swap = True; xi, yi = xj, yj
                    else: all_pos[mi, 0], all_pos[mj, 0] = xi, xj; all_pos[mi, 1], all_pos[mj, 1] = yi, yj
            if not any_swap: break

        # Final proxy check after swaps
        if plc is not None and _time.time() - _time.time() < 57:
            cur_placement = placement.clone()
            cur_placement[:num_macros] = all_pos[:num_macros].cpu()
            soft_p = self._place_soft_macros(benchmark, cur_placement.clone(), sizes, canvas_w, canvas_h, dev)
            try:
                costs = compute_proxy_cost(soft_p, benchmark, plc)
                if costs['overlap_count'] == 0 and costs['proxy_cost'] < best_proxy:
                    best_proxy = costs['proxy_cost']
                    best_placement = soft_p.clone()
            except Exception:
                pass

        return best_placement

    # ── Soft macro placement ─────────────────────────────────────────────

    def _place_soft_macros(self, benchmark, placement, sizes, canvas_w, canvas_h, dev):
        num_macros = benchmark.num_macros; num_hard = benchmark.num_hard_macros
        num_ports = benchmark.port_positions.shape[0]
        soft_indices = list(range(num_hard, num_macros))
        if not soft_indices: return placement
        soft_set = set(soft_indices)
        pull_x = torch.zeros(num_macros, device=dev)
        pull_y = torch.zeros(num_macros, device=dev)
        pull_w = torch.zeros(num_macros, device=dev)
        pos_d = placement.to(dev)
        port_d = benchmark.port_positions.to(dev) if num_ports > 0 else None
        for ni in range(len(benchmark.net_nodes)):
            nodes = benchmark.net_nodes[ni]; w = benchmark.net_weights[ni].item()
            if len(nodes) < 2: continue
            ax, ay, ns = [], [], []
            for nid in nodes.tolist():
                if nid in soft_set: ns.append(nid)
                elif nid < num_macros: ax.append(pos_d[nid, 0]); ay.append(pos_d[nid, 1])
                elif port_d is not None and nid < num_macros + num_ports:
                    pi = nid - num_macros; ax.append(port_d[pi, 0]); ay.append(port_d[pi, 1])
            if not ax or not ns: continue
            cx = torch.stack(ax).mean(); cy = torch.stack(ay).mean()
            nw = w / max(1, len(ax))
            for s in ns: pull_x[s] += cx * nw; pull_y[s] += cy * nw; pull_w[s] += nw
        valid = pull_w > 0; sm = torch.zeros(num_macros, dtype=torch.bool, device=dev)
        for s in soft_indices: sm[s] = True
        upd = valid & sm
        if upd.any():
            idx = torch.where(upd)[0]; sizes_d = sizes.to(dev)
            wh = sizes_d[idx, 0] / 2; hh_t = sizes_d[idx, 1] / 2
            pos_d[idx, 0] = (pull_x[idx] / pull_w[idx]).clamp(min=wh, max=canvas_w - wh)
            pos_d[idx, 1] = (pull_y[idx] / pull_w[idx]).clamp(min=hh_t, max=canvas_h - hh_t)
        return pos_d.cpu()
