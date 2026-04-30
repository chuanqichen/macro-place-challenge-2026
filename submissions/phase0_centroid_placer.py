"""
Phase 0: Centroid Placer — Connectivity-Aware Warm-Up

A minimal but connectivity-aware placer that:
1. Computes the weighted centroid of each macro's net neighbors
2. Places each movable hard macro at its centroid (clamped to canvas)
3. Resolves overlaps with a simple greedy push-apart pass
4. Places soft macros at the centroid of their connected hard macros

This should beat the greedy row placer by actually using the netlist,
while being trivial to implement and debug (~50 lines of logic).

Usage:
    uv run evaluate submissions/phase0_centroid_placer.py -b ibm01
    uv run evaluate submissions/phase0_centroid_placer.py --all
"""

import torch
from macro_place.benchmark import Benchmark


class CentroidPlacer:
    """
    Place each macro at the weighted centroid of its net neighbors,
    then greedily resolve overlaps.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        num_hard = benchmark.num_hard_macros

        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()

        # ── Step 1: Build per-macro connectivity weights ─────────────────
        # For each macro, accumulate (weighted_x, weighted_y, total_weight)
        # from all nets it belongs to. Net neighbors pull it toward them.
        pull_x = torch.zeros(benchmark.num_macros)
        pull_y = torch.zeros(benchmark.num_macros)
        pull_w = torch.zeros(benchmark.num_macros)

        # Include port positions as additional pull targets
        num_ports = benchmark.port_positions.shape[0]
        num_macros = benchmark.num_macros

        for net_idx in range(benchmark.num_nets):
            nodes = benchmark.net_nodes[net_idx]
            weight = benchmark.net_weights[net_idx].item()
            if len(nodes) < 2:
                continue

            # Collect positions of all nodes in this net
            # (macros use current placement, ports use fixed positions)
            net_positions = []
            net_macro_indices = []
            for node_id in nodes.tolist():
                if node_id < num_macros:
                    net_positions.append(placement[node_id])
                    net_macro_indices.append(node_id)
                elif num_ports > 0 and node_id < num_macros + num_ports:
                    port_idx = node_id - num_macros
                    net_positions.append(benchmark.port_positions[port_idx])

            if len(net_positions) < 2:
                continue

            # Compute centroid of this net
            pos_stack = torch.stack(net_positions)
            centroid = pos_stack.mean(dim=0)

            # Each macro in the net gets pulled toward the net centroid
            # Weight inversely with net degree (large nets = weaker pull per node)
            net_weight = weight / len(net_positions)
            for macro_id in net_macro_indices:
                pull_x[macro_id] += centroid[0] * net_weight
                pull_y[macro_id] += centroid[1] * net_weight
                pull_w[macro_id] += net_weight

        # ── Step 2: Place movable hard macros at their centroid ──────────
        for idx in movable_indices:
            if pull_w[idx] > 0:
                cx = pull_x[idx] / pull_w[idx]
                cy = pull_y[idx] / pull_w[idx]
            else:
                # No connectivity info — keep original position
                continue

            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            # Clamp to canvas bounds (positions are centers)
            cx = max(w / 2, min(canvas_w - w / 2, cx.item()))
            cy = max(h / 2, min(canvas_h - h / 2, cy.item()))

            placement[idx, 0] = cx
            placement[idx, 1] = cy

        # ── Step 3: Greedy overlap removal ───────────────────────────────
        # Place macros one-by-one in area-descending order.
        # For each macro, find the closest legal position to its centroid
        # that doesn't overlap any already-placed macro.
        gap = 0.01

        movable_by_area = sorted(
            movable_indices,
            key=lambda i: sizes[i, 0].item() * sizes[i, 1].item(),
            reverse=True,
        )

        # Collect fixed hard macros as already-placed obstacles
        placed = []
        for j in range(num_hard):
            if j not in movable_indices:
                placed.append(j)

        def _overlaps_any(idx, px, py):
            """Check if macro idx at (px, py) overlaps any placed macro."""
            wi = sizes[idx, 0].item()
            hi = sizes[idx, 1].item()
            for j in placed:
                xj, yj = placement[j, 0].item(), placement[j, 1].item()
                wj, hj = sizes[j, 0].item(), sizes[j, 1].item()
                if (abs(px - xj) < (wi + wj) / 2 + gap and
                        abs(py - yj) < (hi + hj) / 2 + gap):
                    return True
            return False

        def _find_legal_pos(idx):
            """Find closest legal position to current centroid via spiral search."""
            cx = placement[idx, 0].item()
            cy = placement[idx, 1].item()
            wi = sizes[idx, 0].item()
            hi = sizes[idx, 1].item()

            # Try original position first
            cx = max(wi / 2, min(canvas_w - wi / 2, cx))
            cy = max(hi / 2, min(canvas_h - hi / 2, cy))
            if not _overlaps_any(idx, cx, cy):
                return cx, cy

            # Spiral search: try offsets in increasing distance
            step = max(wi, hi) * 0.5
            for radius_mult in range(1, 200):
                r = step * radius_mult
                # Try 8 directions + 8 diagonals at this radius
                for dx_frac, dy_frac in [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                    (0.5, 1), (0.5, -1), (-0.5, 1), (-0.5, -1),
                    (1, 0.5), (1, -0.5), (-1, 0.5), (-1, -0.5),
                ]:
                    nx = cx + dx_frac * r
                    ny = cy + dy_frac * r
                    # Clamp to canvas
                    nx = max(wi / 2, min(canvas_w - wi / 2, nx))
                    ny = max(hi / 2, min(canvas_h - hi / 2, ny))
                    if not _overlaps_any(idx, nx, ny):
                        return nx, ny

            # Fallback: return clamped centroid (may still overlap)
            return cx, cy

        for idx in movable_by_area:
            px, py = _find_legal_pos(idx)
            placement[idx, 0] = px
            placement[idx, 1] = py
            placed.append(idx)

        # ── Step 4: Place soft macros at centroid of connected hard macros ─
        soft_mask = benchmark.get_soft_macro_mask()
        soft_indices = torch.where(soft_mask)[0].tolist()

        if soft_indices:
            soft_pull_x = torch.zeros(benchmark.num_macros)
            soft_pull_y = torch.zeros(benchmark.num_macros)
            soft_pull_w = torch.zeros(benchmark.num_macros)

            for net_idx in range(benchmark.num_nets):
                nodes = benchmark.net_nodes[net_idx]
                weight = benchmark.net_weights[net_idx].item()
                if len(nodes) < 2:
                    continue

                # Collect current positions (hard macros already placed)
                net_positions = []
                net_soft_indices = []
                for node_id in nodes.tolist():
                    if node_id < num_hard:
                        net_positions.append(placement[node_id])
                    elif node_id < num_macros and node_id in soft_indices:
                        net_soft_indices.append(node_id)
                    elif num_ports > 0 and node_id >= num_macros and node_id < num_macros + num_ports:
                        port_idx = node_id - num_macros
                        net_positions.append(benchmark.port_positions[port_idx])

                if not net_positions or not net_soft_indices:
                    continue

                pos_stack = torch.stack(net_positions)
                centroid = pos_stack.mean(dim=0)
                net_weight = weight / len(net_positions)

                for s_idx in net_soft_indices:
                    soft_pull_x[s_idx] += centroid[0] * net_weight
                    soft_pull_y[s_idx] += centroid[1] * net_weight
                    soft_pull_w[s_idx] += net_weight

            for idx in soft_indices:
                if soft_pull_w[idx] > 0:
                    cx = (soft_pull_x[idx] / soft_pull_w[idx]).item()
                    cy = (soft_pull_y[idx] / soft_pull_w[idx]).item()
                    w = sizes[idx, 0].item()
                    h = sizes[idx, 1].item()
                    placement[idx, 0] = max(w / 2, min(canvas_w - w / 2, cx))
                    placement[idx, 1] = max(h / 2, min(canvas_h - h / 2, cy))

        return placement
