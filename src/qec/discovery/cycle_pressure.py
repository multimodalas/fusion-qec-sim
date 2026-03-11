"""
v9.0.0 — Cycle-Pressure Heatmap.

Computes local edge stress based on short cycle participation,
two-hop overlap, and local density.  Returns ranked edges for
guided mutation.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12


def compute_cycle_pressure(H: np.ndarray) -> dict[str, Any]:
    """Compute cycle pressure for each edge in the Tanner graph.

    cycle_pressure(e) = 4_cycle_count + two_hop_overlap + local_density

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``edge_pressures`` : list of (ci, vi, pressure)
        - ``ranked_edges`` : list of (ci, vi) sorted by descending pressure
        - ``max_pressure`` : float
        - ``mean_pressure`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    # Precompute adjacency
    check_vars: dict[int, list[int]] = {}
    var_checks: dict[int, list[int]] = {}
    for ci in range(m):
        check_vars[ci] = sorted(vi for vi in range(n) if H_arr[ci, vi] != 0)
    for vi in range(n):
        var_checks[vi] = sorted(ci for ci in range(m) if H_arr[ci, vi] != 0)

    edge_pressures: list[tuple[int, int, float]] = []

    for ci in range(m):
        for vi in check_vars[ci]:
            # 4-cycle count: how many other checks share a variable with ci
            # and also connect to vi's other checks
            four_cycle_count = 0
            for vj in check_vars[ci]:
                if vj == vi:
                    continue
                for cj in var_checks[vj]:
                    if cj == ci:
                        continue
                    if H_arr[cj, vi] != 0:
                        four_cycle_count += 1

            # Two-hop overlap: number of variable neighbours shared by
            # check ci with other checks connected to vi
            two_hop_overlap = 0
            for cj in var_checks[vi]:
                if cj == ci:
                    continue
                shared = len(set(check_vars[ci]) & set(check_vars[cj]))
                two_hop_overlap += shared

            # Local density: product of check degree and variable degree
            local_density = len(check_vars[ci]) * len(var_checks[vi])

            pressure = float(four_cycle_count + two_hop_overlap + local_density)
            edge_pressures.append((ci, vi, round(pressure, _ROUND)))

    # Sort by pressure descending, then by (ci, vi) for determinism
    edge_pressures.sort(key=lambda e: (-e[2], e[0], e[1]))

    pressures_only = [p for _, _, p in edge_pressures]
    max_p = max(pressures_only) if pressures_only else 0.0
    mean_p = sum(pressures_only) / len(pressures_only) if pressures_only else 0.0

    return {
        "edge_pressures": edge_pressures,
        "ranked_edges": [(ci, vi) for ci, vi, _ in edge_pressures],
        "max_pressure": round(max_p, _ROUND),
        "mean_pressure": round(mean_p, _ROUND),
    }
