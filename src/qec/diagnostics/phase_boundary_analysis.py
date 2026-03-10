"""
v5.9.0 — Phase Boundary Analysis.

Identifies boundary cells, mixed-region cells, and critical cells
from a decoder phase diagram produced by
:func:`~src.qec.diagnostics.phase_diagram.build_decoder_phase_diagram`.

Boundary cells:
    Cells whose dominant_phase differs from at least one grid-adjacent cell.

Mixed-region cells:
    Cells with phase_entropy above a deterministic threshold.

Critical cells:
    Cells where boundary_fraction is high, OR mean_metastability_score
    is high, OR mean_boundary_eps is near zero.

All thresholds are explicit and deterministic.

Does not modify decoder internals.  Treats the decoder as a pure
function.  All outputs are JSON-serializable.  Fully deterministic:
no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any


# ── Default thresholds ──────────────────────────────────────────────

DEFAULT_ENTROPY_THRESHOLD = 0.5
DEFAULT_BOUNDARY_FRACTION_THRESHOLD = 0.3
DEFAULT_METASTABILITY_THRESHOLD = 0.5
DEFAULT_BOUNDARY_EPS_THRESHOLD = 0.01


# ── Public API ──────────────────────────────────────────────────────


def analyze_phase_boundaries(
    phase_diagram_result: dict[str, Any],
    *,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    boundary_fraction_threshold: float = DEFAULT_BOUNDARY_FRACTION_THRESHOLD,
    metastability_threshold: float = DEFAULT_METASTABILITY_THRESHOLD,
    boundary_eps_threshold: float = DEFAULT_BOUNDARY_EPS_THRESHOLD,
) -> dict[str, Any]:
    """Analyze a phase diagram for boundary, mixed, and critical cells.

    Parameters
    ----------
    phase_diagram_result : dict
        Output from :func:`build_decoder_phase_diagram`.
    entropy_threshold : float
        Phase entropy above which a cell is considered mixed.
    boundary_fraction_threshold : float
        Boundary fraction above which a cell is critical.
    metastability_threshold : float
        Mean metastability score above which a cell is critical.
    boundary_eps_threshold : float
        Mean boundary_eps below which a cell is critical
        (near-zero distance to decision boundary).

    Returns
    -------
    dict[str, Any]
        Analysis with ``boundary_cells``, ``mixed_region_cells``,
        ``critical_cells``, and ``boundary_summary``.

    Raises
    ------
    ValueError
        If phase_diagram_result is missing required keys.
    """
    _validate_phase_diagram(phase_diagram_result)

    grid_axes = phase_diagram_result["grid_axes"]
    cells = phase_diagram_result["cells"]

    x_values = grid_axes["x_values"]
    y_values = grid_axes["y_values"]

    # Build lookup: (x, y) -> cell for adjacency checks.
    cell_lookup: dict[tuple[Any, Any], dict[str, Any]] = {}
    for cell in cells:
        cell_lookup[(cell["x"], cell["y"])] = cell

    # Build index maps for adjacency.
    x_indices = {v: i for i, v in enumerate(x_values)}
    y_indices = {v: i for i, v in enumerate(y_values)}

    boundary_cells: list[dict[str, Any]] = []
    mixed_region_cells: list[dict[str, Any]] = []
    critical_cells: list[dict[str, Any]] = []

    # Deterministic iteration order follows cells list order.
    for cell in cells:
        cx, cy = cell["x"], cell["y"]
        xi = x_indices.get(cx)
        yi = y_indices.get(cy)

        if xi is None or yi is None:
            continue

        # ── Boundary detection: dominant_phase differs from neighbor ──
        is_boundary = False
        dominant = cell["dominant_phase"]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_i = xi + dx
            ny_i = yi + dy
            if 0 <= nx_i < len(x_values) and 0 <= ny_i < len(y_values):
                neighbor_key = (x_values[nx_i], y_values[ny_i])
                neighbor = cell_lookup.get(neighbor_key)
                if neighbor is not None and neighbor["dominant_phase"] != dominant:
                    is_boundary = True
                    break

        if is_boundary:
            boundary_cells.append(_cell_ref(cell))

        # ── Mixed-region detection: high phase entropy ──
        if cell["phase_entropy"] > entropy_threshold:
            mixed_region_cells.append(_cell_ref(cell))

        # ── Critical-cell detection ──
        is_critical = False

        if cell["boundary_fraction"] > boundary_fraction_threshold:
            is_critical = True

        ms = cell.get("mean_metastability_score")
        if ms is not None and ms > metastability_threshold:
            is_critical = True

        be = cell.get("mean_boundary_eps")
        if be is not None and 0.0 <= be < boundary_eps_threshold:
            is_critical = True

        if is_critical:
            critical_cells.append(_cell_ref(cell))

    return {
        "boundary_cells": boundary_cells,
        "mixed_region_cells": mixed_region_cells,
        "critical_cells": critical_cells,
        "boundary_summary": {
            "num_boundary_cells": len(boundary_cells),
            "num_mixed_cells": len(mixed_region_cells),
            "num_critical_cells": len(critical_cells),
        },
    }


# ── Validation ───────────────────────────────────────────────────────


def _validate_phase_diagram(result: dict[str, Any]) -> None:
    """Validate phase diagram structure."""
    if "grid_axes" not in result:
        raise ValueError("phase_diagram_result missing 'grid_axes'")
    if "cells" not in result:
        raise ValueError("phase_diagram_result missing 'cells'")
    axes = result["grid_axes"]
    for key in ("x_name", "x_values", "y_name", "y_values"):
        if key not in axes:
            raise ValueError(f"grid_axes missing '{key}'")


# ── Helpers ──────────────────────────────────────────────────────────


def _cell_ref(cell: dict[str, Any]) -> dict[str, Any]:
    """Return a compact cell reference for boundary analysis output."""
    return {
        "x": cell["x"],
        "y": cell["y"],
        "dominant_phase": cell["dominant_phase"],
        "phase_entropy": cell["phase_entropy"],
        "success_fraction": cell["success_fraction"],
        "boundary_fraction": cell["boundary_fraction"],
        "failure_fraction": cell["failure_fraction"],
    }
