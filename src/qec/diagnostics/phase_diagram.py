"""
v5.9.0 / v6.0.0 — Decoder Phase Diagram Aggregation.

Builds empirical decoder phase diagrams by sweeping a deterministic 2D
parameter grid, running decoding experiments at each grid point, and
aggregating ternary topology classifications with continuous diagnostics
into a phase-diagram-ready JSON artifact.

The ternary classifier output (+1 / 0 / -1) serves as the primary
categorical phase label.  Continuous observables (boundary distance,
barrier height, metastability, oscillation, alignment, cluster count)
are preserved for downstream validation of phase boundaries.

These are *empirical decoder dynamical phases*, not thermodynamic phases.

Does not modify decoder internals.  Treats the decoder as a pure
function.  All outputs are JSON-serializable.  Fully deterministic:
no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any


# ── Grid specification ──────────────────────────────────────────────


def make_phase_grid(
    x_name: str,
    x_values: Sequence[float | int],
    y_name: str,
    y_values: Sequence[float | int],
) -> dict[str, Any]:
    """Create a deterministic 2D parameter grid specification.

    Parameters
    ----------
    x_name : str
        Name of the x-axis parameter (e.g. ``"physical_error_rate"``).
    x_values : Sequence[float | int]
        Sorted values along the x-axis.
    y_name : str
        Name of the y-axis parameter (e.g. ``"llr_scale"``).
    y_values : Sequence[float | int]
        Sorted values along the y-axis.

    Returns
    -------
    dict[str, Any]
        Grid specification with ``x_name``, ``x_values``,
        ``y_name``, ``y_values``.

    Raises
    ------
    ValueError
        If either axis has fewer than one value, or names are empty.
    """
    if not x_name or not y_name:
        raise ValueError("axis names must be non-empty strings")
    x_list = list(x_values)
    y_list = list(y_values)
    if len(x_list) < 1:
        raise ValueError("x_values must contain at least one value")
    if len(y_list) < 1:
        raise ValueError("y_values must contain at least one value")
    return {
        "x_name": str(x_name),
        "x_values": x_list,
        "y_name": str(y_name),
        "y_values": y_list,
    }


# ── Shannon entropy ─────────────────────────────────────────────────


def _shannon_entropy(fractions: Sequence[float]) -> float:
    """Compute Shannon entropy of a discrete probability distribution.

    Uses natural logarithm.  Zero-probability entries are skipped.
    """
    h = 0.0
    for p in fractions:
        if p > 0.0:
            h -= p * math.log(p)
    return h


# ── Safe mean helper ─────────────────────────────────────────────────


def _safe_mean(values: list[float | None]) -> float | None:
    """Return the mean of non-None values, or None if all are None."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


# ── Phase diagram builder ───────────────────────────────────────────


def build_decoder_phase_diagram(
    parameter_grid: dict[str, Any],
    trial_runner: Callable[[float | int, float | int], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build a decoder phase diagram from a 2D parameter sweep.

    Parameters
    ----------
    parameter_grid : dict
        Grid specification from :func:`make_phase_grid`.
    trial_runner : callable
        ``trial_runner(x_value, y_value) -> list[dict]``

        Called once per grid point.  Must return a list of per-trial
        result dicts, each containing at minimum:

        - ``"final_ternary_state"`` : int (+1, 0, or -1)

        Optionally may include (from existing v5 diagnostics):

        - ``"evidence"`` : dict with ``"boundary_eps_final"``,
          ``"barrier_eps_final"``, ``"oscillation_score"``,
          ``"alignment_max_final"``
        - ``"metastability_score"`` : float
        - ``"cluster_count"`` : int

    Returns
    -------
    dict[str, Any]
        Phase diagram with ``grid_axes`` and ``cells`` keys.

    Raises
    ------
    ValueError
        If parameter_grid is missing required keys.
    TypeError
        If trial_runner is not callable.
    """
    # ── Validate inputs ─────────────────────────────────────────
    _validate_grid(parameter_grid)
    if not callable(trial_runner):
        raise TypeError("trial_runner must be callable")

    x_name = parameter_grid["x_name"]
    x_values = parameter_grid["x_values"]
    y_name = parameter_grid["y_name"]
    y_values = parameter_grid["y_values"]

    cells: list[dict[str, Any]] = []

    # Deterministic sweep order: x outer, y inner.
    for x_val in x_values:
        for y_val in y_values:
            trial_results = trial_runner(x_val, y_val)
            cell = _aggregate_cell(x_val, y_val, trial_results)
            cells.append(cell)

    return {
        "grid_axes": {
            "x_name": x_name,
            "x_values": list(x_values),
            "y_name": y_name,
            "y_values": list(y_values),
        },
        "cells": cells,
    }


# ── Validation ───────────────────────────────────────────────────────


def _validate_grid(grid: dict[str, Any]) -> None:
    """Validate parameter grid structure."""
    for key in ("x_name", "x_values", "y_name", "y_values"):
        if key not in grid:
            raise ValueError(f"parameter_grid missing required key '{key}'")
    if not grid["x_values"]:
        raise ValueError("x_values must not be empty")
    if not grid["y_values"]:
        raise ValueError("y_values must not be empty")


# ── Cell aggregation ─────────────────────────────────────────────────


def _aggregate_cell(
    x_val: float | int,
    y_val: float | int,
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate trial results for a single grid cell."""
    trial_count = len(trial_results)

    if trial_count == 0:
        return {
            "x": x_val,
            "y": y_val,
            "trial_count": 0,
            "success_fraction": 0.0,
            "boundary_fraction": 0.0,
            "failure_fraction": 0.0,
            "dominant_phase": 0,
            "phase_entropy": 0.0,
            "mean_boundary_eps": None,
            "mean_barrier_eps": None,
            "mean_metastability_score": None,
            "mean_oscillation_score": None,
            "mean_alignment_max": None,
            "mean_cluster_count": None,
            "mean_spectral_radius": None,
            "mean_bethe_min_eigenvalue": None,
            "mean_bp_stability_score": None,
            "mean_jacobian_spectral_radius_est": None,
            "mean_nb_max_ipr": None,
            "mean_nb_num_localized_modes": None,
            "mean_nb_top_localization_score": None,
            "mean_nb_candidate_nodes": None,
            "mean_nb_max_node_participation": None,
            "mean_nb_candidate_clusters": None,
            "mean_spectral_bp_alignment": None,
            "mean_candidate_node_overlap_fraction": None,
            "mean_candidate_cluster_overlap_fraction": None,
            "mean_cluster_risk": None,
            "max_cluster_risk": None,
            "mean_num_high_risk_clusters": None,
        }

    # ── Count ternary states ────────────────────────────────────
    success_count = 0
    boundary_count = 0
    failure_count = 0

    boundary_eps_values: list[float | None] = []
    barrier_eps_values: list[float | None] = []
    metastability_values: list[float | None] = []
    oscillation_values: list[float | None] = []
    alignment_values: list[float | None] = []
    cluster_values: list[float | None] = []

    # v6.0 spectral stability collectors (opt-in, additive).
    spectral_radius_values: list[float | None] = []
    bethe_min_values: list[float | None] = []
    bp_stability_values: list[float | None] = []
    jacobian_est_values: list[float | None] = []

    # v6.1 localization collectors (opt-in, additive).
    nb_max_ipr_values: list[float | None] = []
    nb_num_localized_values: list[float | None] = []
    nb_top_localization_values: list[float | None] = []

    # v6.2 trapping-set candidate collectors (opt-in, additive).
    nb_candidate_nodes_values: list[float | None] = []
    nb_max_participation_values: list[float | None] = []
    nb_candidate_clusters_values: list[float | None] = []

    # v6.3 spectral-BP alignment collectors (opt-in, additive).
    sbpa_alignment_values: list[float | None] = []
    sbpa_cand_overlap_values: list[float | None] = []
    sbpa_cluster_overlap_values: list[float | None] = []

    # v6.4 spectral failure risk collectors (opt-in, additive).
    sfr_mean_cluster_risk_values: list[float | None] = []
    sfr_max_cluster_risk_values: list[float | None] = []
    sfr_num_high_risk_values: list[float | None] = []

    for trial in trial_results:
        state = trial.get("final_ternary_state", 0)
        if state == 1:
            success_count += 1
        elif state == -1:
            failure_count += 1
        else:
            boundary_count += 1

        # Extract continuous observables from evidence dict.
        evidence = trial.get("evidence", {})
        boundary_eps_values.append(evidence.get("boundary_eps_final"))
        barrier_eps_values.append(evidence.get("barrier_eps_final"))
        oscillation_values.append(evidence.get("oscillation_score"))
        alignment_values.append(evidence.get("alignment_max_final"))

        # Metastability from v5.8 transition metrics.
        metastability_values.append(trial.get("metastability_score"))

        # Cluster count from trapping-set diagnostics if available.
        cluster_values.append(trial.get("cluster_count"))

        # v6.0 spectral stability diagnostics (opt-in, additive).
        spectral_radius_values.append(trial.get("spectral_radius"))
        bethe_min_values.append(trial.get("bethe_min_eigenvalue"))
        bp_stability_values.append(trial.get("bp_stability_score"))
        jacobian_est_values.append(trial.get("jacobian_spectral_radius_est"))

        # v6.1 localization diagnostics (opt-in, additive).
        nb_max_ipr_values.append(trial.get("nb_max_ipr"))
        nb_num_localized_values.append(trial.get("nb_num_localized_modes"))
        nb_top_localization_values.append(trial.get("nb_top_localization_score"))

        # v6.2 trapping-set candidate diagnostics (opt-in, additive).
        nb_candidate_nodes_values.append(trial.get("nb_num_candidate_nodes"))
        nb_max_participation_values.append(trial.get("nb_max_node_participation"))
        nb_candidate_clusters_values.append(trial.get("nb_num_candidate_clusters"))

        # v6.3 spectral-BP alignment diagnostics (opt-in, additive).
        sbpa_alignment_values.append(trial.get("spectral_bp_alignment_score"))
        sbpa_cand_overlap_values.append(trial.get("candidate_node_overlap_fraction"))
        sbpa_cluster_overlap_values.append(trial.get("max_cluster_alignment"))

        # v6.4 spectral failure risk diagnostics (opt-in, additive).
        sfr_mean_cluster_risk_values.append(trial.get("mean_cluster_risk"))
        sfr_max_cluster_risk_values.append(trial.get("max_cluster_risk"))
        sfr_num_high_risk_values.append(trial.get("num_high_risk_clusters"))

    # ── Fractions ───────────────────────────────────────────────
    n = float(trial_count)
    success_fraction = float(success_count) / n
    boundary_fraction = float(boundary_count) / n
    failure_fraction = float(failure_count) / n

    # ── Dominant phase ──────────────────────────────────────────
    # Deterministic tie-breaking: +1 > 0 > -1 priority.
    fracs = [
        (success_fraction, 1),
        (boundary_fraction, 0),
        (failure_fraction, -1),
    ]
    # Sort by fraction descending, then by phase value descending for
    # deterministic tie-breaking.
    fracs.sort(key=lambda t: (-t[0], -t[1]))
    dominant_phase = fracs[0][1]

    # ── Phase entropy ───────────────────────────────────────────
    phase_entropy = _shannon_entropy([
        success_fraction, boundary_fraction, failure_fraction,
    ])

    return {
        "x": x_val,
        "y": y_val,
        "trial_count": trial_count,
        "success_fraction": success_fraction,
        "boundary_fraction": boundary_fraction,
        "failure_fraction": failure_fraction,
        "dominant_phase": dominant_phase,
        "phase_entropy": phase_entropy,
        "mean_boundary_eps": _safe_mean(boundary_eps_values),
        "mean_barrier_eps": _safe_mean(barrier_eps_values),
        "mean_metastability_score": _safe_mean(metastability_values),
        "mean_oscillation_score": _safe_mean(oscillation_values),
        "mean_alignment_max": _safe_mean(alignment_values),
        "mean_cluster_count": _safe_mean(cluster_values),
        "mean_spectral_radius": _safe_mean(spectral_radius_values),
        "mean_bethe_min_eigenvalue": _safe_mean(bethe_min_values),
        "mean_bp_stability_score": _safe_mean(bp_stability_values),
        "mean_jacobian_spectral_radius_est": _safe_mean(jacobian_est_values),
        "mean_nb_max_ipr": _safe_mean(nb_max_ipr_values),
        "mean_nb_num_localized_modes": _safe_mean(nb_num_localized_values),
        "mean_nb_top_localization_score": _safe_mean(nb_top_localization_values),
        "mean_nb_candidate_nodes": _safe_mean(nb_candidate_nodes_values),
        "mean_nb_max_node_participation": _safe_mean(nb_max_participation_values),
        "mean_nb_candidate_clusters": _safe_mean(nb_candidate_clusters_values),
        "mean_spectral_bp_alignment": _safe_mean(sbpa_alignment_values),
        "mean_candidate_node_overlap_fraction": _safe_mean(sbpa_cand_overlap_values),
        "mean_candidate_cluster_overlap_fraction": _safe_mean(sbpa_cluster_overlap_values),
        "mean_cluster_risk": _safe_mean(sfr_mean_cluster_risk_values),
        "max_cluster_risk": _safe_mean(sfr_max_cluster_risk_values),
        "mean_num_high_risk_clusters": _safe_mean(sfr_num_high_risk_values),
    }
