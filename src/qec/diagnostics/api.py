"""
v8.3.0 — Diagnostics Public API.

Exposes all public diagnostic functions from a single module.
Import from here for stable access to the diagnostics layer.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
"""

from __future__ import annotations

# ── v7.6.1: NB spectral diagnostics ──────────────────────────────
from src.qec.diagnostics.spectral_nb import (
    SPECTRAL_SCHEMA_VERSION,
    compute_nb_spectrum,
    compute_edge_sensitivity_ranking,
)

# ── v7.7.0: Spectral heatmaps ────────────────────────────────────
from src.qec.diagnostics.spectral_heatmaps import (
    compute_spectral_heatmaps,
    rank_variable_nodes_by_heat,
    rank_check_nodes_by_heat,
    rank_edges_by_heat,
)

# ── v7.9.0: Incremental spectra ──────────────────────────────────
from src.qec.diagnostics.spectral_incremental import (
    update_nb_eigenpair_incremental,
    update_nb_eigenpair_localized,
    detect_edge_swap,
    identify_affected_nb_edges,
    score_repair_candidate_incremental,
)

# ── Core diagnostics ─────────────────────────────────────────────
from src.qec.diagnostics.bp_dynamics import (
    compute_bp_dynamics_metrics,
    classify_bp_regime,
)
from src.qec.diagnostics.bp_regime_trace import compute_bp_regime_trace
from src.qec.diagnostics.bp_phase_diagram import compute_bp_phase_diagram
from src.qec.diagnostics.bp_freeze_detection import compute_bp_freeze_detection
from src.qec.diagnostics.bp_fixed_point_analysis import (
    compute_bp_fixed_point_analysis,
)
from src.qec.diagnostics.bp_basin_analysis import compute_bp_basin_analysis
from src.qec.diagnostics.bp_landscape_mapping import compute_bp_landscape_map
from src.qec.diagnostics.bp_barrier_analysis import compute_bp_barrier_analysis
from src.qec.diagnostics.bp_boundary_analysis import compute_bp_boundary_analysis
from src.qec.diagnostics.tanner_spectral_analysis import (
    compute_tanner_spectral_analysis,
)
from src.qec.diagnostics.spectral_boundary_alignment import (
    compute_spectral_boundary_alignment,
)
from src.qec.diagnostics.spectral_trapping_sets import (
    compute_spectral_trapping_sets,
)
from src.qec.diagnostics.bp_phase_space import (
    compute_bp_phase_space,
    compute_metastability_score,
)
from src.qec.diagnostics.ternary_decoder_topology import (
    compute_ternary_decoder_topology,
)
from src.qec.diagnostics.basin_probe import probe_local_ternary_basin
from src.qec.diagnostics.phase_diagram import (
    build_decoder_phase_diagram,
    make_phase_grid,
)
from src.qec.diagnostics.phase_boundary_analysis import analyze_phase_boundaries
from src.qec.diagnostics.non_backtracking_spectrum import (
    compute_non_backtracking_spectrum,
)
from src.qec.diagnostics.bethe_hessian import compute_bethe_hessian
from src.qec.diagnostics.bp_stability_proxy import estimate_bp_stability
from src.qec.diagnostics.bp_jacobian_estimator import (
    estimate_bp_jacobian_spectral_radius,
)
from src.qec.diagnostics.nb_localization import compute_nb_localization_metrics
from src.qec.diagnostics.nb_trapping_candidates import (
    compute_nb_trapping_candidates,
)
from src.qec.diagnostics.spectral_bp_alignment import (
    compute_spectral_bp_alignment,
)
from src.qec.diagnostics.spectral_failure_risk import (
    compute_spectral_failure_risk,
)
from src.qec.diagnostics.phase_heatmap import print_phase_heatmap
from src.qec.diagnostics.bp_stability_predictor import (
    compute_bp_stability_prediction,
)
from src.qec.diagnostics.sensitivity_map import (
    compute_proxy_sensitivity_scores,
    compute_measured_instability_deltas,
    compute_sensitivity_map,
)
from src.qec.diagnostics.nb_localization_detector import (
    detect_nb_eigenvector_localization,
)
from src.qec.diagnostics.nb_energy_heatmap import compute_nb_energy_heatmap
from src.qec.diagnostics.nb_sign_pattern_detector import (
    detect_nb_sign_pattern_trapping_sets,
)

# ── v8.1.0: Spectral stability diagnostics ───────────────────────
from src.qec.diagnostics.spectral_entropy import compute_spectral_entropy
from src.qec.diagnostics.nb_spectral_gap import compute_nb_spectral_gap
from src.qec.diagnostics.bethe_hessian_margin import compute_bethe_hessian_margin
from src.qec.diagnostics.effective_support_dimension import (
    compute_effective_support_dimension,
)
from src.qec.diagnostics.spectral_curvature import compute_spectral_curvature
from src.qec.diagnostics.cycle_space_density import compute_cycle_space_density
from src.qec.diagnostics.spectral_metrics import compute_spectral_metrics
from src.qec.diagnostics.stability_classifier import (
    classify_tanner_graph_stability,
    classify_from_parity_check,
)

# ── v8.2.0: Stability boundary & spectral prediction ──────────────
from src.qec.diagnostics.stability_boundary import estimate_stability_boundary
from src.qec.diagnostics.stability_predictor import predict_bp_stability
from src.qec.diagnostics.nb_sign_trapping_sets import detect_nb_sign_trapping_sets
from src.qec.diagnostics.critical_radius import estimate_critical_spectral_radius
from src.qec.diagnostics.spectral_critical_line import (
    predict_spectral_critical_radius,
)

# ── v8.3.0: Stability optimization & spectral invariant discovery ──
from src.qec.diagnostics.stability_optimizer import (
    optimize_tanner_graph_stability,
)
from src.qec.diagnostics.repair_candidates import generate_repair_candidates
from src.qec.diagnostics.repair_scoring import score_repair_candidate
from src.qec.diagnostics.spectral_invariant_discovery import (
    discover_spectral_invariants,
)
