"""
Layer 1a — BP diagnostics (opt-in, additive).

Provides energy-landscape analysis, basin-switching detection,
iteration-trace diagnostics, BP dynamics regime analysis,
regime transition analysis, phase diagram aggregation,
freeze detection, fixed-point trap analysis, basin-of-attraction
analysis, attractor landscape mapping, free-energy barrier
estimation, pseudocodeword boundary estimation,
Tanner spectral fragility diagnostics,
spectral–boundary alignment diagnostics,
spectral trapping-set diagnostics,
BP phase-space exploration,
ternary decoder topology classification,
decoder phase diagram aggregation,
phase boundary analysis,
non-backtracking spectrum diagnostics,
Bethe Hessian spectral analysis,
BP stability proxy metrics,
BP Jacobian spectral radius estimation,
spectral trapping-set candidate detection,
and ASCII phase heatmap output
for BP convergence traces.
Does not modify decoder internals.
"""

from .bp_dynamics import compute_bp_dynamics_metrics, classify_bp_regime
from .bp_regime_trace import compute_bp_regime_trace
from .bp_phase_diagram import compute_bp_phase_diagram
from .bp_freeze_detection import compute_bp_freeze_detection
from .bp_fixed_point_analysis import compute_bp_fixed_point_analysis
from .bp_basin_analysis import compute_bp_basin_analysis
from .bp_landscape_mapping import compute_bp_landscape_map
from .bp_barrier_analysis import compute_bp_barrier_analysis
from .bp_boundary_analysis import compute_bp_boundary_analysis
from .tanner_spectral_analysis import compute_tanner_spectral_analysis
from .spectral_boundary_alignment import compute_spectral_boundary_alignment
from .spectral_trapping_sets import compute_spectral_trapping_sets
from .bp_phase_space import compute_bp_phase_space, compute_metastability_score
from .ternary_decoder_topology import compute_ternary_decoder_topology
from .basin_probe import probe_local_ternary_basin
from .phase_diagram import build_decoder_phase_diagram, make_phase_grid
from .phase_boundary_analysis import analyze_phase_boundaries
from .non_backtracking_spectrum import compute_non_backtracking_spectrum
from .bethe_hessian import compute_bethe_hessian
from .bp_stability_proxy import estimate_bp_stability
from .bp_jacobian_estimator import estimate_bp_jacobian_spectral_radius
from .nb_localization import compute_nb_localization_metrics
from .nb_trapping_candidates import compute_nb_trapping_candidates
from .phase_heatmap import print_phase_heatmap
