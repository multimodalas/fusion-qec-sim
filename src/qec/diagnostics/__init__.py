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
and ternary decoder topology classification
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
from .bp_phase_space import compute_bp_phase_space
from .ternary_decoder_topology import compute_ternary_decoder_topology
