"""
Layer 1a — BP diagnostics (opt-in, additive).

Provides energy-landscape analysis, basin-switching detection,
iteration-trace diagnostics, BP dynamics regime analysis,
regime transition analysis, phase diagram aggregation,
freeze detection, fixed-point trap analysis, basin-of-attraction
analysis, attractor landscape mapping, free-energy barrier
estimation, and pseudocodeword boundary estimation
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
