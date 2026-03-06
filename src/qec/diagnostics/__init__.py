"""
Layer 1a — BP diagnostics (opt-in, additive).

Provides energy-landscape analysis, basin-switching detection,
iteration-trace diagnostics, BP dynamics regime analysis,
regime transition analysis, phase diagram aggregation,
freeze detection, and fixed-point trap analysis for BP convergence traces.
Does not modify decoder internals.
"""

from .bp_dynamics import compute_bp_dynamics_metrics, classify_bp_regime
from .bp_regime_trace import compute_bp_regime_trace
from .bp_phase_diagram import compute_bp_phase_diagram
from .bp_freeze_detection import compute_bp_freeze_detection
from .bp_fixed_point_analysis import compute_bp_fixed_point_analysis
