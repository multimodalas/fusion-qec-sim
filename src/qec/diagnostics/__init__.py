"""
Layer 1a — BP diagnostics (opt-in, additive).

Provides energy-landscape analysis, basin-switching detection,
iteration-trace diagnostics, BP dynamics regime analysis,
regime transition analysis, and phase diagram aggregation
for BP convergence traces.
Does not modify decoder internals.
"""

from .bp_dynamics import compute_bp_dynamics_metrics, classify_bp_regime
from .bp_regime_trace import compute_bp_regime_trace
from .bp_phase_diagram import compute_bp_phase_diagram
