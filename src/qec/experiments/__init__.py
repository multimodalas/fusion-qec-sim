"""
Layer 1a — Decoder behaviour experiments (opt-in, additive).

Provides risk-aware decoder experiments that use spectral structural
signals (v6.0–v6.4) to guide decoder behaviour.  Each experiment runs
two deterministic decodes (baseline and experimental) and returns
comparison metrics.  Includes Tanner graph fragility repair (v6.6),
spectral Tanner graph optimization (v6.7), and BP prediction
validation (v6.9).

Does not modify decoder internals.  Fully deterministic.
"""

from .risk_aware_damping import run_risk_aware_damping_experiment
from .risk_guided_perturbation import run_risk_guided_perturbation
from .tanner_graph_repair import run_tanner_graph_repair_experiment
from .tanner_graph_repair import run_spectral_graph_optimization_experiment
from .bp_prediction_validation import run_bp_prediction_validation
