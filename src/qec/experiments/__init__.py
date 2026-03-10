"""
Layer 1a — Decoder behaviour experiments (opt-in, additive).

Provides risk-aware decoder experiments that use spectral structural
signals (v6.0–v6.4) to guide decoder behaviour.  Each experiment runs
two deterministic decodes (baseline and experimental) and returns
comparison metrics.

Does not modify decoder internals.  Fully deterministic.
"""

from .risk_aware_damping import run_risk_aware_damping_experiment
from .risk_guided_perturbation import run_risk_guided_perturbation
