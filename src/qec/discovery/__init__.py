"""
v9.0.0 — Deterministic QLDPC Structure Discovery Engine.

Extends the Tanner graph generation framework into a deterministic
structure discovery system including mutation operators, repair
mechanisms, multi-objective ranking, novelty tracking, guided mutation
via cycle-pressure heatmaps, spectral bad-edge detection, and ACE-gated
mutation filtering.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from .api import *  # noqa: F401,F403
