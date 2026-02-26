"""
Layer 2 — Channel and noise-model abstractions.

This package provides pluggable channel models that compute per-variable
LLR vectors consumed by the benchmark runner.  Channel models must NOT
import from or mutate decoder internals (Layer 1).
"""
