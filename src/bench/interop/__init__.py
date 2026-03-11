"""
Interop layer for optional third-party QEC tool integration (v3.1.2).

This package isolates all third-party imports (Stim, PyMatching) behind
try/except gates.  No module outside ``bench/interop/`` may import these
tools.  Core QEC functionality is unaffected by their presence or absence.

Submodules
----------
imports    : Gated import helpers and version probing.
runners    : Tool wrappers (qec_native, stim, pymatching).
env        : Deterministic environment capture.
serialize  : Canonical JSON writer and artifact hash computation.
"""

__all__: list[str] = []
