"""
Shared canonicalization utility.

Converts nested Python/numpy structures into deterministic, JSON-safe
plain-Python objects with sorted dict keys.  Used by both the benchmark
schema layer and the qudit spec layer to avoid logic duplication.
"""

from __future__ import annotations

import copy
from typing import Any


def canonicalize(obj: Any) -> Any:
    """Deep-copy *obj* converting numpy types to plain Python.

    * ``numpy.integer``  → ``int``
    * ``numpy.floating`` → ``float``
    * ``numpy.bool_``    → ``bool``
    * ``numpy.ndarray``  → ``list`` (recursively)
    * ``tuple``          → ``list``
    * ``dict``           → ``dict`` with sorted keys (recursive)

    The result is safe for :func:`json.dumps`.
    """
    try:
        import numpy as np
        _has_numpy = True
    except ImportError:  # pragma: no cover
        _has_numpy = False

    def _convert(v: Any) -> Any:
        if _has_numpy:
            if isinstance(v, np.ndarray):
                return [_convert(x) for x in v.tolist()]
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
        if isinstance(v, dict):
            return {str(k): _convert(val) for k, val in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [_convert(x) for x in v]
        return v

    return _convert(copy.deepcopy(obj))
