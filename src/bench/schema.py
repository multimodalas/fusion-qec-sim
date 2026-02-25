"""
Versioned JSON result schema for benchmark outputs.

Provides canonicalization (numpy → Python), validation, and deterministic
serialization with stable key ordering and compact separators.
"""

from __future__ import annotations

import copy
import json
from typing import Any

SCHEMA_VERSION = "3.0.1"

# Schema versions accepted by the validator.
_SUPPORTED_SCHEMA_VERSIONS = {"3.0.0", "3.0.1"}

# Required top-level keys and their expected Python types.
_REQUIRED_TOP_KEYS: dict[str, type | tuple[type, ...]] = {
    "schema_version": str,
    "created_utc": str,
    "environment": dict,
    "config": dict,
    "results": list,
    "summaries": dict,
}

# Required per-record keys inside each results[] entry.
_REQUIRED_RECORD_KEYS: set[str] = {
    "decoder",
    "distance",
    "p",
    "fer",
    "wer",
    "mean_iters",
}


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
    # Import numpy lazily so schema.py can be imported even without numpy
    # (although the project always has it).
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


def validate_result(obj: Any) -> None:
    """Validate that *obj* conforms to the v3.0.0 benchmark result schema.

    Raises :class:`ValueError` with a descriptive message on failure.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Result must be a dict, got {type(obj).__name__}")

    for key, expected in _REQUIRED_TOP_KEYS.items():
        if key not in obj:
            raise ValueError(f"Missing required top-level key: {key!r}")
        if not isinstance(obj[key], expected):
            raise ValueError(
                f"Key {key!r} must be {expected}, "
                f"got {type(obj[key]).__name__}"
            )

    sv = obj.get("schema_version")
    if sv not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version: {sv!r} "
            f"(expected one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)})"
        )

    for i, rec in enumerate(obj["results"]):
        if not isinstance(rec, dict):
            raise ValueError(f"results[{i}] must be a dict")
        missing = _REQUIRED_RECORD_KEYS - set(rec.keys())
        if missing:
            raise ValueError(
                f"results[{i}] missing required keys: {sorted(missing)}"
            )


def dumps_result(obj: Any) -> str:
    """Serialize a benchmark result to a deterministic JSON string.

    Uses ``sort_keys=True`` and compact separators ``(",", ":")`` to
    guarantee byte-identical output for the same logical content.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
