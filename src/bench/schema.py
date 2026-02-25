"""
Versioned JSON result schema for benchmark outputs.

Provides canonicalization (numpy → Python), validation, and deterministic
serialization with stable key ordering and compact separators.
"""

from __future__ import annotations

import json
from typing import Any

from ..utils.canonicalize import canonicalize  # noqa: F401 — re-exported

SCHEMA_VERSION = "3.0.1"

# v3.1.2 interop schema version (used by bench/interop records).
INTEROP_SCHEMA_VERSION = "3.1.2"

# Schema versions accepted by the validator.
_SUPPORTED_SCHEMA_VERSIONS = {"3.0.0", "3.0.1"}

# Schema versions accepted by the interop validator.
_SUPPORTED_INTEROP_VERSIONS = {"3.1.2"}

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


# ── v3.1.2 Interop Record Schema ────────────────────────────────────

_VALID_BENCHMARK_KINDS = {"direct_comparison", "reference_baseline"}
_VALID_CODE_FAMILIES = {"qldpc_css", "surface_code", "repetition", "other"}
_VALID_REPRESENTATIONS = {"pcm", "stim_circuit", "stim_dem", "other"}
_VALID_TOOL_CATEGORIES = {"native", "third_party"}

# Required keys for a v3.1.2 interop record.
_REQUIRED_INTEROP_KEYS: dict[str, type | tuple[type, ...]] = {
    "schema_version": str,
    "tool": dict,
    "benchmark_kind": str,
    "code_family": str,
    "representation": str,
    "seed": (int, float),
    "noise_model": str,
    "trials": int,
    "results": dict,
    "determinism": dict,
}

# Required keys inside the "tool" sub-dict.
_REQUIRED_TOOL_KEYS = {"name", "version", "category"}

# Required keys inside the "determinism" sub-dict.
_REQUIRED_DETERMINISM_KEYS = {"canonical_json", "stable_sweep_hash", "artifact_hash"}


def validate_interop_record(record: Any) -> None:
    """Validate that *record* conforms to the v3.1.2 interop schema.

    Raises :class:`ValueError` with a descriptive message on failure.
    Skipped records (with ``"status": "skipped"``) are accepted without
    full validation.
    """
    if not isinstance(record, dict):
        raise ValueError(f"Record must be a dict, got {type(record).__name__}")

    # Skipped records are valid (tool unavailable).
    if record.get("status") == "skipped":
        return

    for key, expected in _REQUIRED_INTEROP_KEYS.items():
        if key not in record:
            raise ValueError(f"Missing required key: {key!r}")
        if not isinstance(record[key], expected):
            raise ValueError(
                f"Key {key!r} must be {expected}, "
                f"got {type(record[key]).__name__}"
            )

    sv = record["schema_version"]
    if sv not in _SUPPORTED_INTEROP_VERSIONS:
        raise ValueError(
            f"Unsupported interop schema_version: {sv!r} "
            f"(expected one of {sorted(_SUPPORTED_INTEROP_VERSIONS)})"
        )

    # Validate enum fields.
    bk = record["benchmark_kind"]
    if bk not in _VALID_BENCHMARK_KINDS:
        raise ValueError(
            f"Invalid benchmark_kind: {bk!r} "
            f"(expected one of {sorted(_VALID_BENCHMARK_KINDS)})"
        )

    cf = record["code_family"]
    if cf not in _VALID_CODE_FAMILIES:
        raise ValueError(
            f"Invalid code_family: {cf!r} "
            f"(expected one of {sorted(_VALID_CODE_FAMILIES)})"
        )

    rep = record["representation"]
    if rep not in _VALID_REPRESENTATIONS:
        raise ValueError(
            f"Invalid representation: {rep!r} "
            f"(expected one of {sorted(_VALID_REPRESENTATIONS)})"
        )

    # Validate tool sub-dict.
    tool = record["tool"]
    missing_tool = _REQUIRED_TOOL_KEYS - set(tool.keys())
    if missing_tool:
        raise ValueError(f"tool missing required keys: {sorted(missing_tool)}")

    cat = tool["category"]
    if cat not in _VALID_TOOL_CATEGORIES:
        raise ValueError(
            f"Invalid tool category: {cat!r} "
            f"(expected one of {sorted(_VALID_TOOL_CATEGORIES)})"
        )

    # Validate determinism sub-dict.
    det = record["determinism"]
    missing_det = _REQUIRED_DETERMINISM_KEYS - set(det.keys())
    if missing_det:
        raise ValueError(
            f"determinism missing required keys: {sorted(missing_det)}"
        )

    # Validate optional channel_model field (if present, must be string).
    if "channel_model" in record:
        if not isinstance(record["channel_model"], str):
            raise ValueError(
                f"channel_model must be str, "
                f"got {type(record['channel_model']).__name__}"
            )

    # Validate results has logical_error_rate.
    results = record["results"]
    if "logical_error_rate" not in results:
        raise ValueError("results must contain 'logical_error_rate'")
