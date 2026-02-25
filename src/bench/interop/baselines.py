"""
Reference and direct baseline suite for v3.1.2 interop benchmarks.

Provides a single entry point that runs both QEC-native direct baselines
and Stim/PyMatching reference baselines (when available).

Results are schema-validated and carry the required comparability taxonomy
labels (``benchmark_kind``, ``code_family``, ``representation``).
"""

from __future__ import annotations

import datetime
from typing import Any

from .runners import run_qec_native, run_stim_pymatching_baseline
from .env import capture_environment
from .serialize import canonical_json, artifact_hash
from ..schema import validate_interop_record


def run_baseline_suite(
    *,
    # QEC-native config
    native_distances: list[int] | None = None,
    native_p_values: list[float] | None = None,
    native_trials: int = 200,
    native_max_iters: int = 50,
    # Reference baseline config
    ref_code_families: list[str] | None = None,
    ref_distances: list[int] | None = None,
    ref_p_values: list[float] | None = None,
    ref_trials: int = 1000,
    # Common
    seed: int = 12345,
    runtime_mode: str = "off",
    deterministic_metadata: bool = False,
) -> dict[str, Any]:
    """Run the complete v3.1.2 baseline suite.

    Parameters
    ----------
    native_distances, native_p_values:
        Sweep parameters for QEC-native direct baselines.
        Defaults: [3, 5] and [0.001, 0.003].
    ref_code_families:
        Code families for reference baselines.
        Defaults: ["repetition", "surface_code"].
    ref_distances, ref_p_values:
        Sweep parameters for reference baselines.
        Defaults: [3, 5] and [0.001, 0.01].
    ref_trials:
        Number of samples per reference baseline point.
    seed:
        Base seed for all RNG.
    runtime_mode:
        "off" for deterministic artifacts, "on" for timing.
    deterministic_metadata:
        If True, use fixed placeholders for environment fields.

    Returns
    -------
    Dict with keys: "direct_comparisons", "reference_baselines",
    "environment", "suite_config", "all_records".
    """
    if native_distances is None:
        native_distances = [3, 5]
    if native_p_values is None:
        native_p_values = [0.001, 0.003]
    if ref_code_families is None:
        ref_code_families = ["repetition", "surface_code"]
    if ref_distances is None:
        ref_distances = [3, 5]
    if ref_p_values is None:
        ref_p_values = [0.001, 0.01]

    env = capture_environment(deterministic=deterministic_metadata)

    # ── Direct comparisons (QEC-native) ──
    direct_records = run_qec_native(
        distances=sorted(native_distances),
        p_values=sorted(native_p_values),
        trials=native_trials,
        max_iters=native_max_iters,
        seed=seed,
        runtime_mode=runtime_mode,
        deterministic_metadata=deterministic_metadata,
    )

    # ── Reference baselines (Stim/PyMatching) ──
    ref_records: list[dict[str, Any]] = []
    for family in sorted(ref_code_families):
        records = run_stim_pymatching_baseline(
            code_family=family,
            distances=sorted(ref_distances),
            p_values=sorted(ref_p_values),
            trials=ref_trials,
            seed=seed,
            runtime_mode=runtime_mode,
            deterministic_metadata=deterministic_metadata,
        )
        ref_records.extend(records)

    # Validate all records.
    all_records = direct_records + ref_records
    for rec in all_records:
        validate_interop_record(rec)

    # Add environment to each non-skipped record.
    for rec in all_records:
        if rec.get("status") != "skipped":
            rec["environment"] = env

    suite_config = {
        "native_distances": sorted(native_distances),
        "native_max_iters": native_max_iters,
        "native_p_values": sorted(native_p_values),
        "native_trials": native_trials,
        "ref_code_families": sorted(ref_code_families),
        "ref_distances": sorted(ref_distances),
        "ref_p_values": sorted(ref_p_values),
        "ref_trials": ref_trials,
        "runtime_mode": runtime_mode,
        "seed": seed,
    }

    if deterministic_metadata:
        created_utc = "1970-01-01T00:00:00+00:00"
    else:
        created_utc = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )

    return {
        "schema_version": "3.1.2",
        "created_utc": created_utc,
        "environment": env,
        "suite_config": suite_config,
        "direct_comparisons": direct_records,
        "reference_baselines": ref_records,
        "all_records": all_records,
    }
