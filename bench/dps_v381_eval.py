"""
v4.0.0 — BP Free-Energy Landscape Diagnostics.

Extends v3.9.1 harness with:
  - Energy landscape classification per trial
  - Geometry-induced basin switching detection
  - Aggregate basin statistics per mode

Evaluates Distance Penalty Slope (DPS) across fourteen modes:
  baseline           — flooding, all interventions disabled
  rpc_only           — flooding, RPC augmentation
  geom_v1_only       — geom_v1 schedule
  rpc_geom           — geom_v1 + RPC
  centered           — centered field projection
  prior              — pseudo-prior injection
  centered_prior     — centered field + pseudo-prior
  geom_centered      — geom_v1 + centered field
  geom_centered_prior — geom_v1 + centered field + pseudo-prior
  rpc_centered       — RPC + centered field
  rpc_centered_prior — RPC + centered field + pseudo-prior
  centered_strong    — centered field + geometry_strength=2.0
  centered_normalized — centered field + normalize_geometry
  centered_prior_normalized — centered + prior + normalize_geometry

FER uses syndrome-consistency semantics:
  frame_error := syndrome(H, correction) != syndrome(H, error)

Deterministic: fixed seed, pre-generated error instances reused across
all modes.  No benchmark caching.  Deterministic loop ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from typing import Any

import numpy as np

from src.qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code
from src.qec.decoder.rpc import RPCConfig, StructuralConfig, build_rpc_augmented_system

from src.qec.channel.geometry import (
    centered_syndrome_field,
    syndrome_field,
    pseudo_prior_bias,
    apply_pseudo_prior,
)

from src.qec.channel.geometry_post import apply_geometry_postprocessing

from src.qec.diagnostics.energy_landscape import (
    classify_energy_landscape,
    classify_basin_switch,
    compute_landscape_metrics,
    detect_basin_switch,
)

from src.qec.diagnostics.iteration_trace import (
    compute_iteration_trace_metrics,
)

# ── Mode definitions ─────────────────────────────────────────────────

_RPC_ON = RPCConfig(enabled=True, max_rows=64, w_min=2, w_max=32)
_RPC_OFF = RPCConfig(enabled=False)

MODES: dict[str, dict[str, Any]] = {
    "baseline": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_OFF),
    },
    "rpc_only": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_ON),
    },
    "geom_v1_only": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(rpc=_RPC_OFF),
    },
    "rpc_geom": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(rpc=_RPC_ON),
    },
    # ── v3.9.0 channel-geometry modes ──
    "centered": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_OFF, centered_field=True),
    },
    "prior": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_OFF, pseudo_prior=True),
    },
    "centered_prior": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True, pseudo_prior=True,
        ),
    },
    "geom_centered": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(rpc=_RPC_OFF, centered_field=True),
    },
    "geom_centered_prior": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True, pseudo_prior=True,
        ),
    },
    "rpc_centered": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_ON, centered_field=True),
    },
    "rpc_centered_prior": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_ON, centered_field=True, pseudo_prior=True,
        ),
    },
    # ── v3.9.1 geometry field control modes ──
    "centered_strong": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True,
            pseudo_prior=False, geometry_strength=2.0,
        ),
    },
    "centered_normalized": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True,
            normalize_geometry=True,
        ),
    },
    "centered_prior_normalized": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True,
            pseudo_prior=True, normalize_geometry=True,
        ),
    },
}

MODE_ORDER = [
    "baseline", "rpc_only", "geom_v1_only", "rpc_geom",
    "centered", "prior", "centered_prior",
    "geom_centered", "geom_centered_prior",
    "rpc_centered", "rpc_centered_prior",
    "centered_strong", "centered_normalized", "centered_prior_normalized",
]

# ── Default parameters ───────────────────────────────────────────────

DEFAULT_SEED = 42
DEFAULT_DISTANCES = [3, 5, 7]
DEFAULT_P_VALUES = [0.01, 0.015, 0.02]
DEFAULT_TRIALS = 200
DEFAULT_MAX_ITERS = 50
DEFAULT_BP_MODE = "min_sum"


# ── Helpers ──────────────────────────────────────────────────────────

def _array_checksum(arr: np.ndarray) -> str:
    """Deterministic SHA-256 hex digest (first 12 chars) of an array."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def _pre_generate_instances(
    H: np.ndarray,
    p: float,
    trials: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Pre-generate error instances for reuse across modes."""
    n = H.shape[1]
    instances = []
    for _ in range(trials):
        e = (rng.random(n) < p).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, p)
        instances.append({"e": e, "s": s, "llr": llr})
    return instances


# ── Per-trial activation audit record ────────────────────────────────

def _trial_audit(
    H_original: np.ndarray,
    H_used: np.ndarray,
    s_original: np.ndarray,
    s_used: np.ndarray,
    iters: int,
    s_weight: int,
    e_weight: int,
) -> dict[str, Any]:
    """Collect per-trial activation audit data."""
    original_rows = H_original.shape[0]
    augmented_rows = H_used.shape[0]
    added_rows = augmented_rows - original_rows
    return {
        "original_rows": original_rows,
        "augmented_rows": augmented_rows,
        "added_rows": added_rows,
        "H_checksum": _array_checksum(H_used),
        "syndrome_checksum": _array_checksum(s_used),
        "iter_count": iters,
        "syndrome_weight": s_weight,
        "error_weight": e_weight,
    }


def _summarize_audit(trials_audit: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from per-trial audit records."""
    added = [t["added_rows"] for t in trials_audit]
    augmented = [t["augmented_rows"] for t in trials_audit]
    iters_list = [t["iter_count"] for t in trials_audit]
    syndrome_weights = [t["syndrome_weight"] for t in trials_audit]

    n_trials = len(trials_audit)
    fraction_iters_eq_1 = sum(1 for i in iters_list if i == 1) / n_trials if n_trials else 0.0
    fraction_zero_syndrome = sum(1 for w in syndrome_weights if w == 0) / n_trials if n_trials else 0.0

    return {
        "added_rows_min": min(added),
        "added_rows_mean": sum(added) / n_trials,
        "added_rows_max": max(added),
        "augmented_rows_min": min(augmented),
        "augmented_rows_mean": sum(augmented) / n_trials,
        "augmented_rows_max": max(augmented),
        "H_checksum_first": trials_audit[0]["H_checksum"],
        "H_checksum_last": trials_audit[-1]["H_checksum"],
        "syndrome_checksum_first": trials_audit[0]["syndrome_checksum"],
        "syndrome_checksum_last": trials_audit[-1]["syndrome_checksum"],
        "mean_iters": sum(iters_list) / n_trials,
        "max_iters_observed": max(iters_list),
        "fraction_iters_eq_1": fraction_iters_eq_1,
        "fraction_zero_syndrome": fraction_zero_syndrome,
    }


# ── Single mode evaluation ──────────────────────────────────────────

def run_mode(
    mode_name: str,
    H: np.ndarray,
    instances: list[dict[str, Any]],
    max_iters: int = DEFAULT_MAX_ITERS,
    bp_mode: str = DEFAULT_BP_MODE,
    enable_energy_trace: bool = False,
    enable_landscape: bool = False,
    enable_iteration_diagnostics: bool = False,
) -> dict[str, Any]:
    """Run a single mode over pre-generated instances.

    Returns dict with keys: fer, frame_errors, trials, audit_summary.
    When *enable_energy_trace* or *enable_landscape* is True, also
    returns ``energy_traces``.  When *enable_landscape* is True, also
    returns ``landscape_metrics``, ``basin_switch``, and
    ``energy_delta``.  When *enable_iteration_diagnostics* is True,
    also returns ``iteration_diagnostics``.
    """
    mode_cfg = MODES[mode_name]
    schedule = mode_cfg["schedule"]
    structural: StructuralConfig = mode_cfg["structural"]

    # Landscape mode implies energy trace.
    if enable_landscape:
        enable_energy_trace = True
    # Iteration diagnostics implies energy trace.
    if enable_iteration_diagnostics:
        enable_energy_trace = True

    frame_errors = 0
    residual_mismatches = 0
    trials_audit: list[dict[str, Any]] = []
    all_energy_traces: list[list[float]] = []
    all_landscape_metrics: list[dict[str, Any]] = []
    basin_switches = 0
    energy_deltas: list[float] = []
    all_basin_classifications: list[dict[str, Any]] = []
    all_iteration_diagnostics: list[dict[str, Any]] = []

    for inst in instances:
        e = inst["e"]
        s = inst["s"]
        llr = inst["llr"]

        # Apply RPC augmentation if enabled.
        H_used = H
        s_used = s
        if structural.rpc.enabled:
            H_used, s_used = build_rpc_augmented_system(
                H, s, structural.rpc,
            )

          # ── Channel geometry interventions ──
        llr_used = llr
        if structural.centered_field:
            llr_used = centered_syndrome_field(H_used, s_used)
        elif structural.pseudo_prior:
            llr_used = syndrome_field(H_used, s_used)

        if structural.pseudo_prior:
            bias = pseudo_prior_bias(H_used, s_used)
            llr_used = apply_pseudo_prior(
                llr_used, bias, structural.pseudo_prior_strength,
            )

        llr_used = apply_geometry_postprocessing(llr_used, structural)
      
        # Decode.
        use_energy = enable_energy_trace or structural.energy_trace
        use_llr_history = max_iters if enable_iteration_diagnostics else 0
        result = bp_decode(
            H_used, llr_used,
            max_iters=max_iters,
            mode=bp_mode,
            schedule=schedule,
            syndrome_vec=s_used,
            energy_trace=use_energy,
            llr_history=use_llr_history,
        )
        correction, iters = result[0], result[1]
        # Extract optional outputs based on what was requested.
        llr_hist = None
        trace = None
        if use_llr_history > 0 and use_energy:
            # Return order: (correction, iters, llr_history, energy_trace)
            llr_hist = result[2]
            trace = result[-1]
        elif use_llr_history > 0:
            llr_hist = result[2]
        elif use_energy:
            trace = result[-1]

        if trace is not None:
            all_energy_traces.append(trace)

            if enable_landscape and len(trace) >= 2:
                all_landscape_metrics.append(classify_energy_landscape(trace))

                basin = detect_basin_switch(
                    H_used, llr_used, correction, trace,
                    max_iters, bp_mode, schedule, s_used,
                )
                if basin["switch"]:
                    basin_switches += 1
                energy_deltas.append(abs(basin["energy_base"] - basin["energy_perturbed"]))

                # v4.2.0: landscape metrics (BSI, AD, EE) + classification.
                classification = compute_landscape_metrics(
                    H_used, llr_used, correction, trace,
                    max_iters, bp_mode, schedule, s_used,
                )
                all_basin_classifications.append(classification)

        # v4.3.0: iteration-trace diagnostics.
        if enable_iteration_diagnostics and trace is not None:
            llr_trace_list = (
                [llr_hist[i] for i in range(llr_hist.shape[0])]
                if llr_hist is not None and llr_hist.shape[0] > 0
                else []
            )
            iter_diag = compute_iteration_trace_metrics(
                llr_trace=llr_trace_list,
                energy_trace=list(trace),
                correction_vectors=llr_trace_list,
            )
            all_iteration_diagnostics.append(iter_diag)

        # FER: syndrome-consistency semantics.
        # A frame error occurs when syndrome(H_original, correction) != s_original.
        s_correction = syndrome(H, correction)
        is_frame_error = not np.array_equal(s_correction, s)
        if is_frame_error:
            frame_errors += 1

        # Residual check (diagnostic only).
        residual = np.asarray(e) ^ np.asarray(correction)
        if np.any(residual):
            residual_mismatches += 1

        # Per-trial audit.
        s_weight = int(np.sum(s))
        e_weight = int(np.sum(e))
        audit = _trial_audit(H, H_used, s, s_used, iters, s_weight, e_weight)
        trials_audit.append(audit)

    n_trials = len(instances)
    fer = float(frame_errors) / n_trials if n_trials else 0.0

    out: dict[str, Any] = {
        "fer": fer,
        "frame_errors": frame_errors,
        "residual_mismatches": residual_mismatches,
        "trials": n_trials,
        "audit_summary": _summarize_audit(trials_audit),
    }
    if all_energy_traces:
        out["energy_traces"] = all_energy_traces
    if all_landscape_metrics:
        out["landscape_metrics"] = all_landscape_metrics
        n = len(instances)
        out["basin_switch"] = basin_switches > 0
        out["basin_switch_fraction"] = basin_switches / n if n else 0.0
        out["energy_delta"] = (
            sum(energy_deltas) / len(energy_deltas)
            if energy_deltas else 0.0
        )
    if all_basin_classifications:
        out["basin_classifications"] = all_basin_classifications
        # Aggregate class counts.
        class_counts: dict[str, int] = {}
        for bc in all_basin_classifications:
            cls = bc["basin_switch_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        out["basin_class_counts"] = class_counts
    if all_iteration_diagnostics:
        out["iteration_diagnostics"] = all_iteration_diagnostics
    return out


# ── DPS slope computation ───────────────────────────────────────────

_LOG_EPS = 1e-30


def compute_dps_slope(
    fer_by_distance: dict[int, float],
    eps: float = _LOG_EPS,
) -> float:
    """Compute linear slope of log10(FER + eps) vs distance."""
    distances = sorted(fer_by_distance.keys())
    if len(distances) < 2:
        return 0.0
    d_arr = np.array(distances, dtype=np.float64)
    lf_arr = np.array(
        [math.log10(fer_by_distance[d] + eps) for d in distances],
        dtype=np.float64,
    )
    A = np.vstack([d_arr, np.ones(len(d_arr))]).T
    coef, _, _, _ = np.linalg.lstsq(A, lf_arr, rcond=None)
    return float(coef[0])


# ── Full evaluation ─────────────────────────────────────────────────

def run_evaluation(
    seed: int = DEFAULT_SEED,
    distances: list[int] | None = None,
    p_values: list[float] | None = None,
    trials: int = DEFAULT_TRIALS,
    max_iters: int = DEFAULT_MAX_ITERS,
    bp_mode: str = DEFAULT_BP_MODE,
    enable_energy_trace: bool = False,
    enable_landscape: bool = False,
    enable_iteration_diagnostics: bool = False,
) -> dict[str, Any]:
    """Run the full DPS evaluation across all modes, distances, and p values.

    Returns a dict with:
      results[mode_name][p][distance] = run_mode result
      slopes[mode_name][p] = DPS slope
      audit[mode_name][p][distance] = audit_summary
    """
    if distances is None:
        distances = list(DEFAULT_DISTANCES)
    if p_values is None:
        p_values = list(DEFAULT_P_VALUES)

    distances = sorted(distances)
    p_values = sorted(p_values)

    results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
    slopes: dict[str, dict[float, float]] = {}
    audits: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}

    # Pre-generate all error instances per (distance, p), keyed for reuse.
    rng = np.random.default_rng(seed)
    codes: dict[int, np.ndarray] = {}
    all_instances: dict[tuple[int, float], list[dict[str, Any]]] = {}

    for distance in distances:
        H = create_code(name="rate_0.50", lifting_size=distance, seed=seed).H_X
        codes[distance] = H
        for p in p_values:
            all_instances[(distance, p)] = _pre_generate_instances(H, p, trials, rng)

    # Deterministic loop order: modes → p_values → distances.
    for mode_name in MODE_ORDER:
        results[mode_name] = {}
        slopes[mode_name] = {}
        audits[mode_name] = {}

        for p in p_values:
            fer_by_distance: dict[int, float] = {}
            results[mode_name][p] = {}
            audits[mode_name][p] = {}

            for distance in distances:
                H = codes[distance]
                instances = all_instances[(distance, p)]

                result = run_mode(
                    mode_name, H, instances,
                    max_iters=max_iters, bp_mode=bp_mode,
                    enable_energy_trace=enable_energy_trace,
                    enable_landscape=enable_landscape,
                    enable_iteration_diagnostics=enable_iteration_diagnostics,
                )
                results[mode_name][p][distance] = result
                audits[mode_name][p][distance] = result["audit_summary"]
                fer_by_distance[distance] = result["fer"]

            slopes[mode_name][p] = compute_dps_slope(fer_by_distance)

    return {
        "results": results,
        "slopes": slopes,
        "audits": audits,
        "config": {
            "seed": seed,
            "distances": distances,
            "p_values": p_values,
            "trials": trials,
            "max_iters": max_iters,
            "bp_mode": bp_mode,
        },
    }


# ── Determinism check ────────────────────────────────────────────────

def run_determinism_check(
    seed: int = DEFAULT_SEED,
    distance: int = 3,
    p: float = 0.01,
    trials: int = 50,
    max_iters: int = DEFAULT_MAX_ITERS,
    bp_mode: str = DEFAULT_BP_MODE,
) -> dict[str, Any]:
    """Run a single configuration twice and compare results."""
    results = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        H = create_code(name="rate_0.50", lifting_size=distance, seed=seed).H_X
        instances = _pre_generate_instances(H, p, trials, rng)
        # Run baseline mode.
        r = run_mode("baseline", H, instances, max_iters=max_iters, bp_mode=bp_mode)
        results.append(r)

    r1, r2 = results
    fer_match = r1["fer"] == r2["fer"]
    audit_match = r1["audit_summary"] == r2["audit_summary"]
    passed = fer_match and audit_match

    return {
        "passed": passed,
        "fer_match": fer_match,
        "audit_match": audit_match,
        "fer_run1": r1["fer"],
        "fer_run2": r2["fer"],
    }


# ── Printing ─────────────────────────────────────────────────────────

def print_activation_report(eval_result: dict[str, Any]) -> None:
    """Print the activation audit summary for all modes."""
    audits = eval_result["audits"]
    config = eval_result["config"]
    distances = config["distances"]
    p_values = config["p_values"]

    print("\n" + "=" * 72)
    print("ACTIVATION AUDIT REPORT")
    print("=" * 72)

    for mode_name in MODE_ORDER:
        for p in p_values:
            for distance in distances:
                audit = audits[mode_name][p][distance]
                print(f"\n--- {mode_name} | p={p} | d={distance} ---")
                print(f"  added_rows:     min={audit['added_rows_min']}"
                      f"  mean={audit['added_rows_mean']:.2f}"
                      f"  max={audit['added_rows_max']}")
                print(f"  augmented_rows: min={audit['augmented_rows_min']}"
                      f"  mean={audit['augmented_rows_mean']:.2f}"
                      f"  max={audit['augmented_rows_max']}")
                print(f"  H_checksum:     first={audit['H_checksum_first']}"
                      f"  last={audit['H_checksum_last']}")
                print(f"  synd_checksum:  first={audit['syndrome_checksum_first']}"
                      f"  last={audit['syndrome_checksum_last']}")
                print(f"  mean_iters:     {audit['mean_iters']:.2f}")
                print(f"  max_iters:      {audit['max_iters_observed']}")
                print(f"  frac_iters==1:  {audit['fraction_iters_eq_1']:.4f}")
                print(f"  frac_zero_synd: {audit['fraction_zero_syndrome']:.4f}")

                # Warnings.
                rpc_enabled = MODES[mode_name]["structural"].rpc.enabled
                if rpc_enabled and audit["added_rows_mean"] == 0:
                    print("  WARNING: RPC enabled but added_rows_mean == 0")
                if audit["fraction_iters_eq_1"] > 0.95:
                    print("  WARNING: fraction_iters_eq_1 > 0.95")
                expected_schedule = MODES[mode_name]["schedule"]
                structural = MODES[mode_name]["structural"]
                # Schedule mismatch can't happen in this harness, but flag
                # if the mode definition is inconsistent.
                if expected_schedule not in ("flooding", "geom_v1"):
                    print(f"  WARNING: unexpected schedule {expected_schedule}")
                if structural.centered_field:
                    print(f"  centered_field: ON")
                if structural.pseudo_prior:
                    print(f"  pseudo_prior: ON (kappa={structural.pseudo_prior_strength})")


def print_dps_table(eval_result: dict[str, Any]) -> None:
    """Print the DPS slope table."""
    slopes = eval_result["slopes"]
    config = eval_result["config"]
    p_values = config["p_values"]

    print("\n" + "=" * 72)
    print("DPS SLOPE TABLE")
    print("=" * 72)

    # Header.
    p_headers = [f"p={p}" for p in p_values]
    header = f"{'Mode':<16}" + "".join(f"{h:>12}" for h in p_headers) + "  Inverted?"
    print(header)
    print("-" * len(header))

    for mode_name in MODE_ORDER:
        row = f"{mode_name:<16}"
        any_inverted = False
        for p in p_values:
            slope = slopes[mode_name][p]
            row += f"{slope:>12.4f}"
            if slope > 0:
                any_inverted = True
        inv_marker = "  INVERTED" if any_inverted else ""
        print(row + inv_marker)


def print_determinism_result(det_result: dict[str, Any]) -> None:
    """Print determinism check result."""
    print("\n" + "=" * 72)
    status = "PASS" if det_result["passed"] else "FAIL"
    print(f"Determinism: {status}")
    if not det_result["passed"]:
        print(f"  FER match:   {det_result['fer_match']}")
        print(f"  Audit match: {det_result['audit_match']}")
        print(f"  FER run1:    {det_result['fer_run1']}")
        print(f"  FER run2:    {det_result['fer_run2']}")
    print("=" * 72)


# ── Main ─────────────────────────────────────────────────────────────

def print_energy_trace(eval_result: dict[str, Any]) -> None:
    """Print energy trace summary for modes that have it."""
    results = eval_result["results"]
    config = eval_result["config"]
    p_values = config["p_values"]
    distances = config["distances"]

    has_any = False
    for mode_name in MODE_ORDER:
        for p in p_values:
            for distance in distances:
                r = results[mode_name][p][distance]
                if "energy_traces" in r and r["energy_traces"]:
                    if not has_any:
                        print("\n" + "=" * 72)
                        print("ENERGY TRACE SUMMARY")
                        print("=" * 72)
                        has_any = True
                    traces = r["energy_traces"]
                    # Show first trial's trace.
                    first = traces[0]
                    print(f"\n--- {mode_name} | p={p} | d={distance} ---")
                    print("[ENERGY TRACE]")
                    for i, e in enumerate(first):
                        print(f"iter {i} : {e:.2f}")


def print_basin_statistics(eval_result: dict[str, Any]) -> None:
    """Print aggregate basin switching statistics per mode."""
    results = eval_result["results"]
    config = eval_result["config"]
    p_values = config["p_values"]
    distances = config["distances"]

    has_any = False
    for mode_name in MODE_ORDER:
        for p in p_values:
            for distance in distances:
                r = results[mode_name][p][distance]
                if "basin_switch_fraction" not in r:
                    continue
                if not has_any:
                    print("\n" + "=" * 72)
                    print("BASIN SWITCHING STATISTICS")
                    print("=" * 72)
                    has_any = True
                frac = r["basin_switch_fraction"]
                delta = r["energy_delta"]
                print(f"  {mode_name:<30} p={p}  d={distance}"
                      f"  switch_frac={frac:.4f}"
                      f"  mean_delta={delta:.6f}")
                if "basin_class_counts" in r:
                    counts = r["basin_class_counts"]
                    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
                    print(f"    classification: {', '.join(parts)}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="v4.0.0 — BP Free-Energy Landscape Diagnostics",
    )
    parser.add_argument("--landscape", action="store_true",
                        help="Enable landscape diagnostics and basin switching")
    parser.add_argument("--iteration-diagnostics", action="store_true",
                        help="Enable iteration-trace diagnostics (PEI, BOI, OD, CIS, CVF)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--bp-mode", type=str, default=DEFAULT_BP_MODE)
    parser.add_argument("--distances", type=int, nargs="+",
                        default=DEFAULT_DISTANCES)
    parser.add_argument("--p-values", type=float, nargs="+",
                        default=DEFAULT_P_VALUES)
    return parser.parse_args()


def main() -> None:
    """Run full evaluation and print all reports."""
    args = _parse_args()

    print("v4.0.0 — BP Free-Energy Landscape Diagnostics")
    print(f"Seed: {args.seed}")
    print(f"Distances: {args.distances}")
    print(f"P values: {args.p_values}")
    print(f"Trials: {args.trials}")
    print(f"Max iters: {args.max_iters}")
    print(f"BP mode: {args.bp_mode}")
    if args.landscape:
        print("Landscape diagnostics: ENABLED")
    if args.iteration_diagnostics:
        print("Iteration-trace diagnostics: ENABLED")

    # Full evaluation.
    eval_result = run_evaluation(
        seed=args.seed,
        distances=args.distances,
        p_values=args.p_values,
        trials=args.trials,
        max_iters=args.max_iters,
        bp_mode=args.bp_mode,
        enable_energy_trace=args.landscape or args.iteration_diagnostics,
        enable_landscape=args.landscape,
        enable_iteration_diagnostics=args.iteration_diagnostics,
    )

    # Print reports.
    print_activation_report(eval_result)
    print_dps_table(eval_result)
    print_energy_trace(eval_result)

    if args.landscape:
        print_basin_statistics(eval_result)

    # Determinism check.
    det_result = run_determinism_check(
        seed=args.seed,
        max_iters=args.max_iters,
        bp_mode=args.bp_mode,
    )
    print_determinism_result(det_result)

    # Exit code.
    if not det_result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
