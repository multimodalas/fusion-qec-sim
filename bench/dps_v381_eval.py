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
from src.qec.decoder.decoder_interface import get_decoder

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

from src.qec.diagnostics.bp_dynamics import (
    compute_bp_dynamics_metrics,
)

from src.qec.diagnostics.bp_regime_trace import (
    compute_bp_regime_trace,
)

from src.qec.diagnostics.bp_phase_diagram import (
    compute_bp_phase_diagram,
)

from src.qec.diagnostics.bp_freeze_detection import (
    compute_bp_freeze_detection,
)

from src.qec.diagnostics.bp_fixed_point_analysis import (
    compute_bp_fixed_point_analysis,
)

from src.qec.diagnostics.bp_basin_analysis import (
    compute_bp_basin_analysis,
)

from src.qec.diagnostics.bp_barrier_analysis import (
    compute_bp_barrier_analysis,
)

from src.qec.diagnostics.bp_boundary_analysis import (
    compute_bp_boundary_analysis,
)

from src.qec.diagnostics.spectral_boundary_alignment import (
    compute_spectral_boundary_alignment,
)
from src.qec.diagnostics.spectral_trapping_sets import (
    compute_spectral_trapping_sets,
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
    enable_bp_dynamics: bool = False,
    enable_bp_transitions: bool = False,
    enable_bp_phase_diagram: bool = False,
    enable_bp_freeze_detection: bool = False,
    decoder_fn=None,
    compare_decoders: bool = False,
    paired_seed: bool = False,
    paired_errors: bool = False,
    enable_bp_fixed_point_analysis: bool = False,
    enable_bp_basin_analysis: bool = False,
    enable_bp_landscape_map: bool = False,
    enable_bp_barrier_analysis: bool = False,
    enable_bp_boundary_analysis: bool = False,
) -> dict[str, Any]:
    """Run a single mode over pre-generated instances.

    Returns dict with keys: fer, frame_errors, trials, audit_summary.
    When *enable_energy_trace* or *enable_landscape* is True, also
    returns ``energy_traces``.  When *enable_landscape* is True, also
    returns ``landscape_metrics``, ``basin_switch``, and
    ``energy_delta``.  When *enable_iteration_diagnostics* is True,
    also returns ``iteration_diagnostics``.  When *enable_bp_dynamics*
    is True, also returns ``bp_dynamics``.  When
    *enable_bp_transitions* is True, also returns
    ``bp_regime_trace``, ``bp_transition_summary``, and
    ``bp_transition_counts``.  When *enable_bp_freeze_detection* is
    True, also returns ``bp_freeze_detection``.  When
    *enable_bp_fixed_point_analysis* is True, also returns
    ``bp_fixed_point_analysis`` and ``bp_fixed_point_summary``.
    When *enable_bp_basin_analysis* is True, also returns
    ``bp_basin_analysis`` and ``bp_basin_summary``.
    When *enable_bp_landscape_map* is True, also returns
    ``bp_landscape_map`` and ``bp_landscape_summary``.
    When *enable_bp_barrier_analysis* is True, also returns
    ``bp_barrier_analysis`` and ``bp_barrier_summary``.
    When *enable_bp_boundary_analysis* is True, also returns
    ``bp_boundary_analysis`` and ``bp_boundary_summary``.
    """
    if decoder_fn is None:
        decoder_fn = bp_decode

    mode_cfg = MODES[mode_name]
    schedule = mode_cfg["schedule"]
    structural: StructuralConfig = mode_cfg["structural"]

    # Landscape mode implies energy trace.
    if enable_landscape:
        enable_energy_trace = True
    # Phase diagram implies BP transitions.
    if enable_bp_phase_diagram:
        enable_bp_transitions = True
    # Barrier analysis implies landscape map.
    if enable_bp_barrier_analysis:
        enable_bp_landscape_map = True
    # Landscape map implies basin analysis.
    if enable_bp_landscape_map:
        enable_bp_basin_analysis = True
    # Basin analysis implies fixed-point analysis.
    if enable_bp_basin_analysis:
        enable_bp_fixed_point_analysis = True
    # Fixed-point analysis implies energy trace and iteration diagnostics.
    if enable_bp_fixed_point_analysis:
        enable_energy_trace = True
        enable_iteration_diagnostics = True
    # Freeze detection implies BP dynamics.
    if enable_bp_freeze_detection:
        enable_bp_dynamics = True
    # Iteration diagnostics implies energy trace.
    if enable_iteration_diagnostics:
        enable_energy_trace = True
    # BP transitions implies BP dynamics.
    if enable_bp_transitions:
        enable_bp_dynamics = True
    # BP dynamics implies energy trace and LLR history.
    if enable_bp_dynamics:
        enable_energy_trace = True
        enable_iteration_diagnostics = True

    frame_errors = 0
    residual_mismatches = 0
    trials_audit: list[dict[str, Any]] = []
    all_energy_traces: list[list[float]] = []
    all_landscape_metrics: list[dict[str, Any]] = []
    basin_switches = 0
    energy_deltas: list[float] = []
    all_basin_classifications: list[dict[str, Any]] = []
    all_iteration_diagnostics: list[dict[str, Any]] = []
    all_bp_dynamics: list[dict[str, Any]] = []
    all_bp_regime_traces: list[dict[str, Any]] = []
    all_bp_freeze_detections: list[dict[str, Any]] = []
    all_bp_fixed_point_analyses: list[dict[str, Any]] = []
    all_bp_basin_analyses: list[dict[str, Any]] = []
    all_bp_landscape_maps: list[dict[str, Any]] = []
    all_bp_barrier_analyses: list[dict[str, Any]] = []
    all_bp_boundary_analyses: list[dict[str, Any]] = []
    comparison_results: list[dict[str, Any]] = []

    for trial_idx, inst in enumerate(instances):
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
        result = decoder_fn(
            H_used, llr_used,
            max_iters=max_iters,
            mode=bp_mode,
            schedule=schedule,
            syndrome_vec=s_used,
            energy_trace=use_energy,
            llr_history=use_llr_history,
        )
        correction, iters = result[0], result[1]

        # ── Decoder comparison mode ──
        if compare_decoders:
            ref_fn = get_decoder("reference")
            exp_fn = get_decoder("experimental")
            # When paired_errors is active, explicitly copy inputs to
            # guarantee both decoders observe identical data.
            ref_llr = np.copy(llr_used) if paired_errors else llr_used
            exp_llr = np.copy(llr_used) if paired_errors else llr_used
            ref_result = ref_fn(
                H_used, ref_llr,
                max_iters=max_iters,
                mode=bp_mode,
                schedule=schedule,
                syndrome_vec=s_used,
            )
            exp_result = exp_fn(
                H_used, exp_llr,
                max_iters=max_iters,
                mode=bp_mode,
                schedule=schedule,
                syndrome_vec=s_used,
            )
            ref_corr, ref_iters = ref_result[0], ref_result[1]
            exp_corr, exp_iters = exp_result[0], exp_result[1]
            s_ref = syndrome(H, ref_corr)
            s_exp = syndrome(H, exp_corr)
            comparison_results.append({
                "reference": {
                    "success": bool(np.array_equal(s_ref, s)),
                    "iterations": int(ref_iters),
                    "syndrome_weight": int(np.sum(s_ref != s)),
                },
                "experimental": {
                    "success": bool(np.array_equal(s_exp, s)),
                    "iterations": int(exp_iters),
                    "syndrome_weight": int(np.sum(s_exp != s)),
                },
            })
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
                correction_vectors=None,
            )
            all_iteration_diagnostics.append(iter_diag)

            # v4.4.0: BP dynamics regime analysis.
            if enable_bp_dynamics:
                bp_dyn = compute_bp_dynamics_metrics(
                    llr_trace=llr_trace_list,
                    energy_trace=list(trace),
                    correction_vectors=None,
                )
                all_bp_dynamics.append(bp_dyn)

                # v4.5.0: BP regime transition analysis.
                if enable_bp_transitions:
                    rt = compute_bp_regime_trace(
                        llr_trace=llr_trace_list,
                        energy_trace=list(trace),
                        correction_vectors=None,
                    )
                    all_bp_regime_traces.append(rt)

                # v4.7.0: BP freeze detection.
                if enable_bp_freeze_detection:
                    fd = compute_bp_freeze_detection(
                        llr_trace=llr_trace_list,
                        energy_trace=list(trace),
                    )
                    all_bp_freeze_detections.append(fd)

        # v4.8.0: BP fixed-point trap analysis.
        if enable_bp_fixed_point_analysis and trace is not None:
            llr_trace_for_fp = (
                [llr_hist[i] for i in range(llr_hist.shape[0])]
                if llr_hist is not None and llr_hist.shape[0] > 0
                else []
            )
            energy_list = list(trace)
            # Build syndrome trace from per-iteration correction syndromes.
            # Use final syndrome weight from the correction.
            s_correction = syndrome(H, correction)
            final_sw = int(np.sum(s_correction != s))
            # Build per-iteration syndrome weights from energy trace length.
            syndrome_trace_fp = [final_sw] * len(energy_list)
            fp_result = compute_bp_fixed_point_analysis(
                llr_trace=llr_trace_for_fp if llr_trace_for_fp else [np.zeros(H.shape[1])],
                energy_trace=energy_list if energy_list else [0.0],
                syndrome_trace=syndrome_trace_fp if syndrome_trace_fp else [final_sw],
                final_syndrome_weight=final_sw,
            )
            all_bp_fixed_point_analyses.append(fp_result)

            # v4.9.0: BP basin-of-attraction analysis.
            if enable_bp_basin_analysis:
                basin_result = compute_bp_basin_analysis(
                    H=H,
                    llr=llr_used,
                    baseline_fixed_point_type=fp_result["fixed_point_type"],
                    max_iters=max_iters,
                    bp_mode=bp_mode,
                    schedule=schedule,
                    syndrome_vec=s_used,
                    syndrome_original=s,
                )
                all_bp_basin_analyses.append(basin_result)

            # v5.0.0: BP attractor landscape mapping.
            if enable_bp_landscape_map:
                from src.qec.diagnostics.bp_landscape_mapping import (
                    compute_bp_landscape_map,
                )
                landscape_map_result = compute_bp_landscape_map(
                    H=H,
                    llr=llr_used,
                    max_iters=max_iters,
                    bp_mode=bp_mode,
                    schedule=schedule,
                    syndrome_vec=s_used,
                    syndrome_original=s,
                )
                all_bp_landscape_maps.append(landscape_map_result)

            # v5.1.0: BP free-energy barrier estimation.
            if enable_bp_barrier_analysis:
                def _barrier_decode_fn(llr_vec):
                    """Decode wrapper for barrier analysis."""
                    _result = bp_decode(
                        H, llr_vec,
                        max_iters=max_iters,
                        mode=bp_mode,
                        schedule=schedule,
                        syndrome_vec=s_used,
                        energy_trace=True,
                        llr_history=max_iters,
                    )
                    _correction, _iters = _result[0], _result[1]
                    _llr_hist = _result[2]
                    _trace = _result[-1]

                    _llr_trace_list = (
                        [_llr_hist[i] for i in range(_llr_hist.shape[0])]
                        if _llr_hist is not None and _llr_hist.shape[0] > 0
                        else []
                    )
                    _energy_list = list(_trace) if _trace is not None else []

                    _s_correction = syndrome(H, _correction)
                    _final_sw = int(np.sum(_s_correction != s))

                    _syndrome_trace = [_final_sw] * len(_energy_list)

                    return {
                        "llr_trace": _llr_trace_list if _llr_trace_list else [np.zeros(H.shape[1])],
                        "energy_trace": _energy_list if _energy_list else [0.0],
                        "syndrome_trace": _syndrome_trace if _syndrome_trace else [_final_sw],
                        "final_syndrome_weight": _final_sw,
                    }

                barrier_result = compute_bp_barrier_analysis(
                    decode_fn=_barrier_decode_fn,
                    llr_init=llr_used,
                )
                all_bp_barrier_analyses.append(barrier_result)

            # v5.3.0: BP boundary analysis (attractor basin distance).
            if enable_bp_boundary_analysis:
                def _boundary_decode_fn(llr_vec):
                    """Decode wrapper for boundary analysis."""
                    _result = bp_decode(
                        H, llr_vec,
                        max_iters=max_iters,
                        mode=bp_mode,
                        schedule=schedule,
                        syndrome_vec=s_used,
                    )
                    return _result[0]

                boundary_result = compute_bp_boundary_analysis(
                    llr_vector=llr_used,
                    decoder_fn=_boundary_decode_fn,
                    parity_check_matrix=H,
                )
                all_bp_boundary_analyses.append(boundary_result)

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
    if all_bp_dynamics:
        out["bp_dynamics"] = all_bp_dynamics
        # Aggregate regime counts.
        regime_counts: dict[str, int] = {}
        for bd in all_bp_dynamics:
            r = bd["regime"]
            regime_counts[r] = regime_counts.get(r, 0) + 1
        out["bp_regime_counts"] = regime_counts
    if all_bp_regime_traces:
        out["bp_regime_trace"] = all_bp_regime_traces
        # Aggregate transition summary and counts across all trials.
        agg_counts: dict[str, int] = {}
        total_events = 0
        total_transitions = 0
        total_iters = 0
        max_dwell_all = 0
        for rt in all_bp_regime_traces:
            for k, v in rt["transition_counts"].items():
                agg_counts[k] = agg_counts.get(k, 0) + v
            total_events += rt["summary"]["num_events"]
            total_transitions += len(rt["transitions"])
            total_iters += len(rt["regime_trace"])
            if rt["summary"]["max_dwell"] > max_dwell_all:
                max_dwell_all = rt["summary"]["max_dwell"]
        out["bp_transition_counts"] = {
            k: agg_counts[k] for k in sorted(agg_counts.keys())
        }
        out["bp_transition_summary"] = {
            "total_transitions": total_transitions,
            "total_events": total_events,
            "total_iters": total_iters,
            "max_dwell": max_dwell_all,
            "aggregate_switch_rate": (
                float(total_transitions) / float(total_iters)
                if total_iters > 0 else 0.0
            ),
        }
    if all_bp_freeze_detections:
        out["bp_freeze_detection"] = all_bp_freeze_detections
        # Aggregate: count how many trials had freeze detected.
        freeze_count = sum(
            1 for fd in all_bp_freeze_detections if fd["freeze_detected"]
        )
        out["bp_freeze_summary"] = {
            "freeze_count": freeze_count,
            "freeze_fraction": (
                float(freeze_count) / float(len(all_bp_freeze_detections))
                if all_bp_freeze_detections else 0.0
            ),
            "max_freeze_score": max(
                fd["freeze_score"] for fd in all_bp_freeze_detections
            ),
        }
    if all_bp_fixed_point_analyses:
        out["bp_fixed_point_analysis"] = all_bp_fixed_point_analyses
        # Aggregate fixed-point type counts.
        fp_type_counts: dict[str, int] = {}
        total_iters_to_fp: list[int] = []
        for fpa in all_bp_fixed_point_analyses:
            fpt = fpa["fixed_point_type"]
            fp_type_counts[fpt] = fp_type_counts.get(fpt, 0) + 1
            total_iters_to_fp.append(fpa["iterations_to_fixed_point"])
        n_fp = len(all_bp_fixed_point_analyses)
        out["bp_fixed_point_summary"] = {
            "correct_fixed_point_probability": float(
                fp_type_counts.get("correct_fixed_point", 0)
            ) / float(n_fp),
            "degenerate_fixed_point_probability": float(
                fp_type_counts.get("degenerate_fixed_point", 0)
            ) / float(n_fp),
            "incorrect_fixed_point_probability": float(
                fp_type_counts.get("incorrect_fixed_point", 0)
            ) / float(n_fp),
            "mean_iterations_to_fixed_point": (
                float(sum(total_iters_to_fp)) / float(n_fp)
            ),
            "no_convergence_probability": float(
                fp_type_counts.get("no_convergence", 0)
            ) / float(n_fp),
            "type_counts": {
                k: fp_type_counts[k]
                for k in sorted(fp_type_counts.keys())
            },
        }
    if all_bp_basin_analyses:
        out["bp_basin_analysis"] = all_bp_basin_analyses
        # Aggregate basin statistics.
        n_ba = len(all_bp_basin_analyses)
        total_correct = sum(ba["num_correct"] for ba in all_bp_basin_analyses)
        total_incorrect = sum(ba["num_incorrect"] for ba in all_bp_basin_analyses)
        total_degenerate = sum(ba["num_degenerate"] for ba in all_bp_basin_analyses)
        total_perturbations = sum(ba["num_perturbations"] for ba in all_bp_basin_analyses)
        boundary_values = [
            ba["basin_boundary_eps"]
            for ba in all_bp_basin_analyses
            if ba["basin_boundary_eps"] is not None
        ]
        out["bp_basin_summary"] = {
            "mean_basin_correct_probability": (
                float(total_correct) / float(total_perturbations)
                if total_perturbations > 0 else 0.0
            ),
            "mean_basin_incorrect_probability": (
                float(total_incorrect) / float(total_perturbations)
                if total_perturbations > 0 else 0.0
            ),
            "mean_basin_degenerate_probability": (
                float(total_degenerate) / float(total_perturbations)
                if total_perturbations > 0 else 0.0
            ),
            "num_trials_with_boundary": len(boundary_values),
            "mean_boundary_eps": (
                float(sum(boundary_values)) / float(len(boundary_values))
                if boundary_values else None
            ),
            "min_boundary_eps": (
                float(min(boundary_values)) if boundary_values else None
            ),
            "total_perturbations": total_perturbations,
            "total_trials": n_ba,
        }
    if all_bp_landscape_maps:
        out["bp_landscape_map"] = all_bp_landscape_maps
        # Aggregate landscape statistics.
        n_lm = len(all_bp_landscape_maps)
        total_attractors = sum(
            lm["num_attractors"] for lm in all_bp_landscape_maps
        )
        total_pseudocodewords = sum(
            lm["num_pseudocodewords"] for lm in all_bp_landscape_maps
        )
        basin_fractions = [
            lm["largest_basin_fraction"] for lm in all_bp_landscape_maps
        ]
        out["bp_landscape_summary"] = {
            "mean_largest_basin_fraction": (
                float(sum(basin_fractions)) / float(n_lm)
            ),
            "mean_num_attractors": float(total_attractors) / float(n_lm),
            "total_pseudocodewords": total_pseudocodewords,
            "total_trials": n_lm,
        }
    if all_bp_barrier_analyses:
        out["bp_barrier_analysis"] = all_bp_barrier_analyses
        # Aggregate barrier statistics.
        n_bar = len(all_bp_barrier_analyses)
        escaped_count = sum(
            1 for ba in all_bp_barrier_analyses if ba["escaped"]
        )
        barrier_eps_values = [
            ba["barrier_eps"] for ba in all_bp_barrier_analyses
            if ba["barrier_eps"] is not None
        ]
        total_trials_sum = sum(
            ba["num_trials"] for ba in all_bp_barrier_analyses
        )
        out["bp_barrier_summary"] = {
            "escape_probability": float(escaped_count) / float(n_bar),
            "mean_barrier_eps": (
                float(sum(barrier_eps_values)) / float(len(barrier_eps_values))
                if barrier_eps_values else None
            ),
            "num_trials": total_trials_sum,
        }
    if all_bp_boundary_analyses:
        out["bp_boundary_analysis"] = all_bp_boundary_analyses
        # Aggregate boundary statistics.
        n_bnd = len(all_bp_boundary_analyses)
        crossed_count = sum(
            1 for ba in all_bp_boundary_analyses if ba["boundary_crossed"]
        )
        boundary_eps_values = [
            ba["boundary_eps"] for ba in all_bp_boundary_analyses
            if ba["boundary_eps"] is not None
        ]
        total_boundary_directions = sum(
            ba["num_directions"] for ba in all_bp_boundary_analyses
        )
        out["bp_boundary_summary"] = {
            "boundary_cross_probability": float(crossed_count) / float(n_bnd),
            "mean_boundary_eps": (
                float(sum(boundary_eps_values)) / float(len(boundary_eps_values))
                if boundary_eps_values else None
            ),
            "num_directions": total_boundary_directions,
        }
    if comparison_results:
        out["decoder_comparison"] = comparison_results
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
    enable_bp_dynamics: bool = False,
    enable_bp_transitions: bool = False,
    enable_bp_phase_diagram: bool = False,
    enable_bp_freeze_detection: bool = False,
    enable_bp_fixed_point_analysis: bool = False,
    enable_bp_basin_analysis: bool = False,
    enable_bp_landscape_map: bool = False,
    enable_bp_barrier_analysis: bool = False,
    enable_bp_boundary_analysis: bool = False,
    enable_tanner_spectral_analysis: bool = False,
    enable_spectral_boundary_alignment: bool = False,
    enable_spectral_trapping_sets: bool = False,
    decoder_fn=None,
    compare_decoders: bool = False,
    paired_seed: bool = False,
    paired_errors: bool = False,
) -> dict[str, Any]:
    """Run the full DPS evaluation across all modes, distances, and p values.

    Returns a dict with:
      results[mode_name][p][distance] = run_mode result
      slopes[mode_name][p] = DPS slope
      audit[mode_name][p][distance] = audit_summary
    """
    # v5.5.0: spectral-boundary alignment implies dependencies.
    if enable_spectral_boundary_alignment:
        enable_tanner_spectral_analysis = True
        enable_bp_boundary_analysis = True
        enable_bp_barrier_analysis = True

    # v5.6.0: spectral trapping sets implies tanner spectral analysis.
    if enable_spectral_trapping_sets:
        enable_tanner_spectral_analysis = True

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

    # v5.4.0: Tanner spectral analysis (once per code instance).
    tanner_spectral_results: dict[int, dict[str, Any]] = {}
    if enable_tanner_spectral_analysis:
        from src.qec.diagnostics.tanner_spectral_analysis import (
            compute_tanner_spectral_analysis,
        )
        for distance in distances:
            tanner_spectral_results[distance] = compute_tanner_spectral_analysis(
                codes[distance],
            )

    # v5.5.0: Extract spectral eigenvectors for alignment analysis.
    tanner_spectral_modes: dict[int, list[np.ndarray]] = {}
    if enable_spectral_boundary_alignment:
        for distance in distances:
            H = codes[distance]
            m, n = H.shape
            top = np.concatenate([np.zeros((n, n), dtype=np.float64), H.T.astype(np.float64)], axis=1)
            bottom = np.concatenate([H.astype(np.float64), np.zeros((m, m), dtype=np.float64)], axis=1)
            A = np.concatenate([top, bottom], axis=0)
            eigvals, eigvecs = np.linalg.eigh(A)
            sort_idx = np.argsort(-eigvals)
            eigvecs = eigvecs[:, sort_idx]
            top_k = min(3, eigvecs.shape[1])
            tanner_spectral_modes[distance] = [
                eigvecs[:n, i].copy() for i in range(top_k)
            ]

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
                    enable_bp_dynamics=enable_bp_dynamics,
                    enable_bp_transitions=enable_bp_transitions,
                    enable_bp_phase_diagram=enable_bp_phase_diagram,
                    enable_bp_freeze_detection=enable_bp_freeze_detection,
                    enable_bp_fixed_point_analysis=enable_bp_fixed_point_analysis,
                    enable_bp_basin_analysis=enable_bp_basin_analysis,
                    enable_bp_landscape_map=enable_bp_landscape_map,
                    enable_bp_barrier_analysis=enable_bp_barrier_analysis,
                    enable_bp_boundary_analysis=enable_bp_boundary_analysis,
                    decoder_fn=decoder_fn,
                    compare_decoders=compare_decoders,
                    paired_seed=paired_seed,
                    paired_errors=paired_errors,
                )
                results[mode_name][p][distance] = result
                audits[mode_name][p][distance] = result["audit_summary"]
                fer_by_distance[distance] = result["fer"]

            slopes[mode_name][p] = compute_dps_slope(fer_by_distance)

    out: dict[str, Any] = {
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

    # v5.4.0: Tanner spectral analysis results.
    if tanner_spectral_results:
        out["tanner_spectral_analysis"] = {
            int(d): tanner_spectral_results[d] for d in sorted(tanner_spectral_results.keys())
        }

    # v5.6.0: Spectral trapping-set analysis (once per code instance).
    if enable_spectral_trapping_sets and tanner_spectral_results:
        trapping_set_results: dict[int, dict[str, Any]] = {}
        for distance in sorted(tanner_spectral_results.keys()):
            H = codes[distance]
            m, n = H.shape
            # Extract eigenvectors from the Tanner adjacency matrix.
            top_block = np.concatenate(
                [np.zeros((n, n), dtype=np.float64), H.T.astype(np.float64)],
                axis=1,
            )
            bottom_block = np.concatenate(
                [H.astype(np.float64), np.zeros((m, m), dtype=np.float64)],
                axis=1,
            )
            A = np.concatenate([top_block, bottom_block], axis=0)
            eigvals, eigvecs = np.linalg.eigh(A)
            sort_idx = np.argsort(-eigvals)
            eigvecs = eigvecs[:, sort_idx]
            top_k = min(3, eigvecs.shape[1])
            modes_for_trapping = [eigvecs[:, i].copy() for i in range(top_k)]
            ts = compute_spectral_trapping_sets(modes_for_trapping, n)
            trapping_set_results[distance] = ts
        out["spectral_trapping_sets"] = {
            int(d): trapping_set_results[d]
            for d in sorted(trapping_set_results.keys())
        }

    # v5.5.0: Spectral–boundary alignment analysis.
    if enable_spectral_boundary_alignment:
        alignment_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        for mode_name in MODE_ORDER:
            alignment_results[mode_name] = {}
            for p in p_values:
                alignment_results[mode_name][p] = {}
                for distance in distances:
                    r = results[mode_name][p][distance]
                    spectral_modes = tanner_spectral_modes.get(distance, [])
                    # Extract boundary direction from boundary analysis.
                    boundary_dir = None
                    if "bp_boundary_analysis" in r:
                        for ba in r["bp_boundary_analysis"]:
                            if ba.get("boundary_direction") is not None:
                                boundary_dir = np.asarray(
                                    ba["boundary_direction"], dtype=np.float64
                                )
                                break
                    if boundary_dir is not None and spectral_modes:
                        sba = compute_spectral_boundary_alignment(
                            spectral_modes, boundary_dir,
                        )
                        # Extract barrier_eps and boundary_eps from summaries.
                        barrier_eps = None
                        boundary_eps = None
                        if "bp_barrier_summary" in r:
                            barrier_eps = r["bp_barrier_summary"].get("mean_barrier_eps")
                        if "bp_boundary_summary" in r:
                            boundary_eps = r["bp_boundary_summary"].get("mean_boundary_eps")
                        alignment_results[mode_name][p][distance] = {
                            "alignment_max": sba["max_alignment"],
                            "alignment_mean": sba["mean_alignment"],
                            "dominant_alignment_mode": sba["dominant_alignment_mode"],
                            "mode_count": sba["mode_count"],
                            "p": p,
                            "FER": r["fer"],
                            "boundary_eps": boundary_eps,
                            "barrier_eps": barrier_eps,
                        }
        out["spectral_boundary_alignment"] = alignment_results

    # v4.6.0: BP phase diagram aggregation.
    if enable_bp_phase_diagram:
        phase_run_results: list[dict[str, Any]] = []
        for mode_name in MODE_ORDER:
            for p in p_values:
                for distance in distances:
                    r = results[mode_name][p][distance]
                    if "bp_regime_trace" in r:
                        phase_run_results.append({
                            "distance": distance,
                            "noise": p,
                            "regime_trace_results": r["bp_regime_trace"],
                        })
        out["bp_phase_diagram"] = compute_bp_phase_diagram(phase_run_results)

    return out


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


def print_decoder_report(eval_result: dict[str, Any]) -> None:
    """Print a console FER comparison table from decoder comparison data."""
    results = eval_result["results"]
    config = eval_result["config"]
    distances = config["distances"]
    p_values = config["p_values"]

    print("\nDecoder Comparison Report")
    print("-----------------------------------------------------------")
    print(f"{'mode':<24} {'distance':>8}  {'p':>8}  {'FER_ref':>8}  {'FER_exp':>8}  {'ΔFER':>8}")
    print("-----------------------------------------------------------")

    for mode_name, mode_results in results.items():
        for p in p_values:
            if p not in mode_results:
                continue
            for d in distances:
                r = mode_results[p].get(d)
                if r is None or "decoder_comparison" not in r:
                    continue
                comps = r["decoder_comparison"]
                n = len(comps)
                if n == 0:
                    continue
                fer_ref = sum(1 for c in comps if not c["reference"]["success"]) / n
                fer_exp = sum(1 for c in comps if not c["experimental"]["success"]) / n
                delta = fer_exp - fer_ref
                print(f"{mode_name:<24} {d:>8}  {p:>8.4f}  {fer_ref:>8.4f}  {fer_exp:>8.4f}  {delta:>+8.4f}")
    print()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="v4.0.0 — BP Free-Energy Landscape Diagnostics",
    )
    parser.add_argument("--landscape", action="store_true",
                        help="Enable landscape diagnostics and basin switching")
    parser.add_argument("--iteration-diagnostics", action="store_true",
                        help="Enable iteration-trace diagnostics (PEI, BOI, OD, CIS, CVF)")
    parser.add_argument("--bp-dynamics", action="store_true",
                        help="Enable BP dynamics regime analysis (MSI, CPI, TSL, LEC, CVNE, GOS, EDS, BTI)")
    parser.add_argument("--bp-transitions", action="store_true",
                        help="Enable BP regime transition analysis (regime trace, dwell times, events)")
    parser.add_argument("--bp-phase-diagram", action="store_true",
                        help="Enable BP phase diagram aggregation (implicitly enables --bp-transitions)")
    parser.add_argument("--bp-freeze-detection", action="store_true",
                        help="Enable BP freeze detection (early metastability detection)")
    parser.add_argument("--bp-fixed-point-analysis", action="store_true",
                        help="Enable BP fixed-point trap analysis (correct/incorrect/degenerate classification)")
    parser.add_argument("--bp-basin-analysis", action="store_true",
                        help="Enable BP basin-of-attraction analysis (perturbation-based basin geometry)")
    parser.add_argument("--bp-landscape-map", action="store_true",
                        help="Enable BP attractor landscape mapping (attractor enumeration, pseudocodeword detection)")
    parser.add_argument("--bp-barrier-analysis", action="store_true",
                        help="Enable BP free-energy barrier estimation (escape perturbation measurement)")
    parser.add_argument("--bp-boundary-analysis", action="store_true",
                        help="Enable BP boundary analysis (attractor basin distance estimation)")
    parser.add_argument("--tanner-spectral-analysis", action="store_true",
                        help="Enable Tanner spectral fragility diagnostics (spectral gap, eigenmode localization)")
    parser.add_argument("--spectral-boundary-alignment", action="store_true",
                        help="Enable spectral-boundary alignment diagnostics (v5.5)")
    parser.add_argument("--spectral-trapping-sets", action="store_true",
                        help="Enable spectral trapping-set diagnostics (v5.6)")
    parser.add_argument("--decoder", type=str, default="reference",
                        choices=["reference", "experimental"],
                        help="Decoder implementation to use (default: reference)")
    parser.add_argument("--compare-decoders", action="store_true",
                        help="Run both reference and experimental decoders on each instance")
    parser.add_argument("--paired-seed", action="store_true",
                        help="Ensure both decoders share the same deterministic seed per trial")
    parser.add_argument("--paired-errors", action="store_true",
                        help="Ensure both decoders receive identical error realizations")
    parser.add_argument("--decoder-report", action="store_true",
                        help="Print FER comparison table after evaluation")
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
    if args.bp_dynamics:
        print("BP dynamics regime analysis: ENABLED")
    if args.bp_transitions:
        print("BP regime transition analysis: ENABLED")
    if args.bp_phase_diagram:
        print("BP phase diagram analysis: ENABLED")
    if args.bp_freeze_detection:
        print("BP freeze detection: ENABLED")
    if args.bp_fixed_point_analysis:
        print("BP fixed-point trap analysis: ENABLED")
    if args.bp_basin_analysis:
        print("BP basin-of-attraction analysis: ENABLED")
    if args.bp_landscape_map:
        print("BP attractor landscape mapping: ENABLED")
    if args.bp_barrier_analysis:
        print("BP free-energy barrier estimation: ENABLED")
    if args.bp_boundary_analysis:
        print("BP boundary analysis: ENABLED")
    if args.tanner_spectral_analysis:
        print("Tanner spectral fragility diagnostics: ENABLED")
    if args.spectral_boundary_alignment:
        print("Spectral-boundary alignment diagnostics: ENABLED")
    if args.spectral_trapping_sets:
        print("Spectral trapping-set diagnostics: ENABLED")
    print(f"Decoder: {args.decoder}")
    if args.compare_decoders:
        print("Decoder comparison mode: ENABLED")
    if args.paired_seed:
        print("Paired seed: ENABLED")
    if args.paired_errors:
        print("Paired errors: ENABLED")
    if args.decoder_report:
        print("Decoder report: ENABLED")

    # Resolve decoder callable.
    selected_decoder = get_decoder(args.decoder)

    # Full evaluation.
    eval_result = run_evaluation(
        seed=args.seed,
        distances=args.distances,
        p_values=args.p_values,
        trials=args.trials,
        max_iters=args.max_iters,
        bp_mode=args.bp_mode,
        enable_energy_trace=args.landscape or args.iteration_diagnostics or args.bp_dynamics or args.bp_transitions or args.bp_phase_diagram or args.bp_freeze_detection or args.bp_fixed_point_analysis or args.bp_basin_analysis or args.bp_landscape_map or args.bp_barrier_analysis,
        enable_landscape=args.landscape,
        enable_iteration_diagnostics=args.iteration_diagnostics or args.bp_dynamics or args.bp_transitions or args.bp_phase_diagram or args.bp_freeze_detection or args.bp_fixed_point_analysis or args.bp_basin_analysis or args.bp_landscape_map or args.bp_barrier_analysis,
        enable_bp_dynamics=args.bp_dynamics or args.bp_transitions or args.bp_phase_diagram or args.bp_freeze_detection,
        enable_bp_transitions=args.bp_transitions or args.bp_phase_diagram,
        enable_bp_phase_diagram=args.bp_phase_diagram,
        enable_bp_freeze_detection=args.bp_freeze_detection,
        enable_bp_fixed_point_analysis=args.bp_fixed_point_analysis or args.bp_basin_analysis or args.bp_landscape_map or args.bp_barrier_analysis,
        enable_bp_basin_analysis=args.bp_basin_analysis or args.bp_landscape_map or args.bp_barrier_analysis,
        enable_bp_landscape_map=args.bp_landscape_map or args.bp_barrier_analysis,
        enable_bp_barrier_analysis=args.bp_barrier_analysis or args.spectral_boundary_alignment,
        enable_bp_boundary_analysis=args.bp_boundary_analysis or args.spectral_boundary_alignment,
        enable_tanner_spectral_analysis=args.tanner_spectral_analysis or args.spectral_boundary_alignment or args.spectral_trapping_sets,
        enable_spectral_boundary_alignment=args.spectral_boundary_alignment,
        enable_spectral_trapping_sets=args.spectral_trapping_sets,
        decoder_fn=selected_decoder,
        compare_decoders=args.compare_decoders,
        paired_seed=args.paired_seed,
        paired_errors=args.paired_errors,
    )

    # Print reports.
    print_activation_report(eval_result)
    print_dps_table(eval_result)
    print_energy_trace(eval_result)

    if args.landscape:
        print_basin_statistics(eval_result)

    if args.decoder_report and args.compare_decoders:
        print_decoder_report(eval_result)

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
