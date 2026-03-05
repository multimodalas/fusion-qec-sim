"""
Deterministic BP regime transition analysis (v4.5.0).

Constructs per-iteration regime traces using sliding-window classification,
detects regime transitions, measures dwell times, identifies instanton-like
events, and produces transition statistics.

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .bp_dynamics import (
    compute_bp_dynamics_metrics,
    classify_bp_regime,
)

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_REGIME_TRACE_PARAMS: Dict[str, Any] = {
    "event_factor": 4.0,
}


# ── Public API ───────────────────────────────────────────────────────


def compute_bp_regime_trace(
    llr_trace: list,
    energy_trace: list,
    correction_vectors: Optional[list] = None,
    *,
    window: int = 16,
    params: Optional[dict] = None,
    thresholds: Optional[dict] = None,
) -> dict:
    """Compute deterministic per-iteration BP regime trace and
    transition statistics.

    Trace-only. Decoder-safe.

    Parameters
    ----------
    llr_trace : list
        Per-iteration LLR vectors.
    energy_trace : list
        Per-iteration energy values.
    correction_vectors : list or None
        Per-iteration correction vectors (optional).
    window : int
        Sliding window size for per-iteration classification.
        Default 16.  Reduced deterministically if trace is shorter.
    params : dict or None
        Override default parameters for the underlying metric computation.
    thresholds : dict or None
        Override default thresholds for the regime classifier.

    Returns
    -------
    dict with keys ``regime_trace``, ``transitions``, ``dwell_times``,
    ``transition_counts``, ``summary``.
    All values are JSON-serializable (Python float/int/str/dict/list).
    """
    n_iters = len(llr_trace)
    n_energy = len(energy_trace)

    # Merge user params with defaults.
    p = dict(DEFAULT_REGIME_TRACE_PARAMS)
    if params is not None:
        p.update(params)
    event_factor = float(p["event_factor"])

    # ── Empty / trivial trace handling ───────────────────────────────
    if n_iters == 0 or n_energy == 0:
        return {
            "regime_trace": [],
            "transitions": [],
            "dwell_times": {},
            "transition_counts": {},
            "summary": {
                "switch_rate": 0.0,
                "max_dwell": 0,
                "freeze_score": 0.0,
                "num_events": 0,
            },
        }

    # ── 1) Sliding-window regime classification ─────────────────────
    regime_trace: List[str] = []

    for t in range(n_iters):
        # Determine effective window: shrink if trace is short.
        w = min(window, t + 1)

        # Extract sub-traces for window [t - w + 1 .. t] (inclusive).
        start = t - w + 1
        end = t + 1  # exclusive
        llr_window = llr_trace[start:end]
        energy_window = energy_trace[start:end]

        cv_window: Optional[list] = None
        if correction_vectors is not None:
            cv_window = correction_vectors[start:end]

        # Compute metrics over this window using v4.4 API.
        result = compute_bp_dynamics_metrics(
            llr_window,
            energy_window,
            correction_vectors=cv_window,
            params=params,
        )

        # Classify using v4.4 classifier.
        classification = classify_bp_regime(
            result["metrics"],
            thresholds=thresholds,
        )
        regime_trace.append(classification["regime"])

    # ── 2) Regime transition detection ──────────────────────────────
    e_arr = np.array(energy_trace, dtype=np.float64)
    abs_delta_e = np.abs(np.diff(e_arr)) if len(e_arr) >= 2 else np.array([])

    # Compute median |ΔE| for event detection threshold.
    if len(abs_delta_e) > 0:
        median_abs_de = float(np.median(abs_delta_e))
    else:
        median_abs_de = 0.0
    event_threshold = median_abs_de * event_factor

    transitions: List[Dict[str, Any]] = []
    for t in range(1, len(regime_trace)):
        if regime_trace[t] != regime_trace[t - 1]:
            # Determine if this is an instanton-like event.
            event = False
            if t < len(e_arr) and t - 1 < len(e_arr):
                delta_e = abs(float(e_arr[t]) - float(e_arr[t - 1]))
                if event_threshold > 0.0 and delta_e > event_threshold:
                    event = True

            transitions.append({
                "t": t,
                "from": regime_trace[t - 1],
                "to": regime_trace[t],
                "event": event,
            })

    # ── 3) Dwell time measurement ───────────────────────────────────
    dwell_times: Dict[str, List[int]] = {}
    if len(regime_trace) > 0:
        current_regime = regime_trace[0]
        current_run = 1
        for t in range(1, len(regime_trace)):
            if regime_trace[t] == current_regime:
                current_run += 1
            else:
                if current_regime not in dwell_times:
                    dwell_times[current_regime] = []
                dwell_times[current_regime].append(current_run)
                current_regime = regime_trace[t]
                current_run = 1
        # Final run.
        if current_regime not in dwell_times:
            dwell_times[current_regime] = []
        dwell_times[current_regime].append(current_run)

    # Sort keys lexicographically for deterministic output.
    dwell_times = {k: dwell_times[k] for k in sorted(dwell_times.keys())}

    # ── 4) Transition count matrix ──────────────────────────────────
    transition_counts: Dict[str, int] = {}
    for tr in transitions:
        key = f"{tr['from']}->{tr['to']}"
        transition_counts[key] = transition_counts.get(key, 0) + 1

    # Sort keys lexicographically.
    transition_counts = {
        k: transition_counts[k] for k in sorted(transition_counts.keys())
    }

    # ── 5) Summary statistics ───────────────────────────────────────
    total_iters = len(regime_trace)
    n_transitions = len(transitions)
    num_events = sum(1 for tr in transitions if tr["event"])

    switch_rate = float(n_transitions) / float(total_iters) if total_iters > 0 else 0.0

    all_dwells: List[int] = []
    for dlist in dwell_times.values():
        all_dwells.extend(dlist)
    max_dwell = max(all_dwells) if all_dwells else 0

    freeze_score = float(max_dwell) / float(total_iters) if total_iters > 0 else 0.0

    summary = {
        "switch_rate": float(switch_rate),
        "max_dwell": int(max_dwell),
        "freeze_score": float(freeze_score),
        "num_events": int(num_events),
    }

    return {
        "regime_trace": regime_trace,
        "transitions": transitions,
        "dwell_times": dwell_times,
        "transition_counts": transition_counts,
        "summary": summary,
    }
