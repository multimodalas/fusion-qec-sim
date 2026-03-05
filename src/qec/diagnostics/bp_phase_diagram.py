"""
Deterministic BP phase diagram analysis (v4.6.0).

Aggregates regime-trace diagnostics across decoding runs to compute
deterministic BP phase statistics as a function of code distance and
noise rate.

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).
"""

from __future__ import annotations

from typing import Any, Dict, List


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_METASTABLE_THRESHOLD = 0.5


# ── Public API ───────────────────────────────────────────────────────


def compute_bp_phase_diagram(
    run_results: list[dict],
    *,
    metastable_threshold: float = DEFAULT_METASTABLE_THRESHOLD,
) -> dict:
    """Aggregate regime-trace diagnostics across decoding runs to compute
    deterministic BP phase statistics.

    Parameters
    ----------
    run_results : list of dict
        Each entry must contain:
        - ``"distance"`` : int
        - ``"noise"`` : float
        - ``"regime_trace_results"`` : list of per-trial regime trace dicts,
          each with ``"regime_trace"`` (list[str]) and ``"summary"`` (dict
          with ``"freeze_score"``, ``"switch_rate"``, ``"max_dwell"``,
          ``"num_events"``).
    metastable_threshold : float
        Freeze-score threshold above which a run is considered metastable.
        Default 0.5.

    Returns
    -------
    dict
        JSON-serializable phase diagram dataset with keys:
        ``phase_points``, ``distance_levels``, ``noise_levels``,
        ``phase_statistics``.
    """
    if not run_results:
        return {
            "phase_points": [],
            "distance_levels": [],
            "noise_levels": [],
            "phase_statistics": {
                "total_runs": 0,
                "total_parameter_points": 0,
            },
        }

    # ── Group runs by (distance, noise) ───────────────────────────────
    groups: Dict[tuple, List[dict]] = {}
    for entry in run_results:
        key = (int(entry["distance"]), float(entry["noise"]))
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    # ── Compute phase points ──────────────────────────────────────────
    # Sort keys for deterministic output order.
    sorted_keys = sorted(groups.keys())

    distance_set: set = set()
    noise_set: set = set()
    phase_points: List[Dict[str, Any]] = []
    total_runs = 0

    for distance, noise in sorted_keys:
        distance_set.add(distance)
        noise_set.add(noise)
        entries = groups[(distance, noise)]

        # Collect all per-trial regime trace results for this point.
        all_traces: List[dict] = []
        for entry in entries:
            all_traces.extend(entry["regime_trace_results"])

        num_runs = len(all_traces)
        total_runs += num_runs

        if num_runs == 0:
            phase_points.append({
                "distance": distance,
                "noise": float(noise),
                "num_runs": 0,
                "metastable_probability": 0.0,
                "mean_freeze_score": 0.0,
                "mean_switch_rate": 0.0,
                "mean_max_dwell": 0.0,
                "event_rate": 0.0,
                "regime_frequencies": {},
            })
            continue

        # ── Metastable probability ────────────────────────────────────
        metastable_count = sum(
            1 for rt in all_traces
            if rt["summary"]["freeze_score"] > metastable_threshold
        )
        metastable_probability = float(metastable_count) / float(num_runs)

        # ── Mean freeze score ─────────────────────────────────────────
        freeze_scores = [
            float(rt["summary"]["freeze_score"]) for rt in all_traces
        ]
        mean_freeze_score = sum(freeze_scores) / float(num_runs)

        # ── Mean switch rate ──────────────────────────────────────────
        switch_rates = [
            float(rt["summary"]["switch_rate"]) for rt in all_traces
        ]
        mean_switch_rate = sum(switch_rates) / float(num_runs)

        # ── Mean max dwell ────────────────────────────────────────────
        max_dwells = [
            float(rt["summary"]["max_dwell"]) for rt in all_traces
        ]
        mean_max_dwell = sum(max_dwells) / float(num_runs)

        # ── Event rate ────────────────────────────────────────────────
        num_events_list = [
            float(rt["summary"]["num_events"]) for rt in all_traces
        ]
        event_rate = sum(num_events_list) / float(num_runs)

        # ── Regime frequencies ────────────────────────────────────────
        regime_counts: Dict[str, int] = {}
        total_regime_labels = 0
        for rt in all_traces:
            for regime in rt["regime_trace"]:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                total_regime_labels += 1

        # Normalize and sort lexicographically.
        regime_frequencies: Dict[str, float] = {}
        if total_regime_labels > 0:
            for k in sorted(regime_counts.keys()):
                regime_frequencies[k] = (
                    float(regime_counts[k]) / float(total_regime_labels)
                )

        phase_points.append({
            "distance": distance,
            "noise": float(noise),
            "num_runs": num_runs,
            "metastable_probability": float(metastable_probability),
            "mean_freeze_score": float(mean_freeze_score),
            "mean_switch_rate": float(mean_switch_rate),
            "mean_max_dwell": float(mean_max_dwell),
            "event_rate": float(event_rate),
            "regime_frequencies": regime_frequencies,
        })

    # ── Assemble output ───────────────────────────────────────────────
    distance_levels = sorted(distance_set)
    noise_levels = sorted(noise_set)

    return {
        "phase_points": phase_points,
        "distance_levels": distance_levels,
        "noise_levels": [float(n) for n in noise_levels],
        "phase_statistics": {
            "total_runs": total_runs,
            "total_parameter_points": len(phase_points),
        },
    }
