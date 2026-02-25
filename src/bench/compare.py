"""
Structured comparison suite: threshold tables, runtime scaling, iteration
distribution analysis.

All outputs are deterministic given the same input data.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# A) Threshold Tables
# ─────────────────────────────────────────────────────────────────────

def compute_threshold_table(
    results: list[dict[str, Any]],
    decoder_name: str,
) -> dict[str, Any]:
    """Estimate the error threshold from FER curves across distances.

    For each adjacent pair of distances (d_i, d_j), finds the physical
    error rate *p* at which FER(d_i, p) and FER(d_j, p) cross via
    linear interpolation.  The final threshold estimate is the median
    of all crossing points.

    Parameters
    ----------
    results:
        The ``results`` list from a benchmark output, filtered to
        records belonging to *decoder_name*.
    decoder_name:
        Name of the decoder (for labelling).

    Returns
    -------
    dict with:
        ``decoder``              : str
        ``threshold_estimate_p`` : float | None
        ``crossings``            : list[dict]
        ``method``               : ``"crossing_point"``
        ``notes``                : str
    """
    # Group FER by distance → {distance: [(p, fer), ...]}.
    by_dist: dict[int, list[tuple[float, float]]] = {}
    for rec in results:
        d = rec["distance"]
        by_dist.setdefault(d, []).append((rec["p"], rec["fer"]))

    # Sort each group by p (deterministic).
    for d in by_dist:
        by_dist[d].sort()

    distances = sorted(by_dist.keys())
    crossings: list[dict[str, Any]] = []

    for i in range(len(distances) - 1):
        d_lo, d_hi = distances[i], distances[i + 1]
        pts_lo = by_dist[d_lo]
        pts_hi = by_dist[d_hi]

        # Build p→FER lookup for each distance.
        p_set_lo = {p for p, _ in pts_lo}
        p_set_hi = {p for p, _ in pts_hi}
        common_ps = sorted(p_set_lo & p_set_hi)

        if len(common_ps) < 2:
            continue

        fer_lo = {p: fer for p, fer in pts_lo}
        fer_hi = {p: fer for p, fer in pts_hi}

        # diff = FER(d_lo, p) - FER(d_hi, p)
        # At threshold, smaller distance has higher FER → diff changes sign.
        diffs = [(p, fer_lo[p] - fer_hi[p]) for p in common_ps]

        for j in range(len(diffs) - 1):
            p1, d1 = diffs[j]
            p2, d2 = diffs[j + 1]
            if d1 * d2 < 0:
                # Linear interpolation to find crossing.
                t = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
                p_cross = p1 + t * (p2 - p1)
                crossings.append({
                    "d_low": d_lo,
                    "d_high": d_hi,
                    "p_cross": round(float(p_cross), 8),
                })

    threshold: float | None = None
    if crossings:
        vals = sorted(c["p_cross"] for c in crossings)
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            threshold = vals[mid]
        else:
            threshold = round((vals[mid - 1] + vals[mid]) / 2.0, 8)

    notes = (
        f"{len(crossings)} crossing(s) found across {len(distances)} distances"
        if crossings
        else "No FER curve crossings detected"
    )

    return {
        "decoder": decoder_name,
        "threshold_estimate_p": threshold,
        "crossings": crossings,
        "method": "crossing_point",
        "notes": notes,
    }


# ─────────────────────────────────────────────────────────────────────
# B) Runtime Scaling Summaries
# ─────────────────────────────────────────────────────────────────────

def compute_runtime_scaling(
    results: list[dict[str, Any]],
    decoder_name: str,
) -> dict[str, Any]:
    """Summarize runtime latency vs distance for one decoder variant.

    If there are at least 2 distinct distances with runtime data, a
    log-log least-squares fit is performed to estimate the scaling
    exponent (slope).

    Returns
    -------
    dict with:
        ``decoder``   : str
        ``points``    : list[dict] — (distance, average_latency_us)
        ``slope``     : float | None — log-log scaling exponent
        ``notes``     : str
    """
    # Aggregate: for each distance, collect mean latency values.
    lat_by_dist: dict[int, list[float]] = {}
    for rec in results:
        rt = rec.get("runtime")
        if rt is None:
            continue
        d = rec["distance"]
        lat_by_dist.setdefault(d, []).append(float(rt["average_latency_us"]))

    points: list[dict[str, Any]] = []
    for d in sorted(lat_by_dist.keys()):
        mean_lat = float(np.mean(lat_by_dist[d]))
        points.append({"distance": d, "average_latency_us": round(mean_lat, 2)})

    slope: float | None = None
    if len(points) >= 2:
        log_d = np.array([math.log(pt["distance"]) for pt in points])
        log_l = np.array([
            math.log(pt["average_latency_us"])
            for pt in points
            if pt["average_latency_us"] > 0
        ])
        if len(log_d) == len(log_l) and len(log_l) >= 2:
            # Least-squares: log_l = slope * log_d + intercept
            A = np.vstack([log_d, np.ones(len(log_d))]).T
            result, _, _, _ = np.linalg.lstsq(A, log_l, rcond=None)
            slope = round(float(result[0]), 4)

    notes = (
        f"Log-log slope from {len(points)} distance point(s)"
        if slope is not None
        else "Insufficient runtime data for slope estimation"
    )

    return {
        "decoder": decoder_name,
        "points": points,
        "slope": slope,
        "notes": notes,
    }


# ─────────────────────────────────────────────────────────────────────
# C) Iteration Distribution Analysis
# ─────────────────────────────────────────────────────────────────────

def compute_iteration_histogram(
    iter_counts: list[int],
) -> dict[str, Any]:
    """Build a compact histogram from a list of iteration counts.

    Returns
    -------
    dict with:
        ``mean_iters`` : float
        ``histogram``  : {``"iters"``: list[int], ``"counts"``: list[int]}
    """
    if not iter_counts:
        return {
            "mean_iters": 0.0,
            "histogram": {"iters": [], "counts": []},
        }

    mean_val = float(np.mean(iter_counts))

    # Build histogram using deterministic sorted unique values.
    unique, counts = np.unique(iter_counts, return_counts=True)

    return {
        "mean_iters": round(mean_val, 4),
        "histogram": {
            "iters": [int(x) for x in unique],
            "counts": [int(x) for x in counts],
        },
    }


def aggregate_iteration_summaries(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract iteration distribution summaries from benchmark results.

    Each result record that has an ``"iter_histogram"`` key is included.
    """
    summaries: list[dict[str, Any]] = []
    for rec in results:
        hist = rec.get("iter_histogram")
        if hist is not None:
            summaries.append({
                "decoder": rec["decoder"],
                "distance": rec["distance"],
                "p": rec["p"],
                "mean_iters": rec["mean_iters"],
                "histogram": hist,
            })
    return summaries
