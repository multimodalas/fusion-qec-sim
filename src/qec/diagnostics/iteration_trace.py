"""
Deterministic iteration-trace diagnostics (v4.3.0).

Analyses BP iteration traces (LLR traces, energy traces, correction vectors)
to detect trapping sets, oscillatory message passing, unstable convergence,
and correction vector cycling.

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


# ── 1. Persistent Error Indicator (PEI) ────────────────────────────


def compute_persistent_error_indicator(
    llr_trace: List[np.ndarray],
    window: int = 5,
) -> Dict[str, Any]:
    """Detect trapping sets via persistent sign errors.

    A variable node is flagged if its LLR sign indicates an error
    (negative LLR) for *window* consecutive iterations at the end
    of the trace.

    Parameters
    ----------
    llr_trace : list of 1-D arrays
        Per-iteration LLR vectors.  Not mutated.
    window : int
        Number of consecutive iterations to check (default 5).

    Returns
    -------
    dict with keys ``pei_vector``, ``pei_count``, ``pei_ratio``.
    """
    n_iters = len(llr_trace)
    if n_iters == 0:
        return {"pei_vector": np.array([], dtype=np.int32),
                "pei_count": 0, "pei_ratio": 0.0}

    n_vars = len(llr_trace[0])
    w = min(window, n_iters)

    # A node has persistent error if LLR < 0 for last *w* iterations.
    pei = np.ones(n_vars, dtype=np.int32)
    for t in range(n_iters - w, n_iters):
        llr_t = np.asarray(llr_trace[t], dtype=np.float64)
        pei &= (llr_t < 0).astype(np.int32)

    pei_count = int(np.sum(pei))
    pei_ratio = float(pei_count) / n_vars if n_vars > 0 else 0.0
    return {
        "pei_vector": pei,
        "pei_count": pei_count,
        "pei_ratio": pei_ratio,
    }


# ── 2. Belief Oscillation Index (BOI) ──────────────────────────────


def compute_belief_oscillation_index(
    llr_trace: List[np.ndarray],
) -> Dict[str, Any]:
    """Count LLR sign flips across iterations for each variable node.

    Parameters
    ----------
    llr_trace : list of 1-D arrays
        Per-iteration LLR vectors.  Not mutated.

    Returns
    -------
    dict with keys ``boi_vector``, ``boi_mean``, ``boi_max``.
    """
    n_iters = len(llr_trace)
    if n_iters < 2:
        n_vars = len(llr_trace[0]) if n_iters == 1 else 0
        return {"boi_vector": np.zeros(n_vars, dtype=np.int32),
                "boi_mean": 0.0, "boi_max": 0}

    n_vars = len(llr_trace[0])
    flips = np.zeros(n_vars, dtype=np.int32)
    for t in range(1, n_iters):
        prev = np.asarray(llr_trace[t - 1], dtype=np.float64)
        curr = np.asarray(llr_trace[t], dtype=np.float64)
        # Sign flip: sign changes between iterations (treat 0 as non-negative).
        sign_prev = np.sign(prev)
        sign_curr = np.sign(curr)
        flips += (sign_prev != sign_curr).astype(np.int32)

    boi_mean = float(np.mean(flips))
    boi_max = int(np.max(flips))
    return {
        "boi_vector": flips,
        "boi_mean": boi_mean,
        "boi_max": boi_max,
    }


# ── 3. Oscillation Depth (OD) ──────────────────────────────────────


def compute_oscillation_depth(
    llr_trace: List[np.ndarray],
    window: int = 10,
) -> Dict[str, Any]:
    """Measure LLR oscillation amplitude over the last *window* iterations.

    OD_j = max(LLR_j) − min(LLR_j) over the window.

    Parameters
    ----------
    llr_trace : list of 1-D arrays
        Per-iteration LLR vectors.  Not mutated.
    window : int
        Number of trailing iterations to consider (default 10).

    Returns
    -------
    dict with keys ``od_vector``, ``od_mean``, ``od_max``.
    """
    n_iters = len(llr_trace)
    if n_iters == 0:
        return {"od_vector": np.array([], dtype=np.float64),
                "od_mean": 0.0, "od_max": 0.0}

    n_vars = len(llr_trace[0])
    w = min(window, n_iters)
    start = n_iters - w

    # Stack the window into a (w, n_vars) array.
    window_arr = np.array(
        [np.asarray(llr_trace[t], dtype=np.float64) for t in range(start, n_iters)],
        dtype=np.float64,
    )
    od = np.max(window_arr, axis=0) - np.min(window_arr, axis=0)

    od_mean = float(np.mean(od))
    od_max = float(np.max(od))
    return {
        "od_vector": od,
        "od_mean": od_mean,
        "od_max": od_max,
    }


# ── 4. Convergence Instability Score (CIS) ─────────────────────────


def compute_convergence_instability_score(
    energy_trace: List[float],
    window: int = 10,
) -> Dict[str, Any]:
    """Detect unstable convergence via energy trace variance.

    CIS = variance(energy_trace[-window:]).

    Parameters
    ----------
    energy_trace : list of float
        Per-iteration energy values.  Not mutated.
    window : int
        Number of trailing iterations to consider (default 10).

    Returns
    -------
    dict with key ``cis``.
    """
    if len(energy_trace) == 0:
        return {"cis": 0.0}

    w = min(window, len(energy_trace))
    tail = np.array(energy_trace[-w:], dtype=np.float64)
    cis = float(np.var(tail))
    return {"cis": cis}


# ── 5. Correction Vector Fluctuation (CVF) ─────────────────────────


def compute_correction_vector_fluctuation(
    correction_vectors: List[np.ndarray],
) -> Dict[str, Any]:
    """Detect correction cycling via Euclidean norm of consecutive diffs.

    CVF_i = ||corr[i] − corr[i−1]||_2

    Parameters
    ----------
    correction_vectors : list of 1-D arrays
        Per-iteration correction vectors.  Not mutated.

    Returns
    -------
    dict with keys ``cvf_mean``, ``cvf_max``.
    """
    n = len(correction_vectors)
    if n < 2:
        return {"cvf_mean": 0.0, "cvf_max": 0.0}

    norms = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        diff = (np.asarray(correction_vectors[i], dtype=np.float64)
                - np.asarray(correction_vectors[i - 1], dtype=np.float64))
        norms[i - 1] = float(np.linalg.norm(diff))

    cvf_mean = float(np.mean(norms))
    cvf_max = float(np.max(norms))
    return {
        "cvf_mean": cvf_mean,
        "cvf_max": cvf_max,
    }


# ── Composite Metric ───────────────────────────────────────────────


def compute_iteration_trace_metrics(
    llr_trace: List[np.ndarray],
    energy_trace: List[float],
    correction_vectors: List[np.ndarray],
) -> Dict[str, Any]:
    """Compute all iteration-trace diagnostics in a single call.

    Parameters
    ----------
    llr_trace : list of 1-D arrays
        Per-iteration LLR vectors.
    energy_trace : list of float
        Per-iteration energy values.
    correction_vectors : list of 1-D arrays
        Per-iteration correction vectors.

    Returns
    -------
    dict with keys:
      ``persistent_error_indicator``,
      ``belief_oscillation_index``,
      ``oscillation_depth``,
      ``convergence_instability_score``,
      ``correction_vector_fluctuation``.
    """
    pei = compute_persistent_error_indicator(llr_trace)
    boi = compute_belief_oscillation_index(llr_trace)
    od = compute_oscillation_depth(llr_trace)
    cis_result = compute_convergence_instability_score(energy_trace)
    cvf = compute_correction_vector_fluctuation(correction_vectors)

    return {
        "persistent_error_indicator": pei,
        "belief_oscillation_index": boi,
        "oscillation_depth": od,
        "convergence_instability_score": cis_result["cis"],
        "correction_vector_fluctuation": cvf,
    }
