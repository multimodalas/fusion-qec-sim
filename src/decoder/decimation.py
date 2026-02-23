"""
Deterministic decimation for belief-propagation decoders.

Threshold-based bit commitment with deterministic tie-breaking
and optional degree-1 check peeling.  No randomness introduced.
All operations are fully reproducible across runs.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from typing import Dict, List, Optional, Tuple

# Scaling factor applied to max(|llr|) when clamping committed variables.
# Rationale: committed bits must dominate all other beliefs.  A factor of
# 100x the largest channel LLR ensures the commitment is never overridden
# by message passing, while staying within reasonable float64 range.
LLR_CLAMP_FACTOR = 100.0


def decimate(
    H: np.ndarray,
    beliefs: np.ndarray,
    threshold: float,
    hard_decision: np.ndarray,
    committed: Optional[np.ndarray] = None,
    peel: bool = False,
    syndrome_vec: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Commit high-confidence bits via threshold decimation.

    Variables whose belief magnitude meets or exceeds *threshold* are
    committed to their hard-decision value.  Tie-breaking is by ascending
    variable index (deterministic).

    Args:
        H: Binary parity-check matrix, shape (m, n).
        beliefs: Per-variable total beliefs (L_total), length n.
        threshold: Magnitude threshold for commitment.  Variables with
            ``|beliefs[v]| >= threshold`` are committed.
        hard_decision: Current hard-decision vector, length n.
        committed: Boolean mask of already-committed variables, length n.
            Defaults to all-False (nothing committed yet).
        peel: If True, propagate forced values through degree-1 checks
            after threshold commitment.
        syndrome_vec: Binary syndrome vector, length m.  Defaults to
            all-zeros.

    Returns:
        ``(new_hard, new_committed, forced_values)``:
            *new_hard* — updated hard-decision vector with committed bits.
            *new_committed* — updated boolean committed mask.
            *forced_values* — int array of shape (k, 2) with
            ``(variable_index, value)`` pairs forced by peeling
            (empty shape ``(0, 2)`` if *peel* is False or no forced values).
    """
    H = np.asarray(H, dtype=np.uint8)
    beliefs = np.asarray(beliefs, dtype=np.float64)
    hard_decision = np.asarray(hard_decision, dtype=np.uint8)
    m, n = H.shape

    if committed is None:
        committed = np.zeros(n, dtype=bool)
    else:
        committed = np.asarray(committed, dtype=bool).copy()

    if syndrome_vec is None:
        syndrome_vec = np.zeros(m, dtype=np.uint8)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    new_hard = hard_decision.copy()
    new_committed = committed.copy()

    # ── Threshold commitment ──
    # Determinism audit: ascending index order via range(n).
    for v in range(n):
        if new_committed[v]:
            continue
        if np.abs(beliefs[v]) >= threshold:
            new_hard[v] = np.uint8(0) if beliefs[v] >= 0.0 else np.uint8(1)
            new_committed[v] = True

    # ── Optional degree-1 peeling ──
    forced_list: List[Tuple[int, int]] = []

    if peel:
        # Build residual degree per check (count of uncommitted neighbors).
        # Determinism audit: ascending check-index order for processing.
        check_degree = np.zeros(m, dtype=int)
        check_neighbors: List[List[int]] = [[] for _ in range(m)]
        for c in range(m):
            for v in range(n):
                if H[c, v] == 1 and not new_committed[v]:
                    check_degree[c] += 1
                    check_neighbors[c].append(v)

        # Seed the peeling queue with current degree-1 checks.
        peel_queue: deque[int] = deque()
        for c in range(m):
            if check_degree[c] == 1:
                peel_queue.append(c)

        while peel_queue:
            c = peel_queue.popleft()
            if check_degree[c] != 1:
                continue

            # Find the single uncommitted neighbor.
            target_v = -1
            for v in check_neighbors[c]:
                if not new_committed[v]:
                    target_v = v
                    break
            if target_v == -1:
                continue

            # Compute the forced value: syndrome[c] XOR sum of committed
            # neighbors' contributions.
            rhs = int(syndrome_vec[c])
            for v in range(n):
                if H[c, v] == 1 and v != target_v and new_committed[v]:
                    rhs ^= int(new_hard[v])
            forced_val = rhs & 1

            new_hard[target_v] = np.uint8(forced_val)
            new_committed[target_v] = True
            forced_list.append((target_v, forced_val))

            # Update degrees for all checks involving target_v.
            # Determinism audit: ascending check index via range(m).
            for c2 in range(m):
                if H[c2, target_v] == 1:
                    check_degree[c2] -= 1
                    if check_degree[c2] == 1:
                        peel_queue.append(c2)

    if forced_list:
        forced_values = np.array(forced_list, dtype=int)
    else:
        forced_values = np.empty((0, 2), dtype=int)

    return new_hard, new_committed, forced_values


def decimation_round(
    H: np.ndarray,
    llr: np.ndarray,
    threshold: float,
    bp_kwargs: dict,
    max_rounds: int = 5,
    peel: bool = False,
    syndrome_vec: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int]:
    """Run iterative decimation: BP -> commit -> clamp -> BP -> ...

    Each round:
        1. Run BP with current (possibly clamped) LLRs.
        2. Extract beliefs from the hard-decision sign and LLR magnitudes.
        3. Commit variables above *threshold* via :func:`decimate`.
        4. Clamp committed LLRs to ``sign * clamp_magnitude``.
        5. Repeat until syndrome satisfied, all bits committed, no new
           commits, or *max_rounds* reached.

    The clamping magnitude is derived from the channel:
    ``clamp_magnitude = max(max(|llr|), 1.0) * LLR_CLAMP_FACTOR``.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Channel LLR vector, length n.
        threshold: Magnitude threshold for commitment per round.
        bp_kwargs: Keyword arguments forwarded to :func:`bp_decode`
            (e.g. ``mode``, ``max_iters``, ``schedule``).
        max_rounds: Maximum decimation rounds (default 5).
        peel: Enable degree-1 check peeling in each round.
        syndrome_vec: Binary syndrome vector, length m.

    Returns:
        ``(correction, total_bp_iters, rounds_used)``:
            *correction* — final hard-decision binary vector.
            *total_bp_iters* — sum of BP iterations across all rounds.
            *rounds_used* — number of decimation rounds executed.
    """
    # Lazy import to avoid circular dependency.
    from ..qec_qldpc_codes import bp_decode, syndrome

    H = np.asarray(H, dtype=np.uint8)
    llr = np.asarray(llr, dtype=np.float64)
    m, n = H.shape

    if syndrome_vec is None:
        syndrome_vec = np.zeros(m, dtype=np.uint8)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    clamp_magnitude = max(float(np.max(np.abs(llr))), 1.0) * LLR_CLAMP_FACTOR

    committed = np.zeros(n, dtype=bool)
    clamped_llr = llr.copy()
    total_bp_iters = 0

    for rnd in range(max_rounds):
        # ── BP pass ──
        # Use indexed unpack for safety with llr_history.
        result = bp_decode(
            H, clamped_llr, syndrome_vec=syndrome_vec, **bp_kwargs,
        )
        correction, iters = result[0], result[1]
        total_bp_iters += iters

        # Check convergence.
        residual_syn = syndrome(H, correction)
        if np.array_equal(residual_syn, syndrome_vec):
            return correction, total_bp_iters, rnd + 1

        # ── Build beliefs from current LLR + hard decision ──
        # Belief sign follows hard decision; magnitude from clamped LLR.
        signs = np.where(correction == 0, 1.0, -1.0)
        beliefs = signs * np.abs(clamped_llr)

        # ── Decimate ──
        prev_committed_count = int(np.sum(committed))
        new_hard, committed, _ = decimate(
            H, beliefs, threshold, correction,
            committed=committed, peel=peel, syndrome_vec=syndrome_vec,
        )

        new_committed_count = int(np.sum(committed))
        if new_committed_count == prev_committed_count:
            # No progress; stop.
            return correction, total_bp_iters, rnd + 1

        # ── Clamp committed LLRs ──
        for v in range(n):
            if committed[v]:
                sign = 1.0 if new_hard[v] == 0 else -1.0
                clamped_llr[v] = sign * clamp_magnitude

        if new_committed_count == n:
            # All bits committed; verify syndrome before returning.
            full_syn = syndrome(H, new_hard)
            if np.array_equal(full_syn, syndrome_vec):
                return new_hard, total_bp_iters, rnd + 1
            # Fully committed state fails syndrome; fall back to last
            # BP correction (best available).
            return correction, total_bp_iters, rnd + 1

    # Max rounds exhausted; return last correction.
    return correction, total_bp_iters, max_rounds
