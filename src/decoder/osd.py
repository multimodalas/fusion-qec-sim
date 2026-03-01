"""
Ordered Statistics Decoding (OSD-0, OSD-1, and OSD-CS).

Post-processing steps for belief-propagation decoders.  When BP fails
to converge, OSD uses the soft reliability information (LLRs) to
select an information set via Gaussian elimination and solve for the
most likely error pattern.
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np

from .gf2 import gf2_row_echelon


# ═══════════════════════════════════════════════════════════════════════
# Shared OSD-0 core
# ═══════════════════════════════════════════════════════════════════════

def _osd0_core(H, llr, hard_decision, syndrome_vec):
    """Sort, row-reduce, and back-substitute (OSD-0 core).

    This is the shared computation used by both :func:`osd0` and
    :func:`osd1`.  It does NOT apply the never-degrade guard;
    callers are responsible for that.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Per-variable log-likelihood ratios, length n.
        hard_decision: Binary hard-decision vector from BP, length n.
        syndrome_vec: Target syndrome, length m.

    Returns:
        (result, reliability_order, pivot_cols, rank, valid):
            result — OSD-0 solution in original (un-permuted) space.
            reliability_order — column sort permutation (ascending |llr|).
            pivot_cols — pivot columns in permuted space.
            rank — GF(2) rank of the permuted H.
            valid — True if result satisfies the syndrome.
    """
    m, n = H.shape

    # Step 1 — Sort columns by reliability (ascending: least reliable first).
    # Pivots will land on the least-reliable independent columns;
    # the trailing (most-reliable) non-pivot columns become the info set.
    reliability_order = np.argsort(np.abs(llr), kind="stable")

    # Step 2 — Permute H columns and augment with syndrome.
    H_perm = H[:, reliability_order].astype(np.uint8)
    augmented = np.hstack([H_perm, syndrome_vec.reshape(-1, 1).astype(np.uint8)])

    # Row-reduce; pivots only searched within first n columns (not syndrome).
    R_aug, pivot_cols = gf2_row_echelon(augmented, n_pivot_cols=n)
    rank = len(pivot_cols)

    if rank == 0:
        # No pivots: cannot solve; return hard_decision as the "solution".
        return hard_decision.copy(), reliability_order, pivot_cols, rank, False

    R = R_aug[:, :n]
    s_transformed = R_aug[:, n]

    # Step 3 — Identify info set (non-pivot = most reliable independent).
    pivot_set = set(pivot_cols)
    info_cols = [c for c in range(n) if c not in pivot_set]

    # Step 4 — Set info bits from hard decision (permuted space).
    hard_perm = hard_decision[reliability_order]
    result_perm = np.zeros(n, dtype=np.uint8)
    for c in info_cols:
        result_perm[c] = hard_perm[c]

    # Step 5 — Back-substitute (top-down: R is upper-triangular in pivots).
    for i in range(rank - 1, -1, -1):
        pc = pivot_cols[i]
        rhs = s_transformed[i]
        for j in range(n):
            if j != pc:
                rhs ^= np.uint8(R[i, j]) & result_perm[j]
        result_perm[pc] = np.uint8(rhs) & np.uint8(1)

    # Step 6 — Un-permute.
    result = np.zeros(n, dtype=np.uint8)
    result[reliability_order] = result_perm

    # Check syndrome validity (callers may still apply never-degrade).
    osd_syn = (
        (H.astype(np.int32) @ result.astype(np.int32)) % 2
    ).astype(np.uint8)
    valid = np.array_equal(osd_syn, syndrome_vec)

    return result, reliability_order, pivot_cols, rank, valid


# ═══════════════════════════════════════════════════════════════════════
# Path-metric candidate key (used by OSD-CS)
# ═══════════════════════════════════════════════════════════════════════

def _candidate_key(candidate_vec, llr_abs, tie_index):
    """Deterministic candidate comparison key.

    Returns a tuple suitable for lexicographic sorting:
    ``(weight, metric, tie_index)``.

    - *weight*: Hamming weight (primary, ascending = fewer flips better).
    - *metric*: Sum of ``|llr|`` at flipped positions, rounded to 12
      decimal places via :func:`numpy.round` for float-stable ordering
      (secondary, ascending = flip least-reliable bits preferred).
    - *tie_index*: Enumeration order integer (tertiary, ascending =
      earlier combination preferred).

    Args:
        candidate_vec: Binary candidate vector, length n.
        llr_abs: Precomputed ``|llr|`` vector, length n.
        tie_index: Integer tie-breaking index for this candidate.

    Returns:
        ``(weight, metric, tie_index)`` tuple.
    """
    weight = int(np.sum(candidate_vec))
    # Determinism audit: np.round is NumPy-native, .item() yields a
    # Python float for stable tuple comparison across platforms.
    metric = np.round(
        np.dot(candidate_vec.astype(np.float64), llr_abs),
        12,
    ).item()
    return (weight, metric, tie_index)


# ═══════════════════════════════════════════════════════════════════════
# OSD-0
# ═══════════════════════════════════════════════════════════════════════

def osd0(H, llr, hard_decision, syndrome_vec=None):
    """Order-0 Ordered Statistics Decoding.

    Steps:
        1. Sort columns of *H* by ``|llr|`` ascending (least reliable first)
           using a stable sort.
        2. Row-reduce the permuted *H* over GF(2).  Pivots land on the
           least-reliable independent columns.
        3. The non-pivot (most-reliable) columns form the information set
           whose bits are kept from *hard_decision*.
        4. Back-substitute to solve for the pivot (redundant) bits.
        5. Un-permute and return the corrected word.

    The *never-degrade* guarantee ensures that if the returned word does
    not satisfy the syndrome, the original *hard_decision* is returned
    unchanged.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Per-variable log-likelihood ratios, length n.
        hard_decision: Binary hard-decision vector from BP, length n.
        syndrome_vec: Target syndrome, length m.  Defaults to all-zeros.

    Returns:
        Corrected binary vector, length n, dtype uint8.
    """
    H = np.asarray(H)
    llr = np.asarray(llr, dtype=np.float64)
    hard_decision = np.asarray(hard_decision, dtype=np.uint8)
    m, n = H.shape

    if syndrome_vec is None:
        syndrome_vec = np.zeros(m, dtype=np.uint8)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    result, _, _, _, valid = _osd0_core(H, llr, hard_decision, syndrome_vec)

    # Never-degrade guard.
    if not valid:
        return hard_decision.copy()

    return result


# ═══════════════════════════════════════════════════════════════════════
# OSD-1
# ═══════════════════════════════════════════════════════════════════════

def osd1(H, llr, hard_decision, syndrome_vec=None):
    """Order-1 Ordered Statistics Decoding.

    Extends OSD-0 by testing a single-bit flip on the least-reliable
    pivot column after the OSD-0 solution is obtained.  The candidate
    with the lowest Hamming weight that satisfies the syndrome is
    selected.

    The *never-degrade* guarantee is preserved: if neither OSD-0 nor
    the single-bit flip produces a valid syndrome match, the original
    *hard_decision* is returned unchanged.

    Deterministic tie-breaking: when OSD-0 and the flipped candidate
    have equal Hamming weight, OSD-0 is preferred.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Per-variable log-likelihood ratios, length n.
        hard_decision: Binary hard-decision vector from BP, length n.
        syndrome_vec: Target syndrome, length m.  Defaults to all-zeros.

    Returns:
        Corrected binary vector, length n, dtype uint8.
    """
    H = np.asarray(H)
    llr = np.asarray(llr, dtype=np.float64)
    hard_decision = np.asarray(hard_decision, dtype=np.uint8)
    m, n = H.shape

    if syndrome_vec is None:
        syndrome_vec = np.zeros(m, dtype=np.uint8)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    # ── OSD-0 phase ──
    result_0, reliability_order, pivot_cols, rank, osd0_valid = \
        _osd0_core(H, llr, hard_decision, syndrome_vec)

    if rank == 0:
        # No pivots: nothing to flip.
        return hard_decision.copy()

    # ── OSD-1 phase: flip the single least-reliable pivot bit ──
    # pivot_cols[0] is the least-reliable pivot in permuted space
    # (columns are sorted by ascending |llr|, so index 0 = smallest |llr|).
    flip_col_perm = pivot_cols[0]
    flip_col_orig = reliability_order[flip_col_perm]

    result_1 = result_0.copy()
    result_1[flip_col_orig] ^= 1

    osd1_syn = (
        (H.astype(np.int32) @ result_1.astype(np.int32)) % 2
    ).astype(np.uint8)
    osd1_valid = np.array_equal(osd1_syn, syndrome_vec)

    # ── Select best valid candidate ──
    # Candidates are (weight, tie_order, vector).
    # tie_order=0 for OSD-0 (preferred on ties), tie_order=1 for OSD-1.
    candidates = []
    if osd0_valid:
        candidates.append((int(np.sum(result_0)), 0, result_0))
    if osd1_valid:
        candidates.append((int(np.sum(result_1)), 1, result_1))

    if not candidates:
        # Never-degrade guard: neither candidate satisfies the syndrome.
        return hard_decision.copy()

    # Select candidate with lowest Hamming weight; ties broken by order.
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


# ═══════════════════════════════════════════════════════════════════════
# OSD-CS (Combination Sweep)
# ═══════════════════════════════════════════════════════════════════════

def osd_cs(H, llr, hard_decision, syndrome_vec=None, lam=1):
    """Combination Sweep OSD (OSD-CS).

    Extends OSD-0 by testing all combinations of flipping up to *lam*
    pivot columns.  Candidates are evaluated using the path-metric
    lexicographic ordering produced by :func:`_candidate_key`:
    ``(weight, metric, combination_index)``.

    The *never-degrade* guarantee is preserved: if no candidate satisfies
    the syndrome, the original *hard_decision* is returned.

    Deterministic enumeration: combinations are generated via
    :func:`itertools.combinations` in lexicographic order over pivot
    column indices (ascending).

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Per-variable log-likelihood ratios, length n.
        hard_decision: Binary hard-decision vector from BP, length n.
        syndrome_vec: Target syndrome, length m.  Defaults to all-zeros.
        lam: Maximum number of pivot bits to flip simultaneously.
            Default 1 (tests all single-pivot flips).
            ``lam=0`` is equivalent to OSD-0.

    Returns:
        Corrected binary vector, length n, dtype uint8.

    Raises:
        ValueError: If *lam* < 0.
    """
    H = np.asarray(H)
    llr = np.asarray(llr, dtype=np.float64)
    hard_decision = np.asarray(hard_decision, dtype=np.uint8)
    m, n = H.shape
    # Pre-cast once to avoid repeated allocation inside candidate loop
    H_int = H.astype(np.int32)

    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")

    if syndrome_vec is None:
        syndrome_vec = np.zeros(m, dtype=np.uint8)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    # ── Phase 1: OSD-0 core ──
    result_0, reliability_order, pivot_cols, rank, osd0_valid = \
        _osd0_core(H, llr, hard_decision, syndrome_vec)
    # Ensure dtype stability (backward compatibility & deterministic output)
    result_0 = result_0.astype(np.uint8, copy=False)

    if rank == 0:
        return hard_decision.copy()

    # Effective lambda: cannot flip more pivots than exist.
    effective_lam = min(lam, rank)

    # Performance guard: warn on large candidate counts.
    n_cands = sum(math.comb(rank, k) for k in range(effective_lam + 1))
    if n_cands > 10000:
        warnings.warn(
            f"OSD-CS will evaluate {n_cands} candidates "
            f"(rank={rank}, lam={effective_lam}). This may be slow.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Phase 2: Enumerate candidates ──
    llr_abs = np.abs(llr)
    best_key = None   # (weight, metric, combo_index)
    best_candidate = None
    combo_index = 0

    # OSD-0 candidate (flip 0 pivots).
    if osd0_valid:
        key = _candidate_key(result_0, llr_abs, combo_index)
        best_key = key
        best_candidate = result_0
    combo_index += 1

    # Enumerate combinations of 1..effective_lam pivot flips.
    # Determinism audit: itertools.combinations yields tuples in
    # lexicographic order per the Python specification.
    for num_flips in range(1, effective_lam + 1):
        for combo in itertools.combinations(range(rank), num_flips):
            candidate = result_0.copy()
            for pivot_idx in combo:
                flip_col_perm = pivot_cols[pivot_idx]
                flip_col_orig = reliability_order[flip_col_perm]
                candidate[flip_col_orig] ^= 1

            # Syndrome check.
            cand_syn = (
                (H_int @ candidate.astype(np.int32)) % 2
            ).astype(np.uint8)
            cand_valid = np.array_equal(cand_syn, syndrome_vec)

            if cand_valid:
                key = _candidate_key(candidate, llr_abs, combo_index)
                if best_key is None or key < best_key:
                    best_key = key
                    best_candidate = candidate

            combo_index += 1

    # Never-degrade guard.
    if best_candidate is None:
        return hard_decision.copy()

    return best_candidate


# ═══════════════════════════════════════════════════════════════════════
# MP-aware OSD-1 (v3.5.0)
# ═══════════════════════════════════════════════════════════════════════

def mp_osd1_postprocess(H, llr, hard_bp, L_post, syndrome_vec):
    """MP-aware OSD-1: orders flips by posterior LLR magnitude.

    Unlike standard OSD-1 which sorts columns by channel LLR magnitude,
    this variant uses the BP posterior beliefs ``abs(L_post)`` as the
    reliability metric.  This exploits the message-passing information
    to produce a more informed information-set selection.

    The algorithm is identical to OSD-1 except for the reliability
    ordering source.  All operations are deterministic.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Channel LLR vector, length n (unused for ordering but
            kept for API consistency).
        hard_bp: BP hard-decision vector, length n, dtype uint8.
        L_post: Posterior LLR vector from BP, length n.
        syndrome_vec: Target syndrome, length m, dtype uint8.

    Returns:
        Corrected binary vector, length n, dtype uint8.
        Satisfies the never-degrade guarantee: if neither OSD-0 nor
        OSD-1 produces a valid syndrome match, *hard_bp* is returned.
    """
    H = np.asarray(H)
    L_post = np.asarray(L_post, dtype=np.float64)
    hard_bp = np.asarray(hard_bp, dtype=np.uint8)
    m, n = H.shape

    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    # Use _osd0_core with posterior LLR as the reliability metric.
    result_0, reliability_order, pivot_cols, rank, osd0_valid = \
        _osd0_core(H, L_post, hard_bp, syndrome_vec)

    if rank == 0:
        return hard_bp.copy()

    # OSD-1 phase: flip the single least-reliable pivot bit.
    flip_col_perm = pivot_cols[0]
    flip_col_orig = reliability_order[flip_col_perm]

    result_1 = result_0.copy()
    result_1[flip_col_orig] ^= 1

    osd1_syn = (
        (H.astype(np.int32) @ result_1.astype(np.int32)) % 2
    ).astype(np.uint8)
    osd1_valid = np.array_equal(osd1_syn, syndrome_vec)

    # Select best valid candidate.
    candidates = []
    if osd0_valid:
        candidates.append((int(np.sum(result_0)), 0, result_0))
    if osd1_valid:
        candidates.append((int(np.sum(result_1)), 1, result_1))

    if not candidates:
        return hard_bp.copy()

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]
