"""
v5.5.0 — Spectral–Boundary Alignment Diagnostics.

Measures the cosine similarity between spectral eigenvectors of the
Tanner graph (from v5.4 spectral analysis) and BP decision boundary
directions (from v5.3 boundary analysis).  High alignment indicates
that a spectral mode points in a direction similar to the BP decision
boundary, suggesting that localized spectral modes correspond to
fragile BP decision boundaries.

Operates purely on pre-computed vectors — does not run BP decoding.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_spectral_boundary_alignment(
    spectral_modes: list[np.ndarray],
    boundary_direction: np.ndarray,
) -> dict[str, Any]:
    """Compute alignment between spectral eigenvectors and a BP boundary.

    Parameters
    ----------
    spectral_modes : list[np.ndarray]
        Spectral eigenvectors from v5.4 Tanner spectral analysis.
    boundary_direction : np.ndarray
        Decision boundary vector from v5.3 boundary analysis.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary of alignment diagnostics.

    Raises
    ------
    ValueError
        If no spectral modes are provided, or if dimension mismatches
        are detected between a spectral mode and the boundary direction.
    """
    if len(spectral_modes) == 0:
        raise ValueError("spectral_modes must not be empty")

    d = np.asarray(boundary_direction, dtype=np.float64)

    # ── Normalize boundary direction ──────────────────────────────
    d_norm = np.linalg.norm(d)
    if d_norm == 0.0:
        # Zero boundary direction: all alignments are zero.
        return {
            "alignment_scores": [0.0] * len(spectral_modes),
            "max_alignment": 0.0,
            "mean_alignment": 0.0,
            "dominant_alignment_mode": 0,
            "mode_count": len(spectral_modes),
        }
    d_normalized = d / d_norm

    # ── Compute alignment for each spectral mode ──────────────────
    alignment_scores: list[float] = []

    for i, mode in enumerate(spectral_modes):
        v = np.asarray(mode, dtype=np.float64)

        if v.shape != d.shape:
            raise ValueError(
                f"Dimension mismatch: spectral_modes[{i}] has shape "
                f"{v.shape}, boundary_direction has shape {d.shape}"
            )

        # Deterministic sign orientation: flip if first element is negative.
        if v[0] < 0.0:
            v = -v

        # Normalize eigenvector.
        v_norm = np.linalg.norm(v)
        if v_norm == 0.0:
            alignment_scores.append(0.0)
            continue
        v_normalized = v / v_norm

        # Cosine similarity (absolute value).
        dot = float(np.dot(v_normalized, d_normalized))
        alignment = abs(dot)
        alignment_scores.append(alignment)

    # ── Aggregate metrics ─────────────────────────────────────────
    max_alignment = max(alignment_scores)
    mean_alignment = float(np.mean(alignment_scores))
    dominant_alignment_mode = int(np.argmax(alignment_scores))

    return {
        "alignment_scores": alignment_scores,
        "max_alignment": max_alignment,
        "mean_alignment": mean_alignment,
        "dominant_alignment_mode": dominant_alignment_mode,
        "mode_count": len(spectral_modes),
    }
