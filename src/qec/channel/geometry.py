"""
Channel geometry functions for deterministic LLR construction.

Provides syndrome-field projections and pseudo-prior bias computation
for structural channel-geometry interventions.  All functions are pure
and deterministic.
"""

from __future__ import annotations

import numpy as np


def syndrome_field(H: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Compute standard syndrome field: Hᵀ (1 - 2s).

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    s : np.ndarray
        Syndrome vector, length m.

    Returns
    -------
    np.ndarray
        LLR vector of length n.
    """
    b = (1 - 2 * np.asarray(s, dtype=np.float64))
    return np.asarray(H, dtype=np.float64).T @ b


def centered_syndrome_field(H: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Compute centered syndrome field: Hᵀ (b - mean(b)) where b = 1 - 2s.

    Removes the uniform syndrome bias that collapses inference geometry
    while preserving directional information.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    s : np.ndarray
        Syndrome vector, length m.

    Returns
    -------
    np.ndarray
        Centered LLR vector of length n.
    """
    b = (1 - 2 * np.asarray(s, dtype=np.float64))
    b_centered = b - np.mean(b)
    return np.asarray(H, dtype=np.float64).T @ b_centered


def pseudo_prior_bias(H: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Compute parity-derived pseudo-prior bias: Hᵀ (1 - 2s).

    Deterministic variable prior derived only from parity structure
    and syndrome.  Contains no oracle information.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    s : np.ndarray
        Syndrome vector, length m.

    Returns
    -------
    np.ndarray
        Bias vector of length n.
    """
    b = (1 - 2 * np.asarray(s, dtype=np.float64))
    return np.asarray(H, dtype=np.float64).T @ b


def apply_pseudo_prior(
    llr: np.ndarray,
    bias: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Apply pseudo-prior injection: LLR + κ * bias.

    Parameters
    ----------
    llr : np.ndarray
        Input LLR vector, length n.
    bias : np.ndarray
        Pseudo-prior bias vector, length n.
    kappa : float
        Pseudo-prior strength.

    Returns
    -------
    np.ndarray
        Adjusted LLR vector.
    """
    return np.asarray(llr, dtype=np.float64) + kappa * np.asarray(bias, dtype=np.float64)
