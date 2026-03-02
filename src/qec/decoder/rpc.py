"""
Redundant Parity Check (RPC) augmentation — deterministic row-pair
combination for structural geometry intervention.

This module builds an augmented parity-check system by appending
XOR-combined row pairs to the original matrix H, together with
corresponding syndrome entries.  The augmentation is fully
deterministic: row pairs are enumerated in lexicographic order and
filtered by Hamming-weight bounds.

When disabled (``RPCConfig.enabled is False``), the original (H, s)
pair is returned unchanged, guaranteeing baseline identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RPCConfig:
    """Configuration for redundant parity-check augmentation.

    Attributes
    ----------
    enabled : bool
        Master switch.  When *False* the builder returns (H, s) unchanged.
    max_rows : int
        Maximum number of augmented rows to append.
    w_min : int
        Minimum Hamming weight of a combined row to be accepted.
    w_max : int
        Maximum Hamming weight of a combined row to be accepted.
    """

    enabled: bool = False
    max_rows: int = 64
    w_min: int = 2
    w_max: int = 32


@dataclass(frozen=True)
class StructuralConfig:
    """Top-level structural geometry configuration.

    Groups all opt-in structural extensions under a single object
    so that callers need only pass one config argument.

    Attributes
    ----------
    rpc : RPCConfig
        Redundant parity-check augmentation settings.
    """

    rpc: RPCConfig = field(default_factory=RPCConfig)


def build_rpc_augmented_system(
    H: np.ndarray,
    s: np.ndarray,
    config: RPCConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an augmented (H', s') by appending XOR row-pair combinations.

    Row pairs (i, j) with i < j are enumerated in strict lexicographic
    order.  For each pair the combined row ``r = H[i] ^ H[j]`` is
    computed; if its Hamming weight falls within ``[w_min, w_max]`` the
    row and the corresponding syndrome bit ``s[i] ^ s[j]`` are appended.

    Collection stops once *max_rows* feasible rows have been gathered or
    all pairs have been exhausted.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape ``(m, n)``, dtype uint8.
    s : np.ndarray
        Syndrome vector of length ``m``, dtype uint8.
    config : RPCConfig
        Augmentation parameters.

    Returns
    -------
    (H_aug, s_aug) : tuple of np.ndarray
        If ``config.enabled`` is *False*, returns the original ``(H, s)``
        without copy.  Otherwise returns new arrays with appended rows.
    """
    if not config.enabled:
        return H, s

    H = np.asarray(H, dtype=np.uint8)
    s = np.asarray(s, dtype=np.uint8)

    m, n = H.shape

    new_rows = []
    new_synd = []
    collected = 0

    for i in range(m):
        if collected >= config.max_rows:
            break
        for j in range(i + 1, m):
            if collected >= config.max_rows:
                break
            combined = H[i] ^ H[j]
            w = int(np.sum(combined))
            if config.w_min <= w <= config.w_max:
                new_rows.append(combined)
                new_synd.append(s[i] ^ s[j])
                collected += 1

    if not new_rows:
        return H, s

    H_aug = np.vstack([H, np.array(new_rows, dtype=np.uint8)])
    s_aug = np.concatenate([s, np.array(new_synd, dtype=np.uint8)])

    return H_aug, s_aug
