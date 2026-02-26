"""
BSC syndrome-only channel model.

Returns a uniform LLR vector whose magnitude reflects the channel
crossover probability but whose sign carries no information about
individual error positions.  The decoder must rely entirely on the
syndrome to locate errors, producing realistic (non-zero) FER.
"""

from __future__ import annotations

import numpy as np

from .base import ChannelModel


class BSCSyndromeChannel(ChannelModel):
    """BSC syndrome-only channel: uniform LLR with no sign leakage.

    The returned LLR vector is identical regardless of the error vector
    content, ensuring the decoder receives no oracle side-information.
    """

    def compute_llr(
        self,
        p: float,
        n: int,
        error_vector: np.ndarray | None = None,
    ) -> np.ndarray:
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")

        eps = 1e-30
        base_llr = np.log((1.0 - p + eps) / (p + eps))
        return np.full(n, base_llr, dtype=np.float64)
