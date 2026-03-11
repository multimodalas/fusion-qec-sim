"""
v11.0.0 — BP Stability Probe.

Runs short belief-propagation probes to detect decoder instability
without modifying the decoder core.  This module is observational:
it invokes the decoder on small synthetic error patterns and tracks
convergence signals.

Layer 1 extension — read-only diagnostic.
Does not modify the BP algorithm or decoder internals.
Fully deterministic: explicit seed injection, no hidden randomness.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import numpy as np


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


class BPStabilityProbe:
    """Probe decoder stability via short BP runs.

    Generates small random error patterns, computes syndromes, runs
    abbreviated BP, and tracks convergence signals.

    Parameters
    ----------
    trials : int
        Number of probe trials (default 50).
    iterations : int
        Maximum BP iterations per trial (default 10).
    seed : int
        Base seed for deterministic execution (default 0).
    """

    def __init__(
        self,
        trials: int = 50,
        iterations: int = 10,
        seed: int = 0,
    ) -> None:
        self.trials = trials
        self.iterations = iterations
        self.seed = seed

    def probe(self, H: np.ndarray) -> dict[str, Any]:
        """Probe decoder stability for a parity-check matrix.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``bp_stability_score`` : float — composite stability score
            - ``divergence_rate`` : float — fraction of divergent trials
            - ``stagnation_rate`` : float — fraction of stagnant trials
            - ``oscillation_score`` : float — mean oscillation magnitude
            - ``average_iterations`` : float — mean iterations to converge
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        if m == 0 or n == 0:
            return self._empty_result()

        diverged = 0
        stagnated = 0
        oscillation_total = 0.0
        iter_total = 0.0

        for t in range(self.trials):
            trial_seed = _derive_seed(self.seed, f"trial_{t}")
            result = self._run_trial(H_arr, m, n, trial_seed)
            if result["diverged"]:
                diverged += 1
            if result["stagnated"]:
                stagnated += 1
            oscillation_total += result["oscillation"]
            iter_total += result["iterations_used"]

        div_rate = diverged / max(self.trials, 1)
        stag_rate = stagnated / max(self.trials, 1)
        osc_score = oscillation_total / max(self.trials, 1)
        avg_iters = iter_total / max(self.trials, 1)

        # Stability score: (1 - div_rate) * (1 - stag_rate) * exp(-osc_score)
        stability = (1.0 - div_rate) * (1.0 - stag_rate) * math.exp(-osc_score)

        return {
            "bp_stability_score": round(stability, _ROUND),
            "divergence_rate": round(div_rate, _ROUND),
            "stagnation_rate": round(stag_rate, _ROUND),
            "oscillation_score": round(osc_score, _ROUND),
            "average_iterations": round(avg_iters, _ROUND),
        }

    def _run_trial(
        self,
        H: np.ndarray,
        m: int,
        n: int,
        trial_seed: int,
    ) -> dict[str, Any]:
        """Run a single BP probe trial using min-sum message passing.

        This implements a lightweight min-sum BP iteration loop directly,
        avoiding any dependency on the protected decoder core.
        """
        rng = np.random.RandomState(trial_seed)

        # Generate sparse random error pattern (weight ~ 1-3)
        weight = rng.randint(1, min(4, n + 1))
        error = np.zeros(n, dtype=np.float64)
        error_bits = rng.choice(n, size=weight, replace=False)
        error[error_bits] = 1.0

        # Compute syndrome
        syndrome = (H @ error) % 2

        # Initial LLR: channel reliability (moderate SNR)
        channel_llr = np.where(error == 0, 2.0, -2.0)

        # Run abbreviated min-sum BP
        # Messages: check-to-variable (c2v) and variable-to-check (v2c)
        c2v = np.zeros((m, n), dtype=np.float64)
        syndrome_weights = []
        prev_hard = None
        oscillation = 0.0

        for it in range(self.iterations):
            # Variable-to-check messages
            v2c = np.zeros((m, n), dtype=np.float64)
            for vi in range(n):
                total = channel_llr[vi] + np.sum(c2v[:, vi])
                for ci in range(m):
                    if H[ci, vi] != 0:
                        v2c[ci, vi] = total - c2v[ci, vi]

            # Check-to-variable messages (min-sum)
            c2v_new = np.zeros((m, n), dtype=np.float64)
            for ci in range(m):
                connected = [vi for vi in range(n) if H[ci, vi] != 0]
                if len(connected) < 2:
                    continue
                s = 1.0 - 2.0 * syndrome[ci]
                for vi in connected:
                    sign = s
                    min_mag = float("inf")
                    for vj in connected:
                        if vj == vi:
                            continue
                        msg = v2c[ci, vj]
                        if msg < 0:
                            sign *= -1.0
                        mag = abs(msg)
                        if mag < min_mag:
                            min_mag = mag
                    c2v_new[ci, vi] = sign * min(min_mag, 20.0)
            c2v = c2v_new

            # Hard decision
            total_llr = channel_llr + np.sum(c2v, axis=0)
            hard = (total_llr < 0).astype(np.float64)

            # Check syndrome weight
            residual = (H @ hard) % 2
            sw = int(np.sum(np.abs(residual - syndrome)))
            syndrome_weights.append(sw)

            # Track oscillation
            if prev_hard is not None:
                flips = int(np.sum(np.abs(hard - prev_hard)))
                oscillation += flips / max(n, 1)
            prev_hard = hard.copy()

            # Early exit if converged
            if sw == 0:
                return {
                    "diverged": False,
                    "stagnated": False,
                    "oscillation": round(oscillation / max(it + 1, 1), _ROUND),
                    "iterations_used": it + 1,
                }

        # Analyse convergence behaviour
        diverged = False
        stagnated_flag = False

        if len(syndrome_weights) >= 3:
            # Divergence: syndrome weight increased over last 3 iterations
            if syndrome_weights[-1] > syndrome_weights[-3]:
                diverged = True
            # Stagnation: last 3 syndrome weights identical and nonzero
            if (syndrome_weights[-1] == syndrome_weights[-2] ==
                    syndrome_weights[-3] and syndrome_weights[-1] > 0):
                stagnated_flag = True

        avg_osc = oscillation / max(self.iterations, 1)

        return {
            "diverged": diverged,
            "stagnated": stagnated_flag,
            "oscillation": round(avg_osc, _ROUND),
            "iterations_used": self.iterations,
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return default result for empty matrices."""
        return {
            "bp_stability_score": 1.0,
            "divergence_rate": 0.0,
            "stagnation_rate": 0.0,
            "oscillation_score": 0.0,
            "average_iterations": 0.0,
        }


def estimate_bp_instability(H: np.ndarray) -> dict[str, float]:
    """Estimate BP instability via Jacobian spectral radius approximation.

    Uses power iteration on the implicit BP Jacobian operator to estimate
    the spectral radius.  This predicts decoder instability without
    running actual BP decoding.

    Interpretation:
    - rho < 1  : stable fixed point
    - rho ~ 1  : marginal stability
    - rho > 1  : unstable (BP likely to diverge/oscillate)

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, float]
        Dictionary with key ``jacobian_spectral_radius_est``.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if m == 0 or n == 0:
        return {"jacobian_spectral_radius_est": 0.0}

    # Build degree vectors
    var_deg = np.sum(H_arr, axis=0)  # shape (n,)
    check_deg = np.sum(H_arr, axis=1)  # shape (m,)

    # The BP Jacobian spectral radius for min-sum on a bipartite graph
    # is approximated by the spectral radius of D_c^{-1} H^T D_v^{-1} H - I
    # where D_v = diag(var_deg), D_c = diag(check_deg).
    # We approximate this via power iteration on the implicit operator:
    #   T(x) = H^T @ (H @ (x / max(var_deg, 1))) / max(check_deg, 1) - x

    # Safe degree inverses (avoid divide-by-zero warnings)
    safe_var = np.maximum(var_deg - 1, 1.0)
    safe_check = np.maximum(check_deg - 1, 1.0)
    inv_var_deg = np.where(var_deg > 1, 1.0 / safe_var, 0.0)
    inv_check_deg = np.where(check_deg > 1, 1.0 / safe_check, 0.0)

    # Power iteration for dominant eigenvalue
    x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)

    rho = 0.0
    for _ in range(60):
        # Apply operator: T(x) = D_v^{-1/2} H^T D_c^{-1} H D_v^{-1/2} x
        # Simplified: scale by (d_v - 1) and (d_c - 1) factors
        step1 = x * inv_var_deg  # scale by 1/(d_v - 1)
        step2 = H_arr @ step1  # project to check space
        step3 = step2 * inv_check_deg  # scale by 1/(d_c - 1)
        y = H_arr.T @ step3  # project back to variable space

        norm_y = np.linalg.norm(y)
        if norm_y < 1e-15:
            return {"jacobian_spectral_radius_est": 0.0}

        # Rayleigh quotient
        rho = float(np.dot(x, y))
        x = y / norm_y

    return {"jacobian_spectral_radius_est": round(abs(rho), _ROUND)}
