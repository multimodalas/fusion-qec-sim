"""
v6.0.0 — BP Stability Proxy Diagnostics.

Computes a deterministic stability proxy for belief propagation
decoding based on the non-backtracking spectral radius and Bethe
Hessian minimum eigenvalue.

The stability score combines two spectral indicators:

    bp_stability_score = (1 / spectral_radius) * min_eigenvalue
                         if spectral_radius > 0
                         else 0.0

Interpretation:
- Large positive score → stable BP regime (small spectral radius,
  positive Bethe Hessian minimum).
- Score near zero → marginal stability.
- Negative score → likely unstable (negative Bethe Hessian eigenvalue
  indicates structural degeneracy).

This is an observational spectral proxy, not a modification to the
decoder.  Does not run BP decoding.  Does not modify decoder internals.
Fully deterministic.
"""

from __future__ import annotations

from typing import Any


def estimate_bp_stability(
    nb_result: dict[str, Any],
    bethe_result: dict[str, Any],
) -> dict[str, Any]:
    """Estimate BP stability from spectral diagnostics.

    Combines the non-backtracking spectral radius with the Bethe
    Hessian minimum eigenvalue into a single stability proxy.

    Formula:

        bp_stability_score = (1 / spectral_radius) * min_eigenvalue

    A high positive score suggests a stable BP regime.  A negative
    score suggests structural instability.

    Parameters
    ----------
    nb_result : dict
        Output from ``compute_non_backtracking_spectrum()``.
        Must contain ``"spectral_radius"``.
    bethe_result : dict
        Output from ``compute_bethe_hessian()``.
        Must contain ``"min_eigenvalue"`` and ``"num_negative"``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:
        - ``bp_stability_score``: float, the stability proxy
        - ``spectral_radius``: float, NB spectral radius
        - ``bethe_min_eigenvalue``: float, Bethe Hessian minimum
        - ``num_negative_bethe``: int, count of negative Bethe eigenvalues
    """
    spectral_radius = float(nb_result["spectral_radius"])
    min_eigenvalue = float(bethe_result["min_eigenvalue"])
    num_negative = int(bethe_result["num_negative"])

    if spectral_radius > 0.0:
        bp_stability_score = (1.0 / spectral_radius) * min_eigenvalue
    else:
        bp_stability_score = 0.0

    return {
        "bp_stability_score": bp_stability_score,
        "spectral_radius": spectral_radius,
        "bethe_min_eigenvalue": min_eigenvalue,
        "num_negative_bethe": num_negative,
    }
