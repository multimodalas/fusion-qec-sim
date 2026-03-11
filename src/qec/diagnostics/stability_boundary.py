"""
v8.2.0 — Stability Boundary Estimator.

Fits a deterministic linear boundary separating BP-converged from
BP-failed observations using simple gradient descent on a logistic
loss function.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12

_FEATURE_KEYS = [
    "spectral_radius",
    "entropy",
    "curvature",
    "cycle_density",
    "bethe_margin",
]


def estimate_stability_boundary(
    dataset: list[dict[str, Any]],
    *,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
) -> dict[str, Any]:
    """Fit a deterministic linear stability boundary.

    Computes weights for:

        score = w1*spectral_radius + w2*entropy + w3*curvature
                + w4*cycle_density + w5*bethe_margin + bias

    Uses gradient descent on logistic loss.

    Parameters
    ----------
    dataset : list[dict]
        Observations with feature keys and ``bp_converged`` (0 or 1).
    learning_rate : float
        Step size for gradient descent.
    num_steps : int
        Number of gradient descent steps.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``weights``, ``bias``, and ``accuracy``.
    """
    if not dataset:
        return {
            "weights": [0.0] * len(_FEATURE_KEYS),
            "bias": 0.0,
            "accuracy": 0.0,
        }

    n = len(dataset)
    num_features = len(_FEATURE_KEYS)

    # Extract features and labels
    X = np.zeros((n, num_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for i, obs in enumerate(dataset):
        for j, key in enumerate(_FEATURE_KEYS):
            X[i, j] = obs.get(key, 0.0)
        y[i] = float(obs.get("bp_converged", 0))

    # Initialize weights to zero (deterministic)
    w = np.zeros(num_features, dtype=np.float64)
    b = 0.0

    # Gradient descent on logistic loss
    for _ in range(num_steps):
        scores = X @ w + b
        # Sigmoid with clipping for numerical stability
        scores_clipped = np.clip(scores, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-scores_clipped))

        # Gradient
        error = probs - y
        grad_w = (X.T @ error) / n
        grad_b = error.mean()

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    # Compute accuracy
    predictions = (X @ w + b) > 0.0
    correct = np.sum(predictions == (y > 0.5))
    accuracy = float(correct) / n if n > 0 else 0.0

    return {
        "weights": [round(float(wi), _ROUND) for wi in w],
        "bias": round(float(b), _ROUND),
        "accuracy": round(accuracy, _ROUND),
    }
