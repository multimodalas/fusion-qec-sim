"""
v8.3.0 — Automatic Spectral Invariant Discovery.

Identifies combinations of spectral metrics that are strongly
correlated with BP failure using deterministic statistical analysis.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


_ROUND = 12

# Candidate feature combinations: (expression, operator, operand_a, operand_b)
_CANDIDATE_FEATURES = [
    ("spectral_radius * entropy", "mul", "spectral_radius", "entropy"),
    ("entropy / spectral_gap", "div", "entropy", "spectral_gap"),
    ("curvature * cycle_density", "mul", "curvature", "cycle_density"),
    ("spectral_radius + curvature", "add", "spectral_radius", "curvature"),
    ("entropy + cycle_density", "add", "entropy", "cycle_density"),
    ("spectral_radius * curvature", "mul", "spectral_radius", "curvature"),
    ("sis * spectral_radius", "mul", "sis", "spectral_radius"),
    ("spectral_radius / entropy", "div", "spectral_radius", "entropy"),
    ("curvature / entropy", "div", "curvature", "entropy"),
    ("spectral_radius * cycle_density", "mul", "spectral_radius", "cycle_density"),
    ("sis + curvature", "add", "sis", "curvature"),
    ("spectral_radius - bethe_margin", "sub", "spectral_radius", "bethe_margin"),
]


def _compute_feature(
    obs: dict[str, Any],
    operator: str,
    operand_a: str,
    operand_b: str,
) -> float:
    """Compute a candidate feature from an observation."""
    a = float(obs.get(operand_a, 0.0))
    b = float(obs.get(operand_b, 0.0))

    if operator == "mul":
        return a * b
    elif operator == "div":
        if abs(b) < 1e-15:
            return 0.0
        return a / b
    elif operator == "add":
        return a + b
    elif operator == "sub":
        return a - b
    return 0.0


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays."""
    n = len(x)
    if n < 2:
        return 0.0

    x_mean = x.mean()
    y_mean = y.mean()

    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = float(np.sum(x_centered * y_centered))
    denom_x = float(np.sqrt(np.sum(x_centered ** 2)))
    denom_y = float(np.sqrt(np.sum(y_centered ** 2)))

    if denom_x < 1e-15 or denom_y < 1e-15:
        return 0.0

    return numerator / (denom_x * denom_y)


def _classification_accuracy(
    feature_values: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute classification accuracy using a simple threshold.

    Uses the median of feature values as the threshold.  Values above
    the threshold predict failure (label=0), values at or below
    predict convergence (label=1).
    """
    n = len(feature_values)
    if n == 0:
        return 0.0

    threshold = float(np.median(feature_values))

    # Higher feature values predict failure (bp_converged=0)
    predictions = (feature_values <= threshold).astype(int)
    correct = int(np.sum(predictions == labels))

    accuracy = correct / n

    # Also try the reversed direction
    predictions_rev = (feature_values > threshold).astype(int)
    correct_rev = int(np.sum(predictions_rev == labels))
    accuracy_rev = correct_rev / n

    return max(accuracy, accuracy_rev)


def discover_spectral_invariants(
    dataset: list[dict[str, Any]],
    *,
    output_path: str = "artifacts/spectral_invariants.json",
) -> list[dict[str, Any]]:
    """Discover spectral invariants correlated with BP failure.

    Procedure:
    1. Load dataset.
    2. Generate candidate features using deterministic combinations.
    3. Compute correlation with BP failure.
    4. Return ranked invariants.

    Parameters
    ----------
    dataset : list[dict[str, Any]]
        Stability dataset with spectral metrics and ``bp_converged``.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    list[dict[str, Any]]
        Ranked invariants with expression, correlation, and accuracy.
    """
    if not dataset:
        return []

    n = len(dataset)
    labels = np.array(
        [float(obs.get("bp_converged", 0)) for obs in dataset],
        dtype=np.float64,
    )

    invariants: list[dict[str, Any]] = []

    for expression, operator, operand_a, operand_b in _CANDIDATE_FEATURES:
        feature_values = np.array(
            [
                _compute_feature(obs, operator, operand_a, operand_b)
                for obs in dataset
            ],
            dtype=np.float64,
        )

        correlation = _pearson_correlation(feature_values, labels)
        accuracy = _classification_accuracy(feature_values, labels.astype(int))

        invariants.append({
            "expression": expression,
            "correlation": round(correlation, _ROUND),
            "accuracy": round(accuracy, _ROUND),
        })

    # Sort by absolute correlation descending, then by expression for stability
    invariants.sort(key=lambda x: (-abs(x["correlation"]), x["expression"]))

    # Save artifact
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(invariants, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return invariants
