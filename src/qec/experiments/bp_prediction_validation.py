"""
v6.9.0 — BP Prediction Validation Experiment.

Validates the spectral BP stability predictor (v6.8) by comparing
predicted BP instability against actual decoder outcomes across the
evaluation pipeline.

Computes confusion matrix metrics (accuracy, precision, recall,
false-positive rate, true-positive rate) and prediction error rate
for phase-diagram visualization.

Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any


def run_bp_prediction_validation(
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate BP stability predictions against actual decoder outcomes.

    For each trial, compares the spectral predictor's
    ``predicted_instability`` flag against actual decoding failure
    (``decoder_success == False``) and builds a confusion matrix.

    Parameters
    ----------
    trial_results : list[dict[str, Any]]
        List of per-trial result dicts, each containing:

        - ``bp_stability_prediction`` : dict with at least
          ``"predicted_instability"`` (bool)
        - ``decoder_success`` : bool

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``prediction_accuracy`` : float — (TP + TN) / N
        - ``precision`` : float — TP / (TP + FP), 0.0 if no positives
        - ``recall`` : float — TP / (TP + FN), 0.0 if no actual failures
        - ``false_positive_rate`` : float — FP / (FP + TN)
        - ``true_positive_rate`` : float — TP / (TP + FN) (same as recall)
        - ``prediction_error_rate`` : float — (FP + FN) / N
        - ``confusion_matrix`` : dict with TP, FP, TN, FN counts
        - ``num_trials`` : int
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for trial in trial_results:
        bp_prediction = trial.get("bp_stability_prediction", {})
        predicted_failure = bp_prediction.get("predicted_instability", False)
        decoder_success = trial.get("decoder_success", True)
        actual_failure = not decoder_success

        if predicted_failure and actual_failure:
            tp += 1
        elif predicted_failure and not actual_failure:
            fp += 1
        elif not predicted_failure and not actual_failure:
            tn += 1
        else:
            fn += 1

    n = tp + fp + tn + fn

    prediction_accuracy = round((tp + tn) / n, 12) if n > 0 else 0.0
    precision = round(tp / (tp + fp), 12) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 12) if (tp + fn) > 0 else 0.0
    false_positive_rate = round(fp / (fp + tn), 12) if (fp + tn) > 0 else 0.0
    true_positive_rate = round(tp / (tp + fn), 12) if (tp + fn) > 0 else 0.0
    prediction_error_rate = round((fp + fn) / n, 12) if n > 0 else 0.0

    return {
        "prediction_accuracy": prediction_accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "prediction_error_rate": prediction_error_rate,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "num_trials": n,
    }
