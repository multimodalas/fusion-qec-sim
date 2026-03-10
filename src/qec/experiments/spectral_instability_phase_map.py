"""
v7.2.0 — Spectral Instability Phase Map.

Predicts the probability of BP decoding failure using only Tanner graph
spectral diagnostics, before running belief propagation.

Builds a deterministic structural predictor that estimates decoder
instability directly from graph properties, then compares predictions
against observed decoding outcomes to produce phase-map metrics.

Consumes:
  - v6.0 NB spectral radius
  - v6.1 IPR localization scores
  - v6.4 cluster risk scores
  - v6.8 BP stability predictor signals
  - Graph degree statistics

Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any


# ── Deterministic weight constants ────────────────────────────────────

_W1_SPECTRAL_RADIUS = 0.35
_W2_IPR_LOCALIZATION = 0.25
_W3_CLUSTER_RISK = 0.25
_W4_INSTABILITY_RATIO = 0.15

_DEFAULT_INSTABILITY_THRESHOLD = 0.5


# ── Core Feature 1: Phase Map Predictor ───────────────────────────────

def compute_spectral_instability_score(
    nb_spectral_radius: float,
    spectral_instability_ratio: float,
    ipr_localization_score: float,
    cluster_risk_scores: list[float],
    avg_variable_degree: float,
    avg_check_degree: float,
    *,
    w1: float = _W1_SPECTRAL_RADIUS,
    w2: float = _W2_IPR_LOCALIZATION,
    w3: float = _W3_CLUSTER_RISK,
    w4: float = _W4_INSTABILITY_RATIO,
) -> float:
    """Compute spectral instability score in [0, 1].

    Estimates decoding instability probability from Tanner graph
    spectral diagnostics.  Pure function of inputs — fully deterministic.

    Parameters
    ----------
    nb_spectral_radius : float
        Non-backtracking spectral radius from v6.0 diagnostics.
    spectral_instability_ratio : float
        Ratio of spectral radius to instability threshold (from v6.8).
    ipr_localization_score : float
        IPR-based localization score (e.g. max IPR from v6.1).
    cluster_risk_scores : list[float]
        Per-cluster risk scores from v6.4 spectral failure risk.
    avg_variable_degree : float
        Average variable-node degree in the Tanner graph.
    avg_check_degree : float
        Average check-node degree in the Tanner graph.

    Returns
    -------
    float
        Spectral instability score clamped to [0, 1], rounded to 12
        decimal places.
    """
    # Normalize spectral radius: use degree-based normalization.
    avg_degree = (avg_variable_degree + avg_check_degree) / 2.0
    if avg_degree > 0.0:
        normalized_spectral_radius = nb_spectral_radius / (avg_degree + 1.0)
    else:
        normalized_spectral_radius = 0.0

    # Clamp normalized spectral radius to [0, 1].
    normalized_spectral_radius = max(0.0, min(1.0, normalized_spectral_radius))

    # IPR localization: already expected in [0, 1].
    ipr_localization = max(0.0, min(1.0, ipr_localization_score))

    # Max cluster risk.
    max_cluster_risk = max(cluster_risk_scores) if cluster_risk_scores else 0.0
    max_cluster_risk = max(0.0, min(1.0, max_cluster_risk))

    # Instability ratio: normalize to [0, 1] via sigmoid-like clamping.
    # Ratio > 1 indicates instability; map [0, 2] -> [0, 1].
    instability_ratio = max(0.0, min(1.0, spectral_instability_ratio / 2.0))

    score = (
        w1 * normalized_spectral_radius
        + w2 * ipr_localization
        + w3 * max_cluster_risk
        + w4 * instability_ratio
    )

    # Clamp final score to [0, 1].
    score = max(0.0, min(1.0, score))

    return round(score, 12)


# ── Core Feature 2: Phase Map Experiment ──────────────────────────────

def run_spectral_phase_map_experiment(
    spectral_instability_score: float,
    decoder_success: bool,
    *,
    instability_threshold: float = _DEFAULT_INSTABILITY_THRESHOLD,
) -> dict[str, Any]:
    """Run a single spectral phase-map trial comparison.

    Compares the spectral instability prediction against the observed
    decoding outcome for one trial.

    Parameters
    ----------
    spectral_instability_score : float
        Pre-computed instability score from
        :func:`compute_spectral_instability_score`.
    decoder_success : bool
        Whether the baseline decode succeeded for this trial.
    instability_threshold : float
        Threshold above which the predictor predicts failure.
        Default 0.5.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``spectral_instability_score`` : float
        - ``predicted_instability`` : bool
        - ``observed_failure`` : bool
        - ``prediction_correct`` : bool

        All floats rounded to 12 decimal places.
    """
    predicted_instability = spectral_instability_score > instability_threshold
    observed_failure = not decoder_success
    prediction_correct = predicted_instability == observed_failure

    return {
        "spectral_instability_score": round(spectral_instability_score, 12),
        "predicted_instability": predicted_instability,
        "observed_failure": observed_failure,
        "prediction_correct": prediction_correct,
    }


# ── Core Feature 3: Aggregated Phase Map Metrics ─────────────────────

def compute_phase_map_aggregate_metrics(
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate phase-map trial results into summary metrics.

    Computes prediction accuracy, false-positive / false-negative rates,
    and mean instability scores across all trials.

    Parameters
    ----------
    trial_results : list[dict[str, Any]]
        List of per-trial results from
        :func:`run_spectral_phase_map_experiment`.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``mean_spectral_instability_score`` : float
        - ``predicted_failure_fraction`` : float
        - ``observed_failure_fraction`` : float
        - ``prediction_accuracy`` : float
        - ``false_positive_rate`` : float
        - ``false_negative_rate`` : float
        - ``confusion_matrix`` : dict with TP, FP, TN, FN counts
        - ``num_trials`` : int

        All floats rounded to 12 decimal places.
    """
    n = len(trial_results)
    if n == 0:
        return {
            "mean_spectral_instability_score": 0.0,
            "predicted_failure_fraction": 0.0,
            "observed_failure_fraction": 0.0,
            "prediction_accuracy": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "confusion_matrix": {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
            },
            "num_trials": 0,
        }

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    score_sum = 0.0
    predicted_count = 0
    observed_count = 0

    for trial in trial_results:
        score_sum += trial["spectral_instability_score"]
        predicted = trial["predicted_instability"]
        observed = trial["observed_failure"]

        if predicted:
            predicted_count += 1
        if observed:
            observed_count += 1

        if predicted and observed:
            tp += 1
        elif predicted and not observed:
            fp += 1
        elif not predicted and not observed:
            tn += 1
        else:
            fn += 1

    mean_score = round(score_sum / n, 12)
    predicted_fraction = round(predicted_count / n, 12)
    observed_fraction = round(observed_count / n, 12)
    accuracy = round((tp + tn) / n, 12)
    fpr = round(fp / (fp + tn), 12) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (fn + tp), 12) if (fn + tp) > 0 else 0.0

    return {
        "mean_spectral_instability_score": mean_score,
        "predicted_failure_fraction": predicted_fraction,
        "observed_failure_fraction": observed_fraction,
        "prediction_accuracy": accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "num_trials": n,
    }
