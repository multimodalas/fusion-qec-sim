"""
Tests for spectral instability phase map (v7.2).

Verifies:
  - Deterministic prediction output
  - Score bounds (0 <= score <= 1)
  - Predictor consistency (identical inputs -> identical outputs)
  - JSON serialization stability
  - Pipeline integration (spectral diagnostics -> predictor -> decode -> comparison)
  - Aggregate metrics correctness
  - No decoder import
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.experiments.spectral_instability_phase_map import (
    compute_spectral_instability_score,
    run_spectral_phase_map_experiment,
    compute_phase_map_aggregate_metrics,
)


# ── Test helpers ──────────────────────────────────────────────────────


def _default_score_inputs():
    """Return a default set of inputs for compute_spectral_instability_score."""
    return {
        "nb_spectral_radius": 2.5,
        "spectral_instability_ratio": 1.2,
        "ipr_localization_score": 0.35,
        "cluster_risk_scores": [0.1, 0.4, 0.2],
        "avg_variable_degree": 3.0,
        "avg_check_degree": 4.0,
    }


# ── Core Feature 1: Instability Score ────────────────────────────────


class TestSpectralInstabilityScore:
    """Tests for compute_spectral_instability_score."""

    def test_output_type(self):
        """Output is a float."""
        score = compute_spectral_instability_score(**_default_score_inputs())
        assert isinstance(score, float)

    def test_score_bounds(self):
        """Score is clamped to [0, 1]."""
        score = compute_spectral_instability_score(**_default_score_inputs())
        assert 0.0 <= score <= 1.0

    def test_score_bounds_extreme_high(self):
        """Score does not exceed 1 even with extreme inputs."""
        score = compute_spectral_instability_score(
            nb_spectral_radius=100.0,
            spectral_instability_ratio=10.0,
            ipr_localization_score=1.0,
            cluster_risk_scores=[5.0, 10.0],
            avg_variable_degree=1.0,
            avg_check_degree=1.0,
        )
        assert score <= 1.0

    def test_score_bounds_zero_inputs(self):
        """Score is zero for all-zero inputs."""
        score = compute_spectral_instability_score(
            nb_spectral_radius=0.0,
            spectral_instability_ratio=0.0,
            ipr_localization_score=0.0,
            cluster_risk_scores=[],
            avg_variable_degree=0.0,
            avg_check_degree=0.0,
        )
        assert score == 0.0

    def test_score_bounds_empty_cluster_risks(self):
        """Score handles empty cluster risk scores."""
        score = compute_spectral_instability_score(
            nb_spectral_radius=2.0,
            spectral_instability_ratio=1.0,
            ipr_localization_score=0.3,
            cluster_risk_scores=[],
            avg_variable_degree=3.0,
            avg_check_degree=4.0,
        )
        assert 0.0 <= score <= 1.0

    def test_higher_spectral_radius_higher_score(self):
        """Higher spectral radius produces higher or equal score."""
        inputs_low = _default_score_inputs()
        inputs_low["nb_spectral_radius"] = 1.0
        inputs_high = _default_score_inputs()
        inputs_high["nb_spectral_radius"] = 5.0

        score_low = compute_spectral_instability_score(**inputs_low)
        score_high = compute_spectral_instability_score(**inputs_high)

        assert score_high >= score_low

    def test_higher_ipr_higher_score(self):
        """Higher IPR localization produces higher or equal score."""
        inputs_low = _default_score_inputs()
        inputs_low["ipr_localization_score"] = 0.0
        inputs_high = _default_score_inputs()
        inputs_high["ipr_localization_score"] = 1.0

        score_low = compute_spectral_instability_score(**inputs_low)
        score_high = compute_spectral_instability_score(**inputs_high)

        assert score_high >= score_low

    def test_higher_cluster_risk_higher_score(self):
        """Higher cluster risk produces higher or equal score."""
        inputs_low = _default_score_inputs()
        inputs_low["cluster_risk_scores"] = [0.0]
        inputs_high = _default_score_inputs()
        inputs_high["cluster_risk_scores"] = [0.9]

        score_low = compute_spectral_instability_score(**inputs_low)
        score_high = compute_spectral_instability_score(**inputs_high)

        assert score_high >= score_low

    def test_rounding_precision(self):
        """Output is rounded to 12 decimal places."""
        score = compute_spectral_instability_score(**_default_score_inputs())
        score_str = f"{score:.12f}"
        assert score == float(score_str)


# ── Determinism ───────────────────────────────────────────────────────


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_score_calls_identical(self):
        """Repeated calls produce identical results."""
        inputs = _default_score_inputs()
        s1 = compute_spectral_instability_score(**inputs)
        s2 = compute_spectral_instability_score(**inputs)
        assert s1 == s2

    def test_repeated_experiment_calls_identical(self):
        """Repeated experiment calls produce identical JSON results."""
        r1 = run_spectral_phase_map_experiment(0.65, False)
        r2 = run_spectral_phase_map_experiment(0.65, False)
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_repeated_aggregate_calls_identical(self):
        """Repeated aggregate calls produce identical JSON results."""
        trials = [
            run_spectral_phase_map_experiment(0.3, True),
            run_spectral_phase_map_experiment(0.7, False),
            run_spectral_phase_map_experiment(0.5, True),
        ]
        a1 = compute_phase_map_aggregate_metrics(trials)
        a2 = compute_phase_map_aggregate_metrics(trials)
        j1 = json.dumps(a1, sort_keys=True)
        j2 = json.dumps(a2, sort_keys=True)
        assert j1 == j2


# ── Core Feature 2: Phase Map Experiment ──────────────────────────────


class TestPhaseMapExperiment:
    """Tests for run_spectral_phase_map_experiment."""

    def test_output_keys(self):
        """Output contains all required keys."""
        result = run_spectral_phase_map_experiment(0.6, True)
        expected_keys = {
            "spectral_instability_score",
            "predicted_instability",
            "observed_failure",
            "prediction_correct",
        }
        assert set(result.keys()) == expected_keys

    def test_output_types(self):
        """Output values have correct types."""
        result = run_spectral_phase_map_experiment(0.6, True)
        assert isinstance(result["spectral_instability_score"], float)
        assert isinstance(result["predicted_instability"], bool)
        assert isinstance(result["observed_failure"], bool)
        assert isinstance(result["prediction_correct"], bool)

    def test_prediction_correct_true_positive(self):
        """Predicted instability matches observed failure -> correct."""
        result = run_spectral_phase_map_experiment(0.7, False)
        assert result["predicted_instability"] is True
        assert result["observed_failure"] is True
        assert result["prediction_correct"] is True

    def test_prediction_correct_true_negative(self):
        """No predicted instability, no observed failure -> correct."""
        result = run_spectral_phase_map_experiment(0.3, True)
        assert result["predicted_instability"] is False
        assert result["observed_failure"] is False
        assert result["prediction_correct"] is True

    def test_prediction_incorrect_false_positive(self):
        """Predicted instability but decode succeeded -> incorrect."""
        result = run_spectral_phase_map_experiment(0.7, True)
        assert result["predicted_instability"] is True
        assert result["observed_failure"] is False
        assert result["prediction_correct"] is False

    def test_prediction_incorrect_false_negative(self):
        """No predicted instability but decode failed -> incorrect."""
        result = run_spectral_phase_map_experiment(0.3, False)
        assert result["predicted_instability"] is False
        assert result["observed_failure"] is True
        assert result["prediction_correct"] is False

    def test_custom_threshold(self):
        """Custom threshold changes prediction behavior."""
        # Score 0.6 with threshold 0.5 -> predicted instability
        r1 = run_spectral_phase_map_experiment(0.6, True, instability_threshold=0.5)
        assert r1["predicted_instability"] is True

        # Score 0.6 with threshold 0.7 -> no predicted instability
        r2 = run_spectral_phase_map_experiment(0.6, True, instability_threshold=0.7)
        assert r2["predicted_instability"] is False

    def test_score_rounding(self):
        """Score in output is rounded to 12 decimal places."""
        result = run_spectral_phase_map_experiment(1.0 / 3.0, True)
        score_str = f"{result['spectral_instability_score']:.12f}"
        assert result["spectral_instability_score"] == float(score_str)


# ── Core Feature 3: Aggregate Metrics ─────────────────────────────────


class TestAggregateMetrics:
    """Tests for compute_phase_map_aggregate_metrics."""

    def test_output_keys(self):
        """Output contains all required keys."""
        trials = [run_spectral_phase_map_experiment(0.6, False)]
        result = compute_phase_map_aggregate_metrics(trials)
        expected_keys = {
            "mean_spectral_instability_score",
            "predicted_failure_fraction",
            "observed_failure_fraction",
            "prediction_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "confusion_matrix",
            "num_trials",
        }
        assert set(result.keys()) == expected_keys

    def test_confusion_matrix_keys(self):
        """Confusion matrix contains all required keys."""
        trials = [run_spectral_phase_map_experiment(0.6, False)]
        result = compute_phase_map_aggregate_metrics(trials)
        cm = result["confusion_matrix"]
        expected_keys = {
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        }
        assert set(cm.keys()) == expected_keys

    def test_empty_trials(self):
        """Empty trial list produces zero metrics."""
        result = compute_phase_map_aggregate_metrics([])
        assert result["num_trials"] == 0
        assert result["mean_spectral_instability_score"] == 0.0
        assert result["prediction_accuracy"] == 0.0

    def test_perfect_prediction(self):
        """All-correct predictions yield accuracy 1.0."""
        trials = [
            run_spectral_phase_map_experiment(0.7, False),  # TP
            run_spectral_phase_map_experiment(0.3, True),   # TN
        ]
        result = compute_phase_map_aggregate_metrics(trials)
        assert result["prediction_accuracy"] == 1.0
        assert result["false_positive_rate"] == 0.0
        assert result["false_negative_rate"] == 0.0

    def test_all_false_positives(self):
        """All false positives -> FPR = 1.0."""
        trials = [
            run_spectral_phase_map_experiment(0.7, True),  # FP
            run_spectral_phase_map_experiment(0.8, True),  # FP
        ]
        result = compute_phase_map_aggregate_metrics(trials)
        assert result["false_positive_rate"] == 1.0
        assert result["prediction_accuracy"] == 0.0

    def test_confusion_matrix_counts(self):
        """Confusion matrix counts are correct."""
        trials = [
            run_spectral_phase_map_experiment(0.7, False),  # TP
            run_spectral_phase_map_experiment(0.7, True),   # FP
            run_spectral_phase_map_experiment(0.3, True),   # TN
            run_spectral_phase_map_experiment(0.3, False),  # FN
        ]
        result = compute_phase_map_aggregate_metrics(trials)
        cm = result["confusion_matrix"]
        assert cm["true_positives"] == 1
        assert cm["false_positives"] == 1
        assert cm["true_negatives"] == 1
        assert cm["false_negatives"] == 1
        assert result["num_trials"] == 4
        assert result["prediction_accuracy"] == 0.5

    def test_mean_score_computation(self):
        """Mean instability score is computed correctly."""
        trials = [
            run_spectral_phase_map_experiment(0.2, True),
            run_spectral_phase_map_experiment(0.8, False),
        ]
        result = compute_phase_map_aggregate_metrics(trials)
        assert abs(result["mean_spectral_instability_score"] - 0.5) < 1e-10

    def test_fraction_metrics(self):
        """Predicted and observed fractions are correct."""
        # 2 predicted failures (score > 0.5), 1 observed failure
        trials = [
            run_spectral_phase_map_experiment(0.7, False),  # predicted=T, observed=T
            run_spectral_phase_map_experiment(0.6, True),   # predicted=T, observed=F
            run_spectral_phase_map_experiment(0.3, True),   # predicted=F, observed=F
        ]
        result = compute_phase_map_aggregate_metrics(trials)
        assert abs(result["predicted_failure_fraction"] - 2.0 / 3.0) < 1e-10
        assert abs(result["observed_failure_fraction"] - 1.0 / 3.0) < 1e-10


# ── JSON Stability ────────────────────────────────────────────────────


class TestJsonStability:
    """Tests for JSON serialization stability."""

    def test_score_json_serializable(self):
        """Score is JSON-serializable."""
        score = compute_spectral_instability_score(**_default_score_inputs())
        serialized = json.dumps(score)
        assert json.loads(serialized) == score

    def test_experiment_json_serializable(self):
        """Experiment result is JSON-serializable."""
        result = run_spectral_phase_map_experiment(0.6, True)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_aggregate_json_serializable(self):
        """Aggregate result is JSON-serializable."""
        trials = [
            run_spectral_phase_map_experiment(0.6, False),
            run_spectral_phase_map_experiment(0.3, True),
        ]
        result = compute_phase_map_aggregate_metrics(trials)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Full pipeline result survives JSON roundtrip."""
        score = compute_spectral_instability_score(**_default_score_inputs())
        trial = run_spectral_phase_map_experiment(score, False)
        agg = compute_phase_map_aggregate_metrics([trial])

        serialized = json.dumps(agg)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(agg, sort_keys=True)


# ── Pipeline Integration ──────────────────────────────────────────────


class TestPipelineIntegration:
    """Tests for full pipeline: spectral diagnostics -> predictor -> decode -> comparison."""

    def test_full_pipeline(self):
        """Full phase map pipeline produces valid artifacts."""
        import numpy as np

        from src.qec.diagnostics.non_backtracking_spectrum import (
            compute_non_backtracking_spectrum,
        )
        from src.qec.diagnostics.nb_localization import (
            compute_nb_localization_metrics,
        )
        from src.qec.diagnostics.nb_trapping_candidates import (
            compute_nb_trapping_candidates,
        )
        from src.qec.diagnostics.spectral_bp_alignment import (
            compute_spectral_bp_alignment,
        )
        from src.qec.diagnostics.spectral_failure_risk import (
            compute_spectral_failure_risk,
        )
        from src.qec.diagnostics.bp_stability_predictor import (
            compute_bp_stability_prediction,
        )

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)

        m, n = H.shape
        num_edges = int(np.count_nonzero(H))
        avg_var_deg = num_edges / n if n > 0 else 0.0
        avg_chk_deg = num_edges / m if m > 0 else 0.0

        # 1. Spectral diagnostics.
        nb_spectrum = compute_non_backtracking_spectrum(H)
        loc = compute_nb_localization_metrics(H)
        trapping = compute_nb_trapping_candidates(H, loc)
        bp_scores = {0: 3.0, 1: 7.0, 2: 1.0, 3: 5.0}
        alignment = compute_spectral_bp_alignment(trapping, bp_scores)
        risk = compute_spectral_failure_risk(loc, trapping, alignment)

        # 2. Instability predictor (v6.8).
        avg_degree = (2.0 * num_edges) / (n + m)
        graph = {
            "num_variable_nodes": n,
            "num_check_nodes": m,
            "num_edges": num_edges,
            "avg_degree": round(avg_degree, 12),
        }
        diagnostics = {
            "spectral_radius": nb_spectrum["spectral_radius"],
            "nb_max_ipr": max(loc["ipr_scores"]) if loc["ipr_scores"] else 0.0,
            "nb_num_localized_modes": len(loc["localized_modes"]),
            "max_cluster_risk": risk["max_cluster_risk"],
        }
        bp_pred = compute_bp_stability_prediction(graph, diagnostics)

        # 3. Phase map: compute score.
        nb_max_ipr = max(loc["ipr_scores"]) if loc["ipr_scores"] else 0.0
        score = compute_spectral_instability_score(
            nb_spectral_radius=bp_pred["spectral_radius"],
            spectral_instability_ratio=bp_pred["spectral_instability_ratio"],
            ipr_localization_score=nb_max_ipr,
            cluster_risk_scores=risk["cluster_risk_scores"],
            avg_variable_degree=round(avg_var_deg, 12),
            avg_check_degree=round(avg_chk_deg, 12),
        )

        assert 0.0 <= score <= 1.0

        # 4. Simulate decode outcome.
        decoder_success = True

        # 5. Phase map experiment.
        trial_result = run_spectral_phase_map_experiment(
            spectral_instability_score=score,
            decoder_success=decoder_success,
        )

        assert "spectral_instability_score" in trial_result
        assert "predicted_instability" in trial_result
        assert "observed_failure" in trial_result
        assert "prediction_correct" in trial_result

        # 6. Aggregate.
        aggregate = compute_phase_map_aggregate_metrics([trial_result])

        assert aggregate["num_trials"] == 1
        assert 0.0 <= aggregate["prediction_accuracy"] <= 1.0
        assert json.dumps(aggregate)  # JSON-serializable


# ── No Decoder Import ─────────────────────────────────────────────────


class TestNoDecoderImport:
    """Tests that this module does not import decoder code."""

    def test_no_decoder_import(self):
        """Module does not import any decoder code."""
        import src.qec.experiments.spectral_instability_phase_map as mod
        source = open(mod.__file__).read()
        assert "bp_decode" not in source
        assert "from src.qec.decoder" not in source
