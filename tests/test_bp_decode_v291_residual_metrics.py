"""
Tests for v2.9.1: Opt-in residual metric instrumentation.

residual_metrics=True collects per-iteration L-inf, L2, and energy
metrics from the check-to-variable message delta.  Collection occurs
only for residual / hybrid_residual schedules.
"""

import numpy as np
import pytest

from src.qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code


@pytest.fixture
def small_code():
    """Small rate-0.50 code for fast tests."""
    return create_code('rate_0.50', lifting_size=8, seed=42)


@pytest.fixture
def noisy_setup(small_code):
    """Code + low-noise error + syndrome + LLR."""
    rng = np.random.default_rng(42)
    e = (rng.random(small_code.n) < 0.01).astype(np.uint8)
    s = syndrome(small_code.H_X, e)
    llr = channel_llr(e, 0.01)
    return small_code, e, s, llr


class TestResidualMetricsReturnSignature:

    def test_disabled_returns_two_tuple(self, noisy_setup):
        """residual_metrics=False → return signature identical to v2.9.0."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="residual",
            residual_metrics=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_enabled_residual_returns_three_tuple(self, noisy_setup):
        """residual_metrics=True + residual schedule → 3-tuple with metrics."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="residual",
            residual_metrics=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        metrics = result[2]
        assert "residual_linf" in metrics
        assert "residual_l2" in metrics
        assert "residual_energy" in metrics


class TestResidualMetricsCollection:

    def test_residual_schedule_nonempty(self, noisy_setup):
        """residual_metrics=True + residual schedule → non-empty lists."""
        code, e, s, llr = noisy_setup
        _, iters, metrics = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="residual",
            residual_metrics=True,
        )
        assert len(metrics["residual_linf"]) > 0
        assert len(metrics["residual_l2"]) > 0
        assert len(metrics["residual_energy"]) > 0

    def test_hybrid_residual_schedule_nonempty(self, noisy_setup):
        """residual_metrics=True + hybrid_residual → non-empty lists."""
        code, e, s, llr = noisy_setup
        _, iters, metrics = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
            residual_metrics=True,
        )
        assert len(metrics["residual_linf"]) > 0

    def test_flooding_schedule_empty(self, noisy_setup):
        """residual_metrics=True + flooding schedule → empty metric lists."""
        code, e, s, llr = noisy_setup
        _, iters, metrics = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
            residual_metrics=True,
        )
        assert metrics["residual_linf"] == []
        assert metrics["residual_l2"] == []
        assert metrics["residual_energy"] == []


class TestResidualMetricsDeterminism:

    def test_two_runs_identical(self, noisy_setup):
        """Decode twice → metrics must be bit-identical."""
        code, e, s, llr = noisy_setup
        kwargs = dict(
            max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="residual",
            residual_metrics=True,
        )
        _, iters1, m1 = bp_decode(code.H_X, llr, **kwargs)
        _, iters2, m2 = bp_decode(code.H_X, llr, **kwargs)
        assert iters1 == iters2
        assert len(m1["residual_linf"]) == len(m2["residual_linf"])
        for a, b in zip(m1["residual_linf"], m2["residual_linf"]):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(m1["residual_l2"], m2["residual_l2"]):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(m1["residual_energy"], m2["residual_energy"]):
            assert a == b


class TestResidualMetricsInvariants:

    def test_metric_length_equals_iterations(self, noisy_setup):
        """len(metric_list) must equal iterations executed."""
        code, e, s, llr = noisy_setup
        _, iters, metrics = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="residual",
            residual_metrics=True,
        )
        assert len(metrics["residual_linf"]) == iters
        assert len(metrics["residual_l2"]) == iters
        assert len(metrics["residual_energy"]) == iters

    def test_no_mutation_after_return(self, noisy_setup):
        """Returned metrics must not be mutated by subsequent decodes."""
        code, e, s, llr = noisy_setup
        kwargs = dict(
            max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="residual",
            residual_metrics=True,
        )
        _, _, m1 = bp_decode(code.H_X, llr, **kwargs)
        snapshot = [arr.copy() for arr in m1["residual_linf"]]
        # Run a second decode — must not mutate m1.
        _ = bp_decode(code.H_X, llr, **kwargs)
        for orig, snap in zip(m1["residual_linf"], snapshot):
            np.testing.assert_array_equal(orig, snap)
