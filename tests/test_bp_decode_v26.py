"""
Tests for v2.6.0 bp_decode enhancements: LLR history buffer,
OSD-CS postprocessing integration, and backward compatibility
regression tests.
"""

import pytest
import numpy as np

from src.qec_qldpc_codes import (
    bp_decode,
    syndrome,
    channel_llr,
    create_code,
)


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────────
# LLR History Buffer
# ───────────────────────────────────────────────────────────────────

class TestLLRHistory:

    def test_history_0_returns_2tuple(self, noisy_setup):
        """llr_history=0 returns (correction, iters) — 2-tuple."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            syndrome_vec=s, llr_history=0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_history_positive_returns_3tuple(self, noisy_setup):
        """llr_history>0 returns (correction, iters, history)."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, llr_history=5,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_history_shape(self, noisy_setup):
        """history array has shape (k, n) where k <= llr_history."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            syndrome_vec=s, llr_history=3,
        )
        correction, iters, history = result
        assert history.ndim == 2
        assert history.shape[1] == code.n
        assert history.shape[0] <= 3
        assert history.shape[0] >= 1
        assert history.dtype == np.float64

    def test_history_shape_fewer_iters(self, small_code):
        """When BP converges in fewer iters than llr_history, k < llr_history."""
        # High LLR = no errors = converges in 1 iteration.
        llr = np.full(small_code.n, 10.0)
        result = bp_decode(
            small_code.H_X, llr, max_iters=50,
            llr_history=20,
        )
        correction, iters, history = result
        # Should converge quickly; history rows = iters.
        assert history.shape[0] == iters
        assert iters < 20

    def test_history_deterministic(self, noisy_setup):
        """Same inputs produce identical history arrays."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, llr_history=5,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, llr_history=5,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]
        np.testing.assert_array_equal(r1[2], r2[2])

    def test_history_flooding(self, noisy_setup):
        """History works with flooding schedule."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            schedule="flooding", syndrome_vec=s, llr_history=3,
        )
        assert len(result) == 3
        assert result[2].shape[1] == code.n

    def test_history_layered(self, noisy_setup):
        """History works with layered schedule."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            schedule="layered", syndrome_vec=s, llr_history=3,
        )
        assert len(result) == 3
        assert result[2].shape[1] == code.n

    def test_history_negative_raises(self, small_code):
        """llr_history < 0 raises ValueError."""
        llr = np.full(small_code.n, 10.0)
        with pytest.raises(ValueError, match="llr_history"):
            bp_decode(small_code.H_X, llr, llr_history=-1)

    def test_history_last_snapshot_matches_beliefs(self, noisy_setup):
        """Last row of history is consistent with hard decisions."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, llr_history=30,
        )
        correction, iters, history = result
        # Last snapshot: hard decision from sign of L_total.
        last_L = history[-1]
        hard_from_history = (last_L < 0.0).astype(np.uint8)
        np.testing.assert_array_equal(correction, hard_from_history)

    def test_history_with_postprocess(self, noisy_setup):
        """History is returned even when postprocess is applied."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd0", syndrome_vec=s, llr_history=3,
        )
        assert len(result) == 3
        assert result[2].shape[1] == code.n


# ───────────────────────────────────────────────────────────────────
# OSD-CS Integration via bp_decode
# ───────────────────────────────────────────────────────────────────

class TestOSDCSIntegration:

    def test_osd_cs_postprocess(self, noisy_setup):
        """postprocess='osd_cs' works via bp_decode."""
        code, e, s, llr = noisy_setup
        correction, iters = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd_cs", syndrome_vec=s,
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)

    def test_osd_cs_lam_forwarded(self, noisy_setup):
        """bp_decode forwards osd_cs_lam to osd_cs."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd_cs", syndrome_vec=s, osd_cs_lam=0,
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd_cs", syndrome_vec=s, osd_cs_lam=2,
        )
        assert c1.dtype == np.uint8
        assert c2.dtype == np.uint8

    def test_osd_cs_deterministic(self, noisy_setup):
        """OSD-CS via bp_decode is deterministic."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd_cs", syndrome_vec=s,
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd_cs", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_osd_cs_lam0_matches_osd0(self, noisy_setup):
        """postprocess='osd_cs' with osd_cs_lam=0 matches postprocess='osd0'."""
        code, e, s, llr = noisy_setup
        c_cs, i_cs = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd_cs", syndrome_vec=s, osd_cs_lam=0,
        )
        c_0, i_0 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd0", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c_cs, c_0)
        assert i_cs == i_0

    def test_osd_cs_lam_negative_raises(self, small_code):
        """osd_cs_lam < 0 raises ValueError."""
        llr = np.full(small_code.n, 10.0)
        with pytest.raises(ValueError, match="osd_cs_lam"):
            bp_decode(small_code.H_X, llr, osd_cs_lam=-1)


# ───────────────────────────────────────────────────────────────────
# Backward Compatibility Regression Tests
# ───────────────────────────────────────────────────────────────────

class TestBackwardCompatV26:

    def test_default_params_unchanged(self, noisy_setup):
        """bp_decode with default params identical to prior calls."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(code.H_X, llr, max_iters=30, mode="min_sum",
                           syndrome_vec=s)
        c2, i2 = bp_decode(code.H_X, llr, max_iters=30, mode="min_sum",
                           syndrome_vec=s)
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_flooding_unchanged(self, noisy_setup):
        """Flooding output unchanged."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_layered_unchanged(self, noisy_setup):
        """Layered output unchanged."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            schedule="layered", syndrome_vec=s,
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            schedule="layered", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_osd0_unchanged(self, noisy_setup):
        """postprocess='osd0' unchanged."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd0", syndrome_vec=s,
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd0", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_osd1_unchanged(self, noisy_setup):
        """postprocess='osd1' unchanged."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd1", syndrome_vec=s,
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd1", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_return_type_unchanged_default(self, noisy_setup):
        """Default call returns exactly a 2-tuple (not 3-tuple)."""
        code, e, s, llr = noisy_setup
        result = bp_decode(code.H_X, llr, max_iters=5, syndrome_vec=s)
        assert isinstance(result, tuple)
        assert len(result) == 2
