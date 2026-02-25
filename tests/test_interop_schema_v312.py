"""
Schema v3.1.2 validation tests.

Tests that validate_interop_record enforces all required fields,
enum constraints, and sub-dict structure.
"""

from __future__ import annotations

import copy

import pytest

from src.bench.schema import (
    validate_interop_record,
    SCHEMA_VERSION,
    INTEROP_SCHEMA_VERSION,
    _SUPPORTED_SCHEMA_VERSIONS,
    _SUPPORTED_INTEROP_VERSIONS,
)


def _make_valid_interop_record() -> dict:
    """Create a minimal valid v3.1.2 interop record."""
    return {
        "schema_version": "3.1.2",
        "tool": {
            "name": "qec_bp",
            "version": "native",
            "category": "native",
        },
        "benchmark_kind": "direct_comparison",
        "code_family": "qldpc_css",
        "representation": "pcm",
        "seed": 12345,
        "noise_model": "independent_depolarizing",
        "trials": 200,
        "results": {
            "logical_error_rate": 0.05,
        },
        "determinism": {
            "canonical_json": {
                "sort_keys": True,
                "separators": [",", ":"],
            },
            "stable_sweep_hash": "abc123",
            "artifact_hash": "def456",
        },
    }


class TestSchemaVersionConstants:
    """Verify schema version constants are correct."""

    def test_core_schema_version_unchanged(self):
        """Core SCHEMA_VERSION must remain 3.0.1."""
        assert SCHEMA_VERSION == "3.0.1"

    def test_interop_schema_version(self):
        """INTEROP_SCHEMA_VERSION must be 3.1.2."""
        assert INTEROP_SCHEMA_VERSION == "3.1.2"

    def test_core_supported_versions(self):
        """Core validator still supports 3.0.0 and 3.0.1."""
        assert _SUPPORTED_SCHEMA_VERSIONS == {"3.0.0", "3.0.1"}

    def test_interop_supported_versions(self):
        """Interop validator supports 3.1.2."""
        assert _SUPPORTED_INTEROP_VERSIONS == {"3.1.2"}


class TestValidInteropRecord:
    """Test that valid records pass validation."""

    def test_valid_direct_comparison(self):
        rec = _make_valid_interop_record()
        validate_interop_record(rec)  # Must not raise

    def test_valid_reference_baseline(self):
        rec = _make_valid_interop_record()
        rec["benchmark_kind"] = "reference_baseline"
        rec["code_family"] = "surface_code"
        rec["representation"] = "stim_circuit"
        rec["tool"] = {
            "name": "stim_pymatching",
            "version": "stim=1.12",
            "category": "third_party",
        }
        validate_interop_record(rec)

    def test_skipped_record_accepted(self):
        rec = {"status": "skipped", "reason": "stim not installed"}
        validate_interop_record(rec)  # Must not raise


class TestMissingRequiredKeys:
    """Test that missing required keys are caught."""

    @pytest.mark.parametrize("key", [
        "schema_version", "tool", "benchmark_kind", "code_family",
        "representation", "seed", "noise_model", "trials", "results",
        "determinism",
    ])
    def test_missing_top_level_key(self, key):
        rec = _make_valid_interop_record()
        del rec[key]
        with pytest.raises(ValueError, match=f"Missing required key.*{key}"):
            validate_interop_record(rec)


class TestInvalidEnumValues:
    """Test that invalid enum values are caught."""

    def test_invalid_benchmark_kind(self):
        rec = _make_valid_interop_record()
        rec["benchmark_kind"] = "invalid_kind"
        with pytest.raises(ValueError, match="Invalid benchmark_kind"):
            validate_interop_record(rec)

    def test_invalid_code_family(self):
        rec = _make_valid_interop_record()
        rec["code_family"] = "invalid_family"
        with pytest.raises(ValueError, match="Invalid code_family"):
            validate_interop_record(rec)

    def test_invalid_representation(self):
        rec = _make_valid_interop_record()
        rec["representation"] = "invalid_repr"
        with pytest.raises(ValueError, match="Invalid representation"):
            validate_interop_record(rec)

    def test_invalid_tool_category(self):
        rec = _make_valid_interop_record()
        rec["tool"]["category"] = "invalid_cat"
        with pytest.raises(ValueError, match="Invalid tool category"):
            validate_interop_record(rec)

    def test_invalid_schema_version(self):
        rec = _make_valid_interop_record()
        rec["schema_version"] = "99.99.99"
        with pytest.raises(ValueError, match="Unsupported interop schema_version"):
            validate_interop_record(rec)


class TestSubDictValidation:
    """Test that sub-dict structure is enforced."""

    def test_missing_tool_name(self):
        rec = _make_valid_interop_record()
        del rec["tool"]["name"]
        with pytest.raises(ValueError, match="tool missing required keys"):
            validate_interop_record(rec)

    def test_missing_determinism_artifact_hash(self):
        rec = _make_valid_interop_record()
        del rec["determinism"]["artifact_hash"]
        with pytest.raises(ValueError, match="determinism missing required keys"):
            validate_interop_record(rec)

    def test_missing_results_logical_error_rate(self):
        rec = _make_valid_interop_record()
        del rec["results"]["logical_error_rate"]
        with pytest.raises(ValueError, match="logical_error_rate"):
            validate_interop_record(rec)


class TestOptionalChannelModel:
    """Test the optional channel_model field."""

    def test_record_without_channel_model_passes(self):
        """Interop record without channel_model is valid."""
        rec = _make_valid_interop_record()
        assert "channel_model" not in rec
        validate_interop_record(rec)  # Must not raise

    def test_record_with_channel_model_oracle_passes(self):
        """Interop record with channel_model='oracle' is valid."""
        rec = _make_valid_interop_record()
        rec["channel_model"] = "oracle"
        validate_interop_record(rec)  # Must not raise

    def test_record_with_channel_model_string_passes(self):
        """Any string channel_model is accepted."""
        rec = _make_valid_interop_record()
        rec["channel_model"] = "estimated"
        validate_interop_record(rec)  # Must not raise

    def test_record_with_channel_model_non_string_fails(self):
        """Non-string channel_model must be rejected."""
        rec = _make_valid_interop_record()
        rec["channel_model"] = 42
        with pytest.raises(ValueError, match="channel_model must be str"):
            validate_interop_record(rec)


class TestAllValidCodeFamilies:
    """Test that all valid code families are accepted."""

    @pytest.mark.parametrize("cf", [
        "qldpc_css", "surface_code", "repetition", "other",
    ])
    def test_valid_code_family(self, cf):
        rec = _make_valid_interop_record()
        rec["code_family"] = cf
        validate_interop_record(rec)

    @pytest.mark.parametrize("rep", [
        "pcm", "stim_circuit", "stim_dem", "other",
    ])
    def test_valid_representation(self, rep):
        rec = _make_valid_interop_record()
        rec["representation"] = rep
        validate_interop_record(rec)
