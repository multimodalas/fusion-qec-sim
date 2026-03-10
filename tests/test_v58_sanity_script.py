"""
Tests for v5.8 sanity check script.

Verifies that the sanity experiment script:
  - executes without error
  - produces ternary_trace output
  - returns JSON-serializable results
  - is deterministic (identical on repeated runs)
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _run_script() -> str:
    """Run the sanity check script and return its stdout."""
    result = subprocess.run(
        [sys.executable, "scripts/run_v58_sanity_check.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Script failed with stderr:\n{result.stderr}"
    )
    return result.stdout


class TestSanityScriptExecution:
    """The script must run without error."""

    def test_exit_code_zero(self):
        _run_script()


class TestSanityScriptOutput:
    """Output must contain expected ternary topology fields."""

    @pytest.fixture(scope="class")
    def script_output(self) -> str:
        return _run_script()

    def test_ternary_trace_printed(self, script_output: str):
        assert "ternary_trace:" in script_output

    def test_final_state_printed(self, script_output: str):
        assert "final_state:" in script_output

    def test_boundary_crossings_printed(self, script_output: str):
        assert "boundary_crossings:" in script_output

    def test_regime_switch_count_printed(self, script_output: str):
        assert "regime_switch_count:" in script_output

    def test_metastability_score_printed(self, script_output: str):
        assert "metastability_score:" in script_output

    def test_basin_probe_printed(self, script_output: str):
        assert "basin_probe:" in script_output
        assert "success_fraction:" in script_output
        assert "failure_fraction:" in script_output
        assert "boundary_fraction:" in script_output

    def test_json_serialization_ok(self, script_output: str):
        assert "JSON serialization: OK" in script_output


class TestSanityScriptDeterminism:
    """Output must be identical across repeated runs."""

    def test_deterministic_output(self):
        out1 = _run_script()
        out2 = _run_script()
        assert out1 == out2


class TestSanityScriptJsonRoundtrip:
    """Results from run_sanity_check must be JSON-serializable."""

    def test_json_roundtrip(self):
        # Import and call the function directly.
        from scripts.run_v58_sanity_check import run_sanity_check

        results = run_sanity_check()
        serialized = json.dumps(results)
        deserialized = json.loads(serialized)
        assert len(deserialized) == len(results)
        for r in deserialized:
            assert "ternary_trace" in r
            assert "final_ternary_state" in r
            assert "boundary_crossings" in r
            assert "regime_switch_count" in r
            assert "metastability_score" in r
            assert "basin_probe" in r
