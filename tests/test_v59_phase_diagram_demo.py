"""
Tests for the v5.9 phase diagram demo script.

Validates:
  - Demo runs cleanly and produces valid output
  - Output is deterministic across repeated runs
  - JSON serialization is stable
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest

# Ensure imports resolve from repository root.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.run_v59_phase_diagram_demo import run_phase_diagram_demo


class TestPhaseDiagramDemo:
    def test_runs_cleanly(self):
        """Demo produces a valid result dict."""
        result = run_phase_diagram_demo()
        assert "phase_diagram" in result
        assert "boundary_analysis" in result

        pd = result["phase_diagram"]
        assert "grid_axes" in pd
        assert "cells" in pd
        assert len(pd["cells"]) > 0

        ba = result["boundary_analysis"]
        assert "boundary_cells" in ba
        assert "mixed_region_cells" in ba
        assert "critical_cells" in ba
        assert "boundary_summary" in ba

    def test_correct_grid_dimensions(self):
        """Grid has expected number of cells."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]
        axes = pd["grid_axes"]
        expected_cells = len(axes["x_values"]) * len(axes["y_values"])
        assert len(pd["cells"]) == expected_cells

    def test_valid_phases(self):
        """All dominant_phase values are in {-1, 0, 1}."""
        result = run_phase_diagram_demo()
        for cell in result["phase_diagram"]["cells"]:
            assert cell["dominant_phase"] in {-1, 0, 1}

    def test_fractions_sum_to_one(self):
        """Phase fractions sum to 1.0 for each cell."""
        result = run_phase_diagram_demo()
        for cell in result["phase_diagram"]["cells"]:
            total = (cell["success_fraction"] +
                     cell["boundary_fraction"] +
                     cell["failure_fraction"])
            assert abs(total - 1.0) < 1e-10

    def test_determinism(self):
        """Repeated runs produce identical output."""
        r1 = run_phase_diagram_demo()
        r2 = run_phase_diagram_demo()
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_roundtrip(self):
        """Output survives JSON serialization."""
        result = run_phase_diagram_demo()
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        # Re-serialize both to compare.
        assert json.dumps(deserialized, sort_keys=True) == json.dumps(result, sort_keys=True)
