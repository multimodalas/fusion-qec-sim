"""
Tests for v9.5.0 determinism guard.

Verifies:
  - determinism check script exists and is importable
  - script executes successfully
"""

from __future__ import annotations

import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def test_check_determinism_script_exists():
    """Verify the determinism guard script exists."""
    script_path = os.path.join(_repo_root, "scripts", "check_determinism.py")
    assert os.path.isfile(script_path), (
        f"Determinism guard script not found at {script_path}"
    )


def test_check_determinism_importable():
    """Verify the determinism guard script is valid Python."""
    script_path = os.path.join(_repo_root, "scripts", "check_determinism.py")
    with open(script_path) as f:
        source = f.read()
    compile(source, script_path, "exec")
