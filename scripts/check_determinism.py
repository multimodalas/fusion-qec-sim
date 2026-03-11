"""
Determinism guard — runs the test suite multiple times to detect
nondeterministic behaviour.

Usage:
    python scripts/check_determinism.py

Exit code 0 means all runs passed identically.
Any non-zero exit code signals potential nondeterminism.
"""

import subprocess
import sys


def run_tests():
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--count=3", "-q"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    ).returncode


if __name__ == "__main__":
    sys.exit(run_tests())
