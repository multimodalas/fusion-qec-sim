#!/usr/bin/env python3
"""
v8.4.0 — Generation Trajectory Visualization.

Loads a generation trajectory artifact and plots metric trajectories.
Falls back to ASCII summaries if matplotlib is unavailable.
"""

from __future__ import annotations

import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


_DEFAULT_PATH = "artifacts/generation_trajectory.json"

_METRICS_TO_PLOT = [
    "instability_score",
    "spectral_radius",
    "entropy",
    "bethe_margin",
]


def _load_trajectory(path: str) -> list[dict]:
    """Load trajectory from JSON file."""
    with open(path) as f:
        return json.load(f)


def _extract_series(trajectory: list[dict], metric: str) -> list[float]:
    """Extract a metric series from trajectory records."""
    values = []
    for record in trajectory:
        metrics = record.get("best_metrics", record)
        values.append(float(metrics.get(metric, 0.0)))
    return values


def _plot_matplotlib(trajectory: list[dict], output_dir: str) -> None:
    """Plot trajectory metrics using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in trajectory]

    for metric in _METRICS_TO_PLOT:
        values = _extract_series(trajectory, metric)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, values, marker="o", linewidth=1.5, markersize=4)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_title(f"Generation Trajectory: {metric}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"trajectory_{metric}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def _print_ascii(trajectory: list[dict]) -> None:
    """Print ASCII summary of trajectory metrics."""
    header = f"{'Step':>5}"
    for metric in _METRICS_TO_PLOT:
        header += f"  {metric:>20}"
    print(header)
    print("-" * len(header))

    for record in trajectory:
        metrics = record.get("best_metrics", record)
        line = f"{record['step']:5d}"
        for metric in _METRICS_TO_PLOT:
            val = metrics.get(metric, 0.0)
            line += f"  {val:20.8f}"
        print(line)


def main() -> None:
    """Load trajectory and visualize."""
    path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PATH

    if not os.path.exists(path):
        print(f"Error: artifact not found at {path}")
        sys.exit(1)

    trajectory = _load_trajectory(path)
    print(f"Loaded trajectory with {len(trajectory)} steps from {path}")

    try:
        import matplotlib  # noqa: F401
        output_dir = os.path.dirname(path) or "."
        print("Plotting with matplotlib...")
        _plot_matplotlib(trajectory, output_dir)
    except ImportError:
        print("matplotlib not available, using ASCII summary:\n")
        _print_ascii(trajectory)


if __name__ == "__main__":
    main()
