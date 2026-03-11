"""
v9.4.0 — Decoder Benchmark Visualization.

Plots spectral_radius vs success_rate and spectral_radius vs iterations
from a decoder benchmark artifact.

Falls back to ASCII summary if matplotlib is unavailable.
"""

from __future__ import annotations

import json
import sys


def _load_benchmark(path: str) -> list[dict]:
    """Load benchmark results from JSON artifact."""
    with open(path) as f:
        data = json.load(f)
    return data["benchmark_results"]


def _ascii_summary(results: list[dict]) -> None:
    """Print ASCII summary table."""
    print(f"{'graph_id':<20} {'spectral_radius':>16} "
          f"{'bp_success_rate':>16} {'avg_iterations':>15}")
    print("-" * 70)
    for entry in results:
        print(f"{entry['graph_id']:<20} {entry['spectral_radius']:>16.4f} "
              f"{entry['bp_success_rate']:>16.4f} "
              f"{entry['avg_iterations']:>15.2f}")


def _plot_matplotlib(results: list[dict], output_prefix: str) -> None:
    """Create matplotlib scatter plots."""
    import matplotlib.pyplot as plt

    radii = [r["spectral_radius"] for r in results]
    success = [r["bp_success_rate"] for r in results]
    iters = [r["avg_iterations"] for r in results]

    # Plot 1: spectral_radius vs success_rate
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(radii, success, marker="o", edgecolors="black", alpha=0.7)
    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("BP Success Rate")
    ax.set_title("Spectral Radius vs BP Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_success.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_prefix}_success.png")

    # Plot 2: spectral_radius vs iterations
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(radii, iters, marker="s", edgecolors="black", alpha=0.7)
    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("Average Iterations")
    ax.set_title("Spectral Radius vs Average Iterations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_iterations.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_prefix}_iterations.png")


def main() -> None:
    """Entry point for benchmark visualization."""
    artifact_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "artifacts/decoder_benchmark.json"
    )
    output_prefix = (
        sys.argv[2] if len(sys.argv) > 2
        else "artifacts/decoder_benchmark"
    )

    results = _load_benchmark(artifact_path)

    if not results:
        print("No benchmark results found.")
        return

    # Always print ASCII summary
    _ascii_summary(results)

    # Attempt matplotlib plots
    try:
        _plot_matplotlib(results, output_prefix)
    except ImportError:
        print("\nmatplotlib not available; skipping plots.")


if __name__ == "__main__":
    main()
