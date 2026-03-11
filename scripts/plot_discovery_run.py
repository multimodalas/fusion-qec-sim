"""
v9.2.0 — Plot Discovery Run.

Loads a discovery run artifact and prints generation-level summary
statistics.  Optional matplotlib plotting when available.

Supports both v9.0.0 (flat) and v9.2.0 (metadata + results) formats.

Usage:
    python scripts/plot_discovery_run.py [artifact_path]
"""

from __future__ import annotations

import json
import sys


def _get_results(data: dict) -> dict:
    """Extract results from artifact, supporting both old and new formats."""
    if "results" in data:
        return data["results"]
    return data


def print_discovery_summary(artifact_path: str = "artifacts/discovery_run.json") -> None:
    """Print a text summary of a discovery run artifact."""
    with open(artifact_path) as f:
        data = json.load(f)

    results = _get_results(data)

    config = results.get("config", {})
    print(f"Discovery Run Summary")
    print(f"  Generations: {config.get('num_generations', '?')}")
    print(f"  Population:  {config.get('population_size', '?')}")
    print(f"  Seed:        {config.get('base_seed', '?')}")
    print()

    summaries = results.get("generation_summaries", [])
    print(f"{'Gen':>4}  {'Composite':>12}  {'Instability':>12}  {'Radius':>10}  {'Archive':>8}  {'Feasible':>8}  {'Novel':>6}")
    print("-" * 75)
    for s in summaries:
        print(
            f"{s['generation']:4d}  "
            f"{s['best_composite_score']:12.6f}  "
            f"{s['best_instability_score']:12.6f}  "
            f"{s['best_spectral_radius']:10.6f}  "
            f"{s['archive_size']:8d}  "
            f"{s['num_feasible']:8d}  "
            f"{s['num_novel']:6d}"
        )

    best = results.get("best_candidate", {})
    if best:
        print()
        print(f"Best candidate: {best.get('candidate_id', '?')}")
        obj = best.get("objectives", {})
        for k, v in sorted(obj.items()):
            print(f"  {k}: {v}")


def plot_discovery_run(artifact_path: str = "artifacts/discovery_run.json") -> None:
    """Plot discovery run if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    with open(artifact_path) as f:
        data = json.load(f)

    results = _get_results(data)

    summaries = results.get("generation_summaries", [])
    gens = [s["generation"] for s in summaries]
    composites = [s["best_composite_score"] for s in summaries]
    instabilities = [s["best_instability_score"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(gens, composites, "b-o", markersize=3)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Composite Score")
    ax1.set_title("Composite Score vs Generation")
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, instabilities, "r-o", markersize=3)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Instability Score")
    ax2.set_title("Instability Score vs Generation")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("artifacts/discovery_run_plot.png", dpi=150)
    print("Plot saved to artifacts/discovery_run_plot.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/discovery_run.json"
    print_discovery_summary(path)
    plot_discovery_run(path)
