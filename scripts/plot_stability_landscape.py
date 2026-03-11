"""
v8.3.0 — Stability Landscape Visualization.

Loads the stability landscape artifact and plots spectral_radius vs SIS
and entropy vs curvature projections, colored by predicted stability.

Falls back to ASCII visualization if matplotlib is unavailable.
"""

from __future__ import annotations

import json
import os
import sys


def _load_landscape(path: str) -> list[dict]:
    """Load stability landscape dataset from JSON."""
    with open(path) as f:
        return json.load(f)


def _ascii_scatter(
    data: list[dict],
    x_key: str,
    y_key: str,
    title: str,
    width: int = 60,
    height: int = 20,
) -> str:
    """Render an ASCII scatter plot."""
    if not data:
        return f"{title}\n(no data)\n"

    x_vals = [d[x_key] for d in data]
    y_vals = [d[y_key] for d in data]
    stable = [d.get("predicted_stable", True) for d in data]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for x, y, s in zip(x_vals, y_vals, stable):
        col = min(int((x - x_min) / x_range * (width - 1)), width - 1)
        row = min(int((y_max - y) / y_range * (height - 1)), height - 1)
        grid[row][col] = "+" if s else "x"

    lines = [title, "=" * len(title)]
    lines.append(f"  {y_key}")
    for row_idx, row in enumerate(grid):
        if row_idx == 0:
            label = f"{y_max:.3f}"
        elif row_idx == height - 1:
            label = f"{y_min:.3f}"
        else:
            label = "      "
        lines.append(f"{label:>8} |{''.join(row)}|")
    lines.append(f"          {'─' * width}")
    lines.append(
        f"          {x_min:<{width // 2}.3f}"
        f"{x_max:>{width - width // 2}.3f}"
    )
    lines.append(f"          {x_key:^{width}}")
    lines.append("")
    lines.append("Legend: + = stable, x = unstable")

    return "\n".join(lines)


def plot_stability_landscape(
    landscape_path: str = "artifacts/stability_landscape.json",
    output_dir: str = "artifacts",
) -> None:
    """Plot stability landscape projections.

    Attempts matplotlib plots first.  Falls back to ASCII if unavailable.
    """
    data = _load_landscape(landscape_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plot_matplotlib(data, output_dir)
        print(f"Plots saved to {output_dir}/")
    except ImportError:
        # Fallback to ASCII
        print(_ascii_scatter(
            data, "spectral_radius", "sis",
            "Spectral Radius vs SIS",
        ))
        print()
        print(_ascii_scatter(
            data, "entropy", "curvature",
            "Entropy vs Curvature",
        ))


def _plot_matplotlib(data: list[dict], output_dir: str) -> None:
    """Plot using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stable = [d for d in data if d.get("predicted_stable", True)]
    unstable = [d for d in data if not d.get("predicted_stable", True)]

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: spectral_radius vs SIS
    fig, ax = plt.subplots(figsize=(8, 6))
    if stable:
        ax.scatter(
            [d["spectral_radius"] for d in stable],
            [d["sis"] for d in stable],
            c="blue", label="Stable", alpha=0.7,
        )
    if unstable:
        ax.scatter(
            [d["spectral_radius"] for d in unstable],
            [d["sis"] for d in unstable],
            c="red", label="Unstable", alpha=0.7,
        )
    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("SIS")
    ax.set_title("Spectral Radius vs SIS")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "landscape_radius_vs_sis.png"), dpi=100)
    plt.close(fig)

    # Plot 2: entropy vs curvature
    fig, ax = plt.subplots(figsize=(8, 6))
    if stable:
        ax.scatter(
            [d["entropy"] for d in stable],
            [d["curvature"] for d in stable],
            c="blue", label="Stable", alpha=0.7,
        )
    if unstable:
        ax.scatter(
            [d["entropy"] for d in unstable],
            [d["curvature"] for d in unstable],
            c="red", label="Unstable", alpha=0.7,
        )
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Curvature")
    ax.set_title("Entropy vs Curvature")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "landscape_entropy_vs_curvature.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/stability_landscape.json"
    plot_stability_landscape(path)
