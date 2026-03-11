"""
v7.7.0 — Tanner Graph Spectral Heatmap Visualization.

Plots spectral instability heatmaps for Tanner graph edges and nodes.

matplotlib is an optional dependency and must not be imported
by core modules.

Usage
-----
    python scripts/plot_tanner_heatmap.py
"""

from __future__ import annotations

import numpy as np


def plot_edge_heatmap(H: np.ndarray) -> None:
    """Plot undirected edge spectral instability heatmap.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    """
    import matplotlib.pyplot as plt

    from qec.diagnostics.spectral_heatmaps import compute_spectral_heatmaps

    heat = compute_spectral_heatmaps(H)["undirected_edge_heat"]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(heat)), heat)

    plt.xlabel("Undirected Edge Index")
    plt.ylabel("Spectral Instability Heat")
    plt.title("Tanner Graph Spectral Heatmap")

    plt.tight_layout()
    plt.show()


def plot_variable_node_heatmap(H: np.ndarray) -> None:
    """Plot variable node spectral instability heatmap.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    """
    import matplotlib.pyplot as plt

    from qec.diagnostics.spectral_heatmaps import compute_spectral_heatmaps

    heat = compute_spectral_heatmaps(H)["variable_node_heat"]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(heat)), heat)

    plt.xlabel("Variable Node Index")
    plt.ylabel("Spectral Instability Heat")
    plt.title("Variable Node Spectral Heatmap")

    plt.tight_layout()
    plt.show()


def plot_check_node_heatmap(H: np.ndarray) -> None:
    """Plot check node spectral instability heatmap.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    """
    import matplotlib.pyplot as plt

    from qec.diagnostics.spectral_heatmaps import compute_spectral_heatmaps

    heat = compute_spectral_heatmaps(H)["check_node_heat"]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(heat)), heat)

    plt.xlabel("Check Node Index")
    plt.ylabel("Spectral Instability Heat")
    plt.title("Check Node Spectral Heatmap")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example: small parity-check matrix
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
    ], dtype=np.float64)

    plot_edge_heatmap(H)
    plot_variable_node_heatmap(H)
    plot_check_node_heatmap(H)
