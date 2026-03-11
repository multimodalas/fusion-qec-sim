"""
v9.3.0 — Tanner Graph Export.

Export parity-check matrices in standard formats for interoperability
with external QEC and LDPC tooling.

Supported formats:
  - Matrix Market (.mtx)
  - Parity-check text (.txt)
  - JSON adjacency (.json)

Layer 5 — IO.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
import os

from scipy.io import mmwrite
from scipy.sparse import csr_matrix

import numpy as np


def export_matrix_market(H: np.ndarray, path: str) -> None:
    """Export parity-check matrix in Matrix Market format.

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix (m x n).
    path : str
        Output file path (.mtx).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mmwrite(path, csr_matrix(H))


def export_parity_check(H: np.ndarray, path: str) -> None:
    """Export parity-check matrix in human-readable text format.

    Format::

        n m
        row_index: variable_indices...

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix (m x n).
    path : str
        Output file path (.txt).
    """
    m, n = H.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{n} {m}\n")
        for i in range(m):
            cols = [str(j) for j in range(n) if H[i, j] != 0]
            f.write(f"{i}: {' '.join(cols)}\n")


def export_json_adjacency(H: np.ndarray, path: str) -> None:
    """Export parity-check matrix as JSON adjacency representation.

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix (m x n).
    path : str
        Output file path (.json).
    """
    m, n = H.shape
    adjacency = {
        "num_variables": int(n),
        "num_checks": int(m),
        "edges": [],
    }
    for i in range(m):
        for j in range(n):
            if H[i, j] != 0:
                adjacency["edges"].append([int(i), int(j)])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(adjacency, f, indent=2, sort_keys=True)
        f.write("\n")
