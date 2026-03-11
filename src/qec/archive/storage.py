"""
v10.0.0 — Persistent Discovery Archive Storage.

SQLite-backed storage for discovered codes.  Each graph stores its
parity-check matrix, fitness, metrics, genealogy, and reproducibility
metadata.  Content-addressable via SHA-256 matrix hash.

Layer 3 — Archive.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _matrix_hash(H: np.ndarray) -> str:
    """Compute a deterministic content-addressable hash."""
    data = np.asarray(H, dtype=np.float64).tobytes()
    return hashlib.sha256(data).hexdigest()


def _serialize_matrix(H: np.ndarray) -> bytes:
    """Serialize a matrix to bytes for storage."""
    return np.asarray(H, dtype=np.float64).tobytes()


def _deserialize_matrix(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Deserialize a matrix from bytes."""
    return np.frombuffer(data, dtype=np.float64).reshape(shape).copy()


class DiscoveryArchive:
    """SQLite-backed persistent archive for discovered codes.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "./discovery_archive.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the archive tables if they do not exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS codes (
                code_id TEXT PRIMARY KEY,
                matrix_hash TEXT NOT NULL,
                num_checks INTEGER NOT NULL,
                num_variables INTEGER NOT NULL,
                H_data BLOB NOT NULL,
                fitness REAL,
                metrics TEXT,
                parent_code TEXT,
                mutation_operator TEXT,
                generation INTEGER,
                seed INTEGER,
                run_metadata TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_codes_fitness
            ON codes(fitness DESC)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_codes_matrix_hash
            ON codes(matrix_hash)
        """)
        self._conn.commit()

    def add_code(
        self,
        H: np.ndarray,
        *,
        fitness: float | None = None,
        metrics: dict[str, Any] | None = None,
        parent_code: str | None = None,
        mutation_operator: str | None = None,
        generation: int | None = None,
        seed: int | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a discovered code to the archive.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).
        fitness : float or None
            Composite fitness score.
        metrics : dict or None
            Detailed metrics dictionary.
        parent_code : str or None
            Code ID of the parent.
        mutation_operator : str or None
            Mutation operator that produced this code.
        generation : int or None
            Discovery generation number.
        seed : int or None
            Seed used during discovery.
        run_metadata : dict or None
            Reproducibility metadata.

        Returns
        -------
        str
            Content-addressable code_id (SHA-256 hash).
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape
        code_id = _matrix_hash(H_arr)

        # Check if already exists
        row = self._conn.execute(
            "SELECT code_id FROM codes WHERE code_id = ?",
            (code_id,),
        ).fetchone()

        if row is not None:
            # Update fitness if better
            if fitness is not None:
                existing = self._conn.execute(
                    "SELECT fitness FROM codes WHERE code_id = ?",
                    (code_id,),
                ).fetchone()
                if existing and (existing[0] is None or fitness > existing[0]):
                    self._conn.execute(
                        "UPDATE codes SET fitness = ?, metrics = ? WHERE code_id = ?",
                        (fitness, json.dumps(metrics or {}, sort_keys=True), code_id),
                    )
                    self._conn.commit()
            return code_id

        timestamp = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO codes
               (code_id, matrix_hash, num_checks, num_variables,
                H_data, fitness, metrics, parent_code, mutation_operator,
                generation, seed, run_metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                code_id,
                code_id,  # matrix_hash == code_id for content-addressable
                m,
                n,
                _serialize_matrix(H_arr),
                fitness,
                json.dumps(metrics or {}, sort_keys=True),
                parent_code,
                mutation_operator,
                generation,
                seed,
                json.dumps(run_metadata or {}, sort_keys=True),
                timestamp,
            ),
        )
        self._conn.commit()
        return code_id

    def get_elite(self, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve the top-k codes by fitness.

        Parameters
        ----------
        top_k : int
            Number of elite codes to retrieve.

        Returns
        -------
        list[dict[str, Any]]
            List of code entries with H matrix.
        """
        rows = self._conn.execute(
            """SELECT code_id, num_checks, num_variables, H_data,
                      fitness, metrics, parent_code, mutation_operator,
                      generation, seed, timestamp
               FROM codes
               WHERE fitness IS NOT NULL
               ORDER BY fitness DESC
               LIMIT ?""",
            (top_k,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_diverse_codes(self, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve diverse codes using greedy distance selection.

        Returns codes that maximise pairwise structural distance
        from the elite set.

        Parameters
        ----------
        top_k : int
            Number of codes to retrieve.

        Returns
        -------
        list[dict[str, Any]]
            List of diverse code entries.
        """
        all_codes = self.get_elite(top_k=top_k * 5)
        if len(all_codes) <= top_k:
            return all_codes

        # Greedy furthest-point selection
        selected = [all_codes[0]]
        remaining = list(range(1, len(all_codes)))

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_dist = -1.0

            for idx in remaining:
                min_dist = float("inf")
                for sel in selected:
                    d = float(np.sum(np.abs(
                        all_codes[idx]["H"] - sel["H"]
                    )))
                    if d < min_dist:
                        min_dist = d
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = idx

            if best_idx >= 0:
                selected.append(all_codes[best_idx])
                remaining.remove(best_idx)
            else:
                break

        return selected

    def seed_population(
        self,
        size: int = 10,
        strategy: str = "elite",
    ) -> list[np.ndarray]:
        """Retrieve matrices to seed a new discovery population.

        Parameters
        ----------
        size : int
            Number of seed matrices to retrieve.
        strategy : str
            Selection strategy: "elite" or "diverse".

        Returns
        -------
        list[np.ndarray]
            List of parity-check matrices.
        """
        if strategy == "diverse":
            entries = self.get_diverse_codes(top_k=size)
        else:
            entries = self.get_elite(top_k=size)

        return [entry["H"] for entry in entries]

    def count(self) -> int:
        """Return the total number of codes in the archive."""
        row = self._conn.execute("SELECT COUNT(*) FROM codes").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _row_to_dict(self, row: tuple) -> dict[str, Any]:
        """Convert a database row to a dictionary."""
        (code_id, num_checks, num_variables, H_data,
         fitness, metrics_str, parent_code, mutation_operator,
         generation, seed, timestamp) = row
        return {
            "code_id": code_id,
            "H": _deserialize_matrix(H_data, (num_checks, num_variables)),
            "fitness": fitness,
            "metrics": json.loads(metrics_str) if metrics_str else {},
            "parent_code": parent_code,
            "mutation_operator": mutation_operator,
            "generation": generation,
            "seed": seed,
            "timestamp": timestamp,
        }
