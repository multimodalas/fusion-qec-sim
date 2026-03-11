"""
v9.2.0 — Discovery Run Experiment.

Runs the structure discovery engine and saves generation-level
artifacts to JSON.  Includes reproducibility metadata.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.utils.reproducibility import collect_environment_metadata
from src.utils.canonicalize import canonicalize


def run_discovery_experiment(
    spec: dict[str, Any],
    *,
    num_generations: int = 10,
    population_size: int = 8,
    base_seed: int = 0,
    archive_top_k: int = 5,
    output_path: str = "artifacts/discovery_run.json",
) -> dict[str, Any]:
    """Run a discovery experiment and save the artifact.

    Parameters
    ----------
    spec : dict[str, Any]
        Generation specification with keys: num_variables, num_checks,
        variable_degree, check_degree.
    num_generations : int
        Number of discovery generations.
    population_size : int
        Population size per generation.
    base_seed : int
        Deterministic base seed.
    archive_top_k : int
        Archive elites per category.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    dict[str, Any]
        Discovery results artifact.
    """
    result = run_structure_discovery(
        spec,
        num_generations=num_generations,
        population_size=population_size,
        base_seed=base_seed,
        archive_top_k=archive_top_k,
    )

    metadata = collect_environment_metadata(
        spec=spec,
        generation_count=num_generations,
        population_size=population_size,
    )

    artifact = {
        "metadata": metadata,
        "results": {
            "spec": {
                "num_variables": spec["num_variables"],
                "num_checks": spec["num_checks"],
                "variable_degree": spec["variable_degree"],
                "check_degree": spec["check_degree"],
            },
            "config": {
                "num_generations": num_generations,
                "population_size": population_size,
                "base_seed": base_seed,
                "archive_top_k": archive_top_k,
            },
            "best_candidate": result["best_candidate"],
            "elite_history": result["elite_history"],
            "archive_summary": result["archive_summary"],
            "generation_summaries": result["generation_summaries"],
        },
    }

    artifact = canonicalize(artifact)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    # Export archive artifact with metadata
    archive_path = output_path.replace("discovery_run", "discovery_archive")
    if archive_path != output_path:
        archive_artifact = {
            "metadata": metadata,
            "archive": result["archive_summary"],
        }
        archive_artifact = canonicalize(archive_artifact)
        with open(archive_path, "w") as f:
            json.dump(archive_artifact, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")

    return artifact
