"""
v9.0.0 — Inspect Discovery Archive.

Loads a discovery run artifact and prints archive category details.

Usage:
    python scripts/inspect_discovery_archive.py [artifact_path]
"""

from __future__ import annotations

import json
import sys


def inspect_archive(artifact_path: str = "artifacts/discovery_run.json") -> None:
    """Print archive summary from a discovery run artifact."""
    with open(artifact_path) as f:
        data = json.load(f)

    archive = data.get("archive_summary", {})

    print("Discovery Archive Summary")
    print("=" * 60)

    for category, info in sorted(archive.items()):
        if isinstance(info, dict):
            print(f"\n  {category}:")
            for k, v in sorted(info.items()):
                print(f"    {k}: {v}")
        else:
            print(f"\n  {category}: {info}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/discovery_run.json"
    inspect_archive(path)
