"""
v9.2.0 — Inspect Discovery Archive.

Loads a discovery run or archive artifact and prints archive category details.

Supports both v9.0.0 (flat) and v9.2.0 (metadata + results/archive) formats.

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

    # Support v9.2.0 archive artifact format
    if "archive" in data:
        archive = data["archive"]
    # Support v9.2.0 discovery_run format (nested under results)
    elif "results" in data:
        archive = data["results"].get("archive_summary", {})
    # Support v9.0.0 flat format
    else:
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
