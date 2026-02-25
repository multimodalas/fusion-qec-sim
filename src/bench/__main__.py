"""
CLI entrypoint for the benchmarking framework.

Usage::

    python -m src.bench --config path/to/config.json --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import BenchmarkConfig
from .runner import run_benchmark
from .schema import dumps_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.bench",
        description="Run a QEC benchmark sweep.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a JSON benchmark config file.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Path for the JSON result file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON output (non-canonical but readable).",
    )

    args = parser.parse_args(argv)

    config = BenchmarkConfig.load(args.config)
    result = run_benchmark(config)

    if args.pretty:
        text = json.dumps(result, sort_keys=True, indent=2)
    else:
        text = dumps_result(result)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
            f.write("\n")
        print(f"Results written to {args.out}", file=sys.stderr)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
