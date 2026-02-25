"""
Pure-Python reporting helpers (Markdown tables + CSV).

No pandas, no matplotlib — just deterministic text outputs.
"""

from __future__ import annotations

import csv
import io
from typing import Any


def to_markdown_tables(result_obj: dict[str, Any]) -> str:
    """Convert a benchmark result object to Markdown tables.

    Sections:
    1. Threshold table
    2. FER results table
    3. Runtime scaling (if available)
    4. Iteration distribution summaries (if available)
    """
    lines: list[str] = []
    summaries = result_obj.get("summaries", {})

    # ── Threshold table ──
    thresholds = summaries.get("thresholds", [])
    if thresholds:
        lines.append("## Threshold Estimates\n")
        lines.append("| Decoder | Threshold p | Method | Notes |")
        lines.append("|---------|------------|--------|-------|")
        for t in thresholds:
            tp = t.get("threshold_estimate_p")
            tp_str = f"{tp:.6f}" if tp is not None else "N/A"
            lines.append(
                f"| {t['decoder']} | {tp_str} "
                f"| {t['method']} | {t['notes']} |"
            )
        lines.append("")

    # ── FER results table ──
    results = result_obj.get("results", [])
    if results:
        lines.append("## FER Results\n")
        has_runtime = any(r.get("runtime") is not None for r in results)
        header = "| Decoder | Distance | p | FER | WER | Mean Iters |"
        sep = "|---------|----------|---|-----|-----|------------|"
        if has_runtime:
            header += " Latency (us) |"
            sep += "--------------|"
        lines.append(header)
        lines.append(sep)
        for r in results:
            row = (
                f"| {r['decoder']} | {r['distance']} | {r['p']} "
                f"| {r['fer']:.6f} | {r['wer']:.6f} | {r['mean_iters']:.2f} |"
            )
            if has_runtime:
                rt = r.get("runtime")
                if rt:
                    row += f" {rt['average_latency_us']} |"
                else:
                    row += " N/A |"
            lines.append(row)
        lines.append("")

    # ── Runtime scaling ──
    scaling = summaries.get("runtime_scaling", [])
    if scaling:
        lines.append("## Runtime Scaling\n")
        for s in scaling:
            lines.append(f"### {s['decoder']}\n")
            if s.get("slope") is not None:
                lines.append(f"Log-log slope: {s['slope']}\n")
            if s.get("points"):
                lines.append("| Distance | Avg Latency (us) |")
                lines.append("|----------|-------------------|")
                for pt in s["points"]:
                    lines.append(
                        f"| {pt['distance']} | {pt['average_latency_us']} |"
                    )
                lines.append("")

    # ── Iteration distribution ──
    iter_dists = summaries.get("iteration_distributions", [])
    if iter_dists:
        lines.append("## Iteration Distributions\n")
        lines.append("| Decoder | Distance | p | Mean Iters |")
        lines.append("|---------|----------|---|------------|")
        for it in iter_dists:
            lines.append(
                f"| {it['decoder']} | {it['distance']} | {it['p']} "
                f"| {it['mean_iters']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def to_csv_rows(result_obj: dict[str, Any]) -> dict[str, str]:
    """Convert a benchmark result object to CSV content.

    Returns a dict mapping logical filename → CSV string.
    """
    output: dict[str, str] = {}

    # ── FER results CSV ──
    results = result_obj.get("results", [])
    if results:
        buf = io.StringIO()
        has_runtime = any(r.get("runtime") is not None for r in results)
        fieldnames = ["decoder", "distance", "p", "fer", "wer", "mean_iters"]
        if has_runtime:
            fieldnames.append("average_latency_us")
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: dict[str, Any] = {
                "decoder": r["decoder"],
                "distance": r["distance"],
                "p": r["p"],
                "fer": r["fer"],
                "wer": r["wer"],
                "mean_iters": r["mean_iters"],
            }
            if has_runtime:
                rt = r.get("runtime")
                row["average_latency_us"] = (
                    rt["average_latency_us"] if rt else ""
                )
            writer.writerow(row)
        output["fer_results.csv"] = buf.getvalue()

    # ── Threshold CSV ──
    summaries = result_obj.get("summaries", {})
    thresholds = summaries.get("thresholds", [])
    if thresholds:
        buf = io.StringIO()
        fieldnames = ["decoder", "threshold_estimate_p", "method", "notes"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for t in thresholds:
            writer.writerow({
                "decoder": t["decoder"],
                "threshold_estimate_p": (
                    t["threshold_estimate_p"]
                    if t["threshold_estimate_p"] is not None
                    else ""
                ),
                "method": t["method"],
                "notes": t["notes"],
            })
        output["thresholds.csv"] = buf.getvalue()

    return output
