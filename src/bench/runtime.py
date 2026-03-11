"""
Deterministic runtime measurement suite (stdlib + numpy only).

Measures decode latency / throughput using ``time.perf_counter_ns``.
Runtime *values* vary by machine, but the measurement *procedure* and
result serialization are fully deterministic.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np


def measure_runtime(
    decode_fn: Callable[[], Any],
    *,
    warmup: int = 5,
    runs: int = 30,
    measure_memory: bool = False,
) -> dict[str, Any]:
    """Measure the latency of *decode_fn* over *runs* invocations.

    Parameters
    ----------
    decode_fn:
        Zero-argument callable that performs one decode.
    warmup:
        Number of warm-up invocations (not measured).
    runs:
        Number of measured invocations.
    measure_memory:
        If ``True``, use :mod:`tracemalloc` to capture peak memory.
        Best-effort; may be inaccurate on some platforms.

    Returns
    -------
    dict with keys:
        ``average_latency_us``       : int
        ``std_latency_us``           : int
        ``confidence_interval_us``   : [int, int]   (mean ± 1.96·σ/√n, clamped ≥0)
        ``throughput_mhz``           : float
        ``memory_usage_mb``          : float | None
        ``runs``                     : int
        ``warmup``                   : int
    """
    # Warm-up (not measured).
    for _ in range(warmup):
        decode_fn()

    # Optional memory tracking.
    mem_peak: float | None = None
    if measure_memory:
        import tracemalloc
        tracemalloc.start()

    # Timed runs.
    latencies_ns: list[int] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        decode_fn()
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)

    if measure_memory:
        import tracemalloc
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_peak = round(peak / (1024 * 1024), 2)

    arr = np.array(latencies_ns, dtype=np.float64)
    mean_ns = float(np.mean(arr))
    std_ns = float(np.std(arr, ddof=1)) if runs > 1 else 0.0

    mean_us = int(round(mean_ns / 1000.0))
    std_us = int(round(std_ns / 1000.0))

    # 95% confidence interval: mean ± 1.96 * (std / sqrt(n)).
    margin_ns = 1.96 * (std_ns / math.sqrt(runs)) if runs > 0 else 0.0
    ci_low = int(round(max(0.0, (mean_ns - margin_ns)) / 1000.0))
    ci_high = int(round((mean_ns + margin_ns) / 1000.0))

    # Throughput: decodes per second / 1e6.
    if mean_ns > 0:
        throughput_mhz = round(1e9 / mean_ns / 1e6, 6)
    else:
        throughput_mhz = 0.0

    return {
        "average_latency_us": mean_us,
        "std_latency_us": std_us,
        "confidence_interval_us": [ci_low, ci_high],
        "throughput_mhz": throughput_mhz,
        "memory_usage_mb": mem_peak,
        "runs": runs,
        "warmup": warmup,
    }
