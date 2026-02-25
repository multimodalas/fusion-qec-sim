"""
Belief-propagation decoder adapter for the benchmarking framework.

Wraps the existing :func:`~src.qec_qldpc_codes.bp_decode` entrypoint
behind the :class:`DecoderAdapter` interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DecoderAdapter
from ..runtime import measure_runtime


class BPAdapter(DecoderAdapter):
    """Adapter wrapping ``bp_decode`` for benchmark use.

    All decoder parameters are captured explicitly so that comparison
    results are reproducible.
    """

    def __init__(self) -> None:
        self._params: dict[str, Any] = {}
        self._H: np.ndarray | None = None

    @property
    def name(self) -> str:
        mode = self._params.get("mode", "sum_product")
        schedule = self._params.get("schedule", "flooding")
        pp = self._params.get("postprocess", "none")
        return f"bp_{mode}_{schedule}_{pp}"

    def initialize(self, *, config: dict[str, Any]) -> None:
        self._params = dict(config)
        # H is stored separately (not part of the identity block).
        self._H = self._params.pop("H", None)

    def decode(
        self,
        *,
        syndrome: Any | None = None,
        llr: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from ...qec_qldpc_codes import bp_decode

        if self._H is None:
            raise RuntimeError("Adapter not initialized (no H matrix)")

        result = bp_decode(
            self._H,
            llr,
            syndrome_vec=syndrome,
            **self._params,
        )
        correction, iters = result[0], result[1]

        e = kwargs.get("error_vector")
        success = False
        if e is not None:
            residual = np.asarray(e) ^ np.asarray(correction)
            success = not bool(np.any(residual))

        return {
            "success": bool(success),
            "correction": correction,
            "iters": int(iters),
            "meta": {},
        }

    def measure_runtime(self, *, workload: dict[str, Any]) -> dict[str, Any]:
        from ...qec_qldpc_codes import bp_decode

        if self._H is None:
            raise RuntimeError("Adapter not initialized (no H matrix)")

        llr = workload["llr"]
        syn = workload.get("syndrome")

        def _decode() -> None:
            bp_decode(self._H, llr, syndrome_vec=syn, **self._params)

        warmup = workload.get("warmup", 5)
        runs = workload.get("runs", 30)
        mem = workload.get("measure_memory", False)

        return measure_runtime(
            _decode, warmup=warmup, runs=runs, measure_memory=mem,
        )

    def serialize_identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {"adapter": "bp"}
        # Sort params for deterministic output.
        identity["params"] = {
            k: _to_python(v) for k, v in sorted(self._params.items())
        }
        return identity


def _to_python(v: Any) -> Any:
    """Convert numpy types to plain Python for serialization."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _to_python(val) for k, val in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_to_python(x) for x in v]
    return v
