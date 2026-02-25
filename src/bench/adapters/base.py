"""
Abstract base class for decoder adapters.

Adapters wrap existing decoder entrypoints behind a uniform interface
so the benchmark runner can treat all decoders identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DecoderAdapter(ABC):
    """Lightweight adapter wrapping a concrete decoder for benchmarking.

    Subclasses must implement all abstract methods.  The returned dicts
    must contain only plain Python types (no numpy scalars/arrays);
    the benchmark runner will call :func:`~schema.canonicalize` as a
    safety net, but adapters should canonicalize proactively.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name (e.g. ``"bp_min_sum"``)."""

    @abstractmethod
    def initialize(self, *, config: dict[str, Any]) -> None:
        """One-time setup using decoder-specific parameters.

        Called once before any decode / measurement calls.
        """

    @abstractmethod
    def decode(
        self,
        *,
        syndrome: Any | None = None,
        llr: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a single decode and return a result dict.

        Returns
        -------
        dict with at least:
            ``"success"`` : bool — whether decoding corrected the error.
            ``"iters"``   : int  — iteration count for this decode.
            ``"meta"``    : dict — adapter-specific metadata.
        """

    @abstractmethod
    def measure_runtime(self, *, workload: dict[str, Any]) -> dict[str, Any]:
        """Measure decode runtime for a fixed workload.

        The workload dict is produced by the runner and contains the
        inputs needed for a single decode call.

        Returns
        -------
        dict with runtime metrics (see :mod:`runtime`).
        """

    @abstractmethod
    def serialize_identity(self) -> dict[str, Any]:
        """Return a stable identity block for this adapter configuration.

        Keys must be sorted so that serialization is deterministic.
        """
