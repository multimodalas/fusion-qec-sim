"""
Abstract interfaces for future nonbinary / qudit decoding extensions.

These interfaces define the contracts that future GF(q) decoders,
nonbinary stabilizer representations, and qudit syndrome models must
satisfy.  **No implementation is provided in this release.**

All interfaces use :class:`typing.Protocol` for structural subtyping
so that concrete implementations need not inherit from these classes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GFqMessagePasser(Protocol):
    """Interface for GF(q) message-passing decoders.

    Future implementations will perform belief propagation (or similar)
    over a Tanner graph with GF(q) alphabet.
    """

    @property
    def field_order(self) -> int:
        """The finite-field order q (prime or prime power)."""
        ...

    def initialize(self, *, parity_check: Any, field_order: int) -> None:
        """Set up the decoder with a parity-check matrix over GF(q)."""
        ...

    def decode(self, *, syndrome: Any, channel_priors: Any) -> dict[str, Any]:
        """Run one decode pass and return a result dict.

        Returns
        -------
        dict with at least:
            ``"success"``   : bool
            ``"correction"`` : array-like correction vector over GF(q)
            ``"iters"``     : int
        """
        ...


@runtime_checkable
class NonbinaryStabilizerCode(Protocol):
    """Interface for nonbinary stabilizer code representations.

    Captures the essential structure of a [[n, k, d]]_q code without
    specifying the internal representation.
    """

    @property
    def n(self) -> int:
        """Number of physical qudits."""
        ...

    @property
    def k(self) -> int:
        """Number of logical qudits."""
        ...

    @property
    def q(self) -> int:
        """Local dimension (field order)."""
        ...

    def stabilizer_matrix(self) -> Any:
        """Return the stabilizer (parity-check) matrix over GF(q)."""
        ...

    def syndrome(self, error: Any) -> Any:
        """Compute the syndrome for a given error vector."""
        ...


@runtime_checkable
class QuditSyndromeModel(Protocol):
    """Interface for qudit-aware syndrome noise models.

    Future implementations will provide syndrome sampling that accounts
    for the local dimension of the quantum system.
    """

    @property
    def dimension(self) -> int:
        """Local Hilbert-space dimension."""
        ...

    def sample_error(self, *, n: int, p: float, seed: int | None = None) -> Any:
        """Sample a random error vector of length *n* over the qudit alphabet."""
        ...

    def sample_syndrome(
        self, *, code: Any, error: Any
    ) -> Any:
        """Compute or sample a (possibly noisy) syndrome."""
        ...
