"""
Placeholder stubs for future nonbinary / qudit decoding.

Every public function or method in this module raises
:class:`NotImplementedError` with a clear message indicating that
the feature is scaffolding only and will be implemented in a future
release.

These stubs exist so that downstream code can reference the API
surface without triggering import errors, while clearly signalling
that no implementation is available yet.
"""

from __future__ import annotations

from typing import Any


def gfq_bp_decode(
    *,
    parity_check: Any,
    field_order: int,
    syndrome: Any,
    channel_priors: Any,
    max_iters: int = 50,
) -> dict[str, Any]:
    """Placeholder for GF(q) belief-propagation decoding.

    .. warning:: Not implemented in v3.0.1.  This is scaffolding only.
    """
    raise NotImplementedError(
        "GF(q) belief-propagation decoding is not implemented in this release. "
        "This placeholder will be replaced in a future version."
    )


def nonbinary_stabilizer_syndrome(
    *,
    stabilizer_matrix: Any,
    error_vector: Any,
    field_order: int,
) -> Any:
    """Placeholder for nonbinary stabilizer syndrome computation.

    .. warning:: Not implemented in v3.0.1.  This is scaffolding only.
    """
    raise NotImplementedError(
        "Nonbinary stabilizer syndrome computation is not implemented in this release. "
        "This placeholder will be replaced in a future version."
    )


def qudit_error_sample(
    *,
    n: int,
    dimension: int,
    p: float,
    seed: int | None = None,
) -> Any:
    """Placeholder for qudit error sampling.

    .. warning:: Not implemented in v3.0.1.  This is scaffolding only.
    """
    raise NotImplementedError(
        "Qudit error sampling is not implemented in this release. "
        "This placeholder will be replaced in a future version."
    )
