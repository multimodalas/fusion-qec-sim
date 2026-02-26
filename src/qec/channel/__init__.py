"""
Pluggable channel models for LLR construction.

Provides:

- :class:`ChannelModel` — abstract base.
- :class:`OracleChannel` — backward-compatible oracle LLR (sign from error vector).
- :class:`BSCSyndromeChannel` — syndrome-only BSC (uniform LLR, no sign leakage).
"""

from .base import ChannelModel
from .oracle import OracleChannel
from .bsc_syndrome import BSCSyndromeChannel

__all__ = ["ChannelModel", "OracleChannel", "BSCSyndromeChannel", "get_channel_model"]

_CHANNEL_REGISTRY = {
    "oracle": OracleChannel,
    "bsc_syndrome": BSCSyndromeChannel,
}


def get_channel_model(name: str) -> ChannelModel:
    """Look up a channel model by name and return an instance.

    Raises :class:`ValueError` if *name* is not a registered channel.
    """
    cls = _CHANNEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown channel_model {name!r}. "
            f"Available: {sorted(_CHANNEL_REGISTRY)}"
        )
    return cls()
