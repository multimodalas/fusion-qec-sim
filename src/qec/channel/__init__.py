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

__all__ = ["ChannelModel", "OracleChannel", "BSCSyndromeChannel"]
