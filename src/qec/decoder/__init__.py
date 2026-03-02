"""
Layer 1a — Opt-in structural decoder extensions.

Extensions in this package are gated by explicit configuration and
must not alter baseline BP behavior when disabled.
"""

from .rpc import RPCConfig, StructuralConfig, build_rpc_augmented_system

__all__ = [
    "RPCConfig",
    "StructuralConfig",
    "build_rpc_augmented_system",
]
