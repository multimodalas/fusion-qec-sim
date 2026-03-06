"""
Decoder experiment interface — registry for selecting decoder implementations.

Provides ``get_decoder`` to resolve a decoder name to its callable.
Used by the benchmark harness to switch between reference and experimental
decoders without modifying decoding logic.
"""

from __future__ import annotations

from .bp_decoder_reference import bp_decode as bp_decode_reference
from .bp_decoder_experimental import bp_decode as bp_decode_experimental

DECODER_REGISTRY = {
    "reference": bp_decode_reference,
    "experimental": bp_decode_experimental,
}


def get_decoder(name: str):
    """Return the decoder callable registered under *name*.

    Raises ``ValueError`` for unknown decoder names.
    """
    if name not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder: {name}")
    return DECODER_REGISTRY[name]
