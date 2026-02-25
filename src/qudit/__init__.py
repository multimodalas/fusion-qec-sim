"""
Optional qudit specification layer (v3.0.1).

Provides :class:`QuditSpec`, a lightweight, frozen, JSON-safe dimension
specification that defaults to qubit behaviour (dimension=2).

This module is physically isolated from core decoding and benchmark
modules: importing ``src.decoder`` or ``src.bench`` does NOT import
this package.
"""

from .spec import QuditSpec

__all__ = ["QuditSpec"]
