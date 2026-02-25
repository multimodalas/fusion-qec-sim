"""
Benchmark Standardization & Comparative Framework (v3.0.0).

This package provides config-driven, deterministic benchmarking for QEC
decoders.  It is designed as an isolated layer that *consumes* the existing
decoding and simulation APIs without modifying them.

Submodules
----------
schema   : Versioned JSON result schema and canonicalization.
config   : BenchmarkConfig dataclass with JSON I/O.
runner   : Benchmark orchestration engine.
runtime  : Deterministic runtime measurement suite.
compare  : Threshold tables, scaling summaries, iteration analysis.
report   : Pure-Python Markdown / CSV reporting helpers.
adapters : Lightweight decoder adapter interface and implementations.
"""

__all__: list[str] = []
