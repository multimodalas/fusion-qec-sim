QEC Roadmap

This document outlines the architectural and research trajectory of the QEC toolkit.

The roadmap prioritizes deterministic guarantees, disciplined benchmarking, and forward compatibility while expanding analytical rigor and dimensional readiness in controlled stages.

Guiding Principles

Determinism first.

Backward compatibility by default.

No hidden randomness.

No unnecessary dependencies.

Measurable progress per release.

Research features remain opt-in until stabilized.

Core decoding logic is never destabilized for experimental additions.

v2.9.0 — Deterministic Adaptive Control (Completed)

Theme: Scheduling formalization without destabilization.

This release introduced a strictly deterministic adaptive schedule controller while preserving all existing decoder guarantees.

Objectives (Completed)

Introduce deterministic adaptive scheduling.

Preserve backward compatibility.

Maintain bit-stability for default calls.

Expand deterministic validation coverage.

Workstreams
Deterministic Adaptive Scheduling

Added:

schedule="adaptive"

Adaptive mode performs:

Phase 1: flooding for k1 iterations.

Phase 2: hybrid_residual for remaining iterations.

Properties:

Strictly one-way switching.

No residual-triggered dynamic switching.

No internal message state shared between phases.

Cumulative iteration accounting.

Deterministic tie-break ordering:

Converged

Lower syndrome weight

Fewer iterations

Phase order.

Default scheduling behavior remained unchanged.

v2.9.1 — Deterministic Measurement Expansion (Completed)

Theme: Internal observability without behavioral drift.

This release expanded residual instrumentation while preserving decoder semantics.

Objectives (Completed)

Residual metric expansion:

residual_linf

residual_l2

residual_energy

Opt-in instrumentation only.

No changes to default decoder return signature.

Deterministic JSON-safe measurement outputs.

No external dependencies.

Outcome

Residual instrumentation integrated under strict opt-in semantics.

Default decoding behavior bit-identical to v2.9.0.

No scheduling logic changes.

No adaptive logic changes.

No ensemble selection changes.

Determinism preserved.

v3.0.0 — Deterministic Benchmark Standardization (Completed)

Theme: Structured benchmarking and reproducible comparison.

This release formalized evaluation into a deterministic, schema-validated benchmarking framework while preserving all core decoding guarantees.

Objectives (Completed)

Standardize benchmark schema.

Introduce decoder adapter abstraction.

Enable structured internal comparisons.

Provide deterministic, reproducible benchmark artifacts.

Workstreams
1. Benchmark Framework Core

Implemented:

Config-driven benchmark execution.

Canonical JSON result schema (SCHEMA_VERSION = "3.0.0").

Deterministic JSON serialization with stable key ordering.

Schema validation prior to return.

Optional deterministic metadata mode for byte-identical artifacts.

All benchmark runs are reproducible.

2. Order-Independent Seed Derivation

Cryptographic SHA-256 sub-seed derivation.

Seed depends only on:

Base seed

Decoder identity

Code distance

Physical error rate

Eliminates sweep-order coupling.

No reliance on Python hash().

Deterministic repeatability guaranteed.

3. Structured Comparison Suite

Implemented reproducible comparison outputs:

Threshold estimation via FER crossing interpolation.

Log–log runtime scaling analysis.

Iteration distribution summaries.

Runtime regression logic hardened to ensure consistent positive-latency filtering.

4. Decoder Adapter Abstraction

Introduced lightweight DecoderAdapter interface:

Enables structured decoder comparison.

BP adapter wraps existing bp_decode.

No modification to core decoding logic.

No import contamination of core modules.

Architectural Guarantees

Core decoding logic unchanged.

No scheduling changes.

No adaptive logic changes.

No ensemble behavior changes.

No new external dependencies.

Determinism preserved.

v3.0.1 — High-Dimensional Readiness & Compatibility Foundations (Planned)

Theme: Forward compatibility without breaking stability.

This release introduces scaffolding for future nonbinary and high-dimensional extensions while preserving full backward compatibility.

It does not introduce full qudit decoding.

Motivation

Advances in high-dimensional quantum gates suggest:

Native qudit operations can reduce gate overhead.

Equivalent qubit decompositions may require significantly more entangling operations.

Stability and phase control are critical at higher dimensionality.

Software frameworks should become dimension-aware even when operating in qubit mode.

Planned Workstreams
1. Qudit Specification Layer

Introduce an optional dimension specification:

class QuditSpec:
    dimension: int
    encoding: str
    metadata: dict

Default remains dimension=2.

No changes to existing decoder logic.

2. Gate Cost Modeling Utilities

Add lightweight analytical resource estimation:

Qubit decomposition cost estimates.

Native qudit gate cost placeholders.

Deterministic resource comparison helpers.

No simulation changes.

3. Backward Compatibility Audit

Ensure all v2.x and v3.0 APIs remain stable.

Provide migration notes if required.

Maintain deterministic guarantees.

4. Nonbinary Scaffolding (Hooks Only)

Prepare internal placeholders for:

GF(q) message passing.

Nonbinary stabilizer representations.

Qudit-aware syndrome modeling.

No implementation in this release.

Long-Term Direction

The roadmap establishes a deliberate progression:

v2.9.0 — Deterministic adaptive control.

v2.9.1 — Deterministic instrumentation expansion.

v3.0.0 — Deterministic benchmarking and structured comparison.

v3.0.1 — High-dimensional readiness foundations.

Each release expands capability without destabilizing the deterministic core.

Determinism Contract

Across all releases:

No hidden randomness.

Fixed ordering decisions.

Explicit seed control.

Order-independent seed derivation.

Stable floating-point behavior where feasible.

Reproducible benchmark outputs.

Determinism is a first-class architectural invariant.

This roadmap is a living document and may evolve as research and benchmarking results inform future priorities.
