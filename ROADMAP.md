# QEC Roadmap

This document outlines the architectural and research trajectory of the QEC toolkit across the next major release cycle.

The roadmap is structured to preserve deterministic guarantees while progressively expanding measurement rigor, benchmarking discipline, and high-dimensional readiness.

---

# Guiding Principles

- Determinism first.
- Backward compatibility by default.
- No hidden randomness.
- No unnecessary dependencies.
- Measurable progress per release.
- Research features remain opt-in until stabilized.

---

# v2.9.0 — Deterministic Adaptive Control

**Theme:** Scheduling formalization without destabilization.

This release introduces a strictly deterministic adaptive schedule controller while preserving all existing decoder guarantees.

## Objectives (Completed)

- Introduce deterministic adaptive scheduling.
- Preserve backward compatibility.
- Maintain bit-stability for default calls.
- Expand deterministic validation coverage.

---

## Workstreams

### 1. Deterministic Adaptive Scheduling (Completed)

Add:

schedule="adaptive"

Adaptive mode performs:

- Phase 1: `flooding` for k1 iterations.
- Phase 2: `hybrid_residual` for remaining iterations.

Properties:

- Strictly one-way switching.
- No residual-triggered dynamic switching.
- No internal message state shared between phases.
- Cumulative iteration accounting.
- Deterministic tie-break ordering:
  - Converged
  - Lower syndrome weight
  - Fewer iterations
  - Phase order

Default scheduling behavior remains unchanged.

---

## Deferred From v2.9.0

The following measurement and instrumentation work has been deferred to a future minor release to preserve release stability:

- Residual metric expansion
- Threshold sweep utilities
- Internal evaluation harnesses

# v2.9.1 — Deterministic Measurement Expansion (Planned)

**Theme:** Internal observability without behavioral drift.

Planned objectives:

- Residual metric expansion:
  - `residual_linf`
  - `residual_l2`
  - `residual_energy`
- Opt-in instrumentation only.
- No changes to default decoder return signature.
- Deterministic JSON-safe measurement outputs.
- No external dependencies.

# v3.0.0 — Benchmark Standardization & Comparative Framework

**Theme:** Structured benchmarking and reproducible comparison.

This release introduces a formal benchmarking framework while preserving deterministic guarantees.

## Objectives

- Standardize benchmark schema.
- Introduce decoder adapter abstraction.
- Enable structured internal comparisons.
- Prepare for optional external benchmarking integrations.

---

## Workstreams

### 1. Benchmark Framework Core

Define:

- Config-driven benchmark execution.
- Stable JSON result schema.
- Standardized metrics:
  - FER
  - WER
  - Iterations
  - Runtime

All benchmark runs must be reproducible.

---

### 2. Decoder Adapter Interface

Introduce a lightweight abstraction layer enabling:

- Internal decoder comparison.
- Optional future external decoder integration.

Adapters remain modular and optional.

---

### 3. Structured Comparison Suite

Generate reproducible comparison outputs such as:

- Threshold tables
- Runtime scaling summaries
- Iteration distribution analysis

Initial scope remains intentionally bounded.

---

### 4. Experimental Research Track (Optional)

Introduce opt-in experimental modules for:

- Energy-based decoder analysis.
- Iteration trajectory inspection.
- Deterministic “temperature-style” control experiments.

These remain isolated from default decoding behavior.

---

# v3.0.1 — Legacy Compatibility & High-Dimensional Readiness

**Theme:** Forward compatibility without breaking stability.

This release introduces foundational scaffolding for future nonbinary and high-dimensional extensions.

It does not introduce full qudit decoding.

---

## Motivation

Recent advances in high-dimensional quantum gates highlight:

- Native qudit operations can reduce gate overhead.
- Equivalent qubit decompositions may require significantly more entangling operations.
- Stability and phase control are critical at higher dimensionality.

Software frameworks should be dimension-aware even if operating in qubit mode.

---

## Workstreams

### 1. Qudit Specification Layer

Introduce an optional dimension specification:

```python
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

4. Future Nonbinary Hooks (Scaffolding Only)

Prepare internal placeholders for:

GF(q) message passing.

Nonbinary stabilizer representations.

Qudit-aware syndrome modeling.

No implementation in this release.

Long-Term Direction

The roadmap establishes a deliberate progression:

v2.9.0 — Deterministic control and measurement.

v3.0.0 — Benchmark rigor and structured comparison.

v3.0.1 — High-dimensional readiness and compatibility foundations.

Each release builds capability without destabilizing the core deterministic design.

Determinism Contract

Across all releases:

No hidden randomness.

Fixed ordering decisions.

Stable floating-point behavior where feasible.

Explicit seed control.

Reproducible benchmark outputs.

Determinism is a first-class architectural invariant.

This roadmap is a living document and may evolve as research and benchmarking results inform future priorities.
