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

# v2.9.0 — Deterministic Measurement & Adaptive Control

**Theme:** Internal observability and control formalization.

This release strengthens the internal measurement and scheduling discipline of the decoder without expanding its external surface area.

## Objectives

- Formalize residual instrumentation.
- Introduce deterministic adaptive scheduling.
- Provide reproducible threshold sweep utilities.
- Add bounded internal evaluation harnesses.

## Workstreams

### 1. Residual Metric Expansion

Extend residual tracking to include:

- `residual_linf`
- `residual_l2`
- `residual_energy` (aggregate delta magnitude)

Requirements:

- Fully deterministic.
- Opt-in only.
- No change to default decoding behavior.
- No new external dependencies.

---

### 2. Deterministic Adaptive Scheduling

Add:


schedule="adaptive"


Adaptive mode selects among existing schedules using:

- Residual thresholds
- Iteration checkpoints
- Stable deterministic tie-break rules

Default scheduling behavior remains unchanged.

---

### 3. Threshold Sweep & Deterministic Fitting

Introduce:

- `threshold_sweep()`
- `estimate_threshold()`

Design goals:

- Fixed parameter grids.
- Fixed seeds.
- Stable JSON output.
- Deterministic curve fitting.

No external statistical libraries.

---

### 4. Internal Evaluation Harnesses

Bounded internal comparison tools:

- Residual vs Hybrid schedule harness.
- Ensemble scaling harness.

Limited to small representative code sets to prevent scope expansion.

---

## Explicitly Out of Scope (v2.9.0)

- Thermodynamic decoding modes
- Temperature schedules
- External decoder adapters
- Industry benchmarking integrations
- Heterogeneous decoder fusion
- Major API redesign
- Core math refactors

---

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
