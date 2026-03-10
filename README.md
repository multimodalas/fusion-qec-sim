# QSOLKCB / QEC  
### Deterministic Quantum Error Correction Research Framework

[![License: CC BY 4.0](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Research Framework](https://img.shields.io/badge/type-research%20framework-blue)]

QEC is a **deterministic research framework for studying belief propagation decoding dynamics in QLDPC CSS quantum error correction codes**.

The system provides a controlled environment for investigating decoder behavior, attractor geometry, spectral structure of Tanner graphs, and decoding stability under reproducible experimental conditions.

Unlike typical simulation toolkits, QEC is designed as an **experimental platform for decoding research**, emphasizing deterministic execution, transparent algorithms, and reproducible benchmarks.

---

# Overview

Belief propagation decoding on sparse parity-check graphs exhibits complex dynamical behavior including:

- trapping sets
- oscillatory convergence
- metastable states
- incorrect fixed points
- spectral fragility

Understanding these phenomena requires controlled experiments where decoder behavior can be observed without stochastic noise or hidden heuristics.

The QEC framework provides a deterministic infrastructure for performing these studies.

Core capabilities include:

- invariant-safe QLDPC CSS code construction
- deterministic belief propagation decoding
- structured decoder experimentation
- spectral Tanner graph diagnostics
- stability prediction and decoder control
- reproducible FER / DPS benchmarking

All components are engineered to maintain **byte-identical results across repeated runs**.

---

# Key Principles

The framework follows several strict design principles.

### Determinism

All algorithms are deterministic.

No stochastic elements are introduced unless explicitly controlled.  
Repeated runs with identical inputs produce **identical outputs**.

### Experimental Transparency

Decoder behavior is explicitly observable and measurable.

Diagnostics operate outside the decoder core and never modify message passing.

### Architectural Separation

The system is organized into clearly separated layers:


Diagnostics → Predictors → Controllers → Decoder → Benchmark Harness


Each layer observes or steers the layer below it without violating its invariants.

### Reproducible Benchmarking

All experiments are designed to support reproducible research results.

If a result cannot be reproduced deterministically, it is not considered a baseline.

---

# Core Capabilities

## Deterministic QLDPC CSS Construction

The framework provides tools for constructing quantum LDPC CSS codes with invariant-safe parity-check matrices.

Features include:

- protograph-based constructions
- deterministic lifting
- parity-check validation
- CSS commutation verification
- reproducible structural transformations

These tools allow controlled experimentation with Tanner graph topology.

---

## Belief Propagation Decoding

Multiple deterministic BP variants are available:

- sum-product
- min-sum
- normalized min-sum
- offset min-sum

Supported scheduling modes include:

- flooding
- layered
- residual
- hybrid residual
- adaptive scheduling

All schedules are implemented without stochastic elements.

---

## Deterministic Postprocessing

Postprocessing algorithms refine BP corrections without altering message passing.

Available strategies include:

- Ordered statistics decoding (OSD)
- combination-sweep OSD
- posterior-guided OSD
- BP-guided deterministic decimation

These methods remain deterministic and fully reproducible.

---

## Spectral Tanner Graph Diagnostics

The framework provides deterministic spectral diagnostics for Tanner graphs.

These diagnostics analyze structural properties of the parity-check matrix including:

- spectral radius
- spectral gaps
- eigenmode localization
- inverse participation ratio (IPR)

Localization analysis identifies nodes associated with fragile spectral modes.

These structural signals often correlate with decoding instability.

---

## BP Dynamics Diagnostics

The toolkit includes a layered diagnostics stack for analyzing belief propagation behavior.

Diagnostics measure properties such as:

- belief oscillation
- energy plateau dynamics
- trapping-set persistence
- basin switching
- convergence regimes

These measurements allow systematic investigation of the BP energy landscape.

---

## Spectral-Guided Decoder Control (v7)

The latest system introduces a **spectral-guided decoder control layer**.

This experimental controller uses structural and dynamical diagnostics to steer decoder behavior.

Control strategies include:

- predictor-guided scheduling
- adaptive per-node damping
- risk-aware decoding policies

The controller operates entirely outside the decoder core, preserving algorithmic invariants.

---

# Benchmarking Framework

The repository includes a deterministic benchmarking harness for measuring decoder performance.

Experiments evaluate:

- Frame Error Rate (FER)
- Distance Performance Scaling (DPS)
- decoding iteration counts
- stability metrics
- spectral control effectiveness

All benchmark runs reuse identical deterministic error instances.

This ensures fair comparisons between decoding strategies.

---

# Architecture

The framework follows a layered experimental architecture.


Code Construction
↓
Channel Model
↓
Decoder Core
↓
Postprocessing
↓
Diagnostics
↓
Predictors
↓
Decoder Control
↓
Benchmark Harness


Each layer is isolated to preserve reproducibility and interpretability.

The decoder core remains the experimental object while diagnostics and controllers act as external instrumentation.

---

# Determinism Guarantees

The system enforces strict deterministic execution.

Key guarantees include:

- no hidden randomness
- deterministic scheduling
- deterministic perturbation experiments
- stable JSON artifacts
- identical outputs across repeated runs

Baseline decoder results remain unchanged when diagnostics are disabled.

---

# Research Applications

The QEC framework enables research into:

- belief propagation attractor geometry
- trapping-set dynamics
- spectral fragility of Tanner graphs
- decoding stability prediction
- distance performance scaling of QLDPC codes
- structure-aware decoding strategies

The system is intended as a **research instrument** for studying inference dynamics in sparse graphical models used in quantum error correction.

---

# Running Experiments

Example benchmark execution:


PYTHONPATH=. python bench/dps_v381_eval.py
--trials 200
--distances 5 7
--p-values 0.03


Additional experimental features can be enabled using CLI flags for diagnostics, predictors, and controller experiments.

---

# Repository Documentation

Key project documentation includes:

- **PROJECT_STATE.md** — architecture snapshot
- **ROADMAP.md** — long-term research direction
- **CHANGELOG.md** — release history
- **AUDIT_CHECKLIST.md** — pre-merge safety verification

These documents define the current system state and research trajectory.

---

# Design Philosophy

The project follows several guiding principles:

Small is beautiful.  
Determinism is essential.  
Transparent algorithms beat opaque heuristics.

Negative results are data.

---

# Author

**Trent Slade**  
QSOL-IMC

ORCID  
https://orcid.org/0009-0002-4515-9237
