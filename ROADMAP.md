QEC Roadmap

Deterministic QLDPC Tanner Graph Stability Platform

Author: Trent Slade — QSOL-IMC
ORCID: 0009-0002-4515-9237

This document defines the architectural invariants and research trajectory of the QEC framework.

QEC evolves under a stability-first philosophy.

Capability may expand.
Determinism and architectural separation must never regress.

1. Core Architectural Invariants

These constraints govern all future releases.

Determinism as Architecture

The framework must produce byte-identical artifacts under fixed configurations.

Requirements:

no hidden randomness

explicit seed control

canonical JSON serialization

deterministic iteration ordering

stable parameter sweep ordering

SHA-256 artifact hashing

runtime_mode="off" must produce identical outputs across runs.

Determinism is not a feature.

It is a structural constraint.

Decoder Core Protection

The BP decoder is a protected experimental object.

Location:

src/qec/decoder/

Rules:

decoder algorithms must remain unchanged

no stochastic message updates

no hidden adaptive behavior

no experimental code merged into decoder modules

All experimentation must occur outside the decoder core.

Import Hygiene

Strict separation:

src/qec/   → decoding core
src/bench/ → benchmarking
src/exp/   → experimental modules

Constraints:

no circular dependencies

no experimental leakage into core modules

Minimal Dependency Surface

Prefer:

Python standard library

deterministic primitives

sparse linear algebra tools

Dependency expansion requires architectural justification.

2. Research Principles

The QEC framework follows a strict research methodology.

These principles guide implementation decisions.

Measure before intervention.
Structural diagnostics must be validated before modifying graph topology.

Localize instability before repairing it.
Graph repairs must target specific instability regions.

Prefer spectral methods over combinatorial search.
Cycle enumeration scales poorly and must be avoided.

Never rely on stochastic optimization.
All algorithms must remain deterministic.

Graph topology is the primary experimental variable.

The decoder is an immutable black box.

Diagnostics must remain observational.

Predictors must never modify decoding behavior.

Optimization must preserve Tanner graph constraints.

Sparse operators must replace dense matrix construction.

Edge-level diagnostics are preferable to global metrics.

Eigenvector localization is a key structural signal.

Graph repair must use mathematically justified gradients.

Algorithmic complexity must scale with graph size.

Large experiments must produce deterministic artifacts.

Heatmaps replace explicit cycle enumeration.

Optimization must remain explainable.

Phase diagrams reveal system-level behavior.

Mitigation strategies must remain external to the decoder.

Reproducibility overrides performance.

These principles ensure the framework remains a deterministic scientific instrument rather than a heuristic optimization system.

3. Architectural Layer Model

The system evolves through controlled architectural layers.

Higher layers may expand, but must not destabilize lower layers.

Layer 1 — Decoder Core

Scope:

deterministic belief propagation decoding

deterministic scheduling strategies

OSD post-processing

deterministic guided decimation

invariant-safe QLDPC construction

The decoder core is the experimental object of the system.

Layer 2 — Benchmark Infrastructure

Scope:

deterministic experiment harness

parameter sweep automation

canonical artifact generation

deterministic artifact hashing

Provides the reproducible environment for experiments.

Layer 3 — Channel Modeling

Scope:

deterministic channel models

pluggable noise abstractions

syndrome-only inference channels

Future expansions:

AWGN channel

erasure channel

Stim-compatible noise models

Channels must not modify decoder logic.

Layer 4 — Structural Diagnostics

Transforms QEC into a decoding observatory.

Diagnostics include:

BP trajectory analysis

attractor basin detection

free-energy landscape metrics

Tanner graph spectral diagnostics

eigenvector localization (IPR)

trapping-set candidate detection

Diagnostics remain strictly observational.

Layer 5 — Instability Prediction

Predict decoding instability before decoding runs.

Inputs:

NB spectral radius

eigenvector localization

spectral trapping-set signals

Outputs:

bp_failure_risk
predicted_instability
spectral_instability_ratio

Predictors must remain deterministic.

Layer 6 — Experimental Decoder Control

External policies guiding decoding experiments.

Examples:

spectral-aware scheduling

instability-aware damping

predictor-guided decoding experiments

Controllers must never modify the decoder implementation.

4. Spectral Tanner Graph Stability Program

The v7-v8 development cycle studies structural BP instability.

Working hypothesis:

cycle clusters
↓
NB eigenvector localization
↓
Bethe-Hessian instability modes
↓
BP convergence failure

Key spectral signals:

non-backtracking spectral radius

dominant NB eigenvector

inverse participation ratio (IPR)

spectral edge sensitivity

Edge sensitivity proxy:

proxy(edge) ≈ |v_i|² · |v_j|²

These signals identify cycle-resonant trapping sets.

5. v7 Development Series

The v7 series establishes QEC as a spectral Tanner-graph research platform.

Pipeline:

Tanner graph
↓
spectral diagnostics
↓
instability localization
↓
graph optimization
↓
decoder experiments

The decoder core remains unchanged.

6. Near-Term Roadmap
v7.6.1 — Diagnostic Validation

Goal:

Validate spectral sensitivity signals.

Features:

IPR metrics

sensitivity ranking

Precision@k validation

instability correlation tests

Success criterion:

Spectral signals correctly identify instability edges.

v7.7.0 — Spectral Trapping-Set Heatmaps

Goal:

Convert spectral signals into spatial instability maps.

Outputs:

node instability heatmaps

edge instability heatmaps

trapping-set candidate regions

Cycle enumeration is intentionally avoided.

v7.8.0 — Gradient-Guided Graph Repair

Goal:

Repair Tanner graphs using spectral gradients.

Procedure:

identify unstable edges
↓
compute eigenvector sensitivity
↓
apply degree-preserving edge swaps
↓
validate graph constraints

Repairs must preserve QLDPC stabilizer commutativity.

v7.9.0 — Incremental Spectral Updates

Goal:

Accelerate optimization loops.

Techniques:

warm-start eigensolvers

low-rank perturbation updates

Bethe-Hessian operator formulation

Full recomputation remains the fallback.

7. v8 Research Program
v8.0.0 — Tanner Graph Stability Phase Diagrams

Goal:

Map decoding stability regimes.

Phase axes:

X: channel noise
Y: NB spectral radius

Observed regimes:

stable decoding

oscillatory BP

trapping-set regime

repaired-graph regime

v8.1.0 — Adaptive Ternary Decoder Experiments

Goal:

Mitigate unavoidable trapping sets.

Method:

Identify unstable nodes via spectral heatmaps.

Modify initial priors:

LLR_i = 0

for high-instability nodes.

The BP decoder remains unchanged.

8. Explicit Anti-Patterns

Prohibited:

modifying the BP decoder core

stochastic graph repair algorithms

machine-learning heuristics replacing spectral analysis

dense NB matrix instantiation

ignoring QLDPC stabilizer constraints

9. Evolution Philosophy

QEC evolves by strengthening invariants first and expanding capability second.

Every release must preserve:

determinism

architectural separation

decoder stability

reproducibility

Capability grows.

Stability does not regress.

10. Governance

This document defines architectural direction.

Release notes belong in:

CHANGELOG.md

Strategic direction belongs here.

Small is beautiful.
Determinism is holy.
Stability is engineered.

If a result cannot be reproduced byte-for-byte, it is not a baseline.
