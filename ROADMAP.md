QEC Roadmap
Deterministic QLDPC CSS Toolkit — Architectural Governance & Research Direction

This document defines the structural trajectory, invariants, and expansion boundaries of QEC.

QEC evolves under a stability-first philosophy.
Experimental expansion is permitted — destabilization is not.

1. Core Architectural Invariants (Non-Negotiable)

These constraints apply to all future releases.

Determinism as Architecture

No hidden randomness

Explicit seed control

Order-independent SHA-256 sub-seed derivation

Canonical JSON serialization

Stable sweep ordering

Artifact hashing over immutable record state

runtime_mode="off" → byte-identical artifacts

Determinism is not a feature.
It is a structural constraint.

Backward Compatibility by Default

Public API stability across minor releases

Schema evolution must be explicitly versioned

Behavioral drift requires version bump

Legacy configs remain runnable

Default decoder behavior must remain bit-stable unless explicitly versioned

Import Hygiene

No circular dependencies

Strict separation:

src/qec/   → core decoding & channel layer
src/bench/ → benchmarking & interop

No experimental leakage into core modules

Optional third-party integrations must be gated

Minimal Dependency Surface

Prefer stdlib

Prefer deterministic primitives

No dependency expansion without architectural justification

No hard dependency on third-party benchmarking tools

2. Architectural Layers (Updated)

QEC evolves in controlled architectural layers.
Each layer may expand — but must not destabilize lower layers.

Layer 1 — Decoder Core (Invariant Backbone)

Scope:

deterministic BP decoding variants

deterministic scheduling strategies

OSD family postprocessing (OSD-0 / OSD-1 / OSD-CS)

deterministic guided decimation

invariant-safe QLDPC CSS construction

Constraints:

decoder semantics must remain stable

no hidden adaptive behavior

no stochastic operations

structural interventions must be opt-in

message passing logic must remain deterministic

This layer represents the experimental object of the framework.

All research instrumentation must remain external to the decoder core.

Layer 2 — Deterministic Benchmark & Interop Infrastructure

Scope:

deterministic DPS evaluation harness

schema-validated benchmark configurations

canonical JSON artifact generation

deterministic artifact hashing

stable parameter sweep ordering

optional reference baseline integrations

Constraints:

must not alter Layer 1 semantics

must preserve byte-identical behavior in deterministic mode

third-party tools must remain optional

This layer provides the reproducible experimental environment for decoding studies.

Layer 3 — Channel & Noise Modeling

Scope:

deterministic channel abstraction

pluggable channel models

syndrome-only inference models

oracle channels

deterministic channel parameterization

Future expansion:

AWGN channel

erasure channel

Stim-compatible noise models

Constraints:

channel models must not mutate decoder logic

channel behavior must be deterministic under fixed seeds

channel metadata must be schema-validated

Layer 4 — Diagnostics & Structural Analysis

Formalized across the v4–v6 development series.

Scope:

BP trajectory diagnostics

attractor basin analysis

free-energy landscape metrics

Tanner graph spectral analysis

eigenmode localization diagnostics

trapping-set candidate detection

These diagnostics characterize decoder behavior across three complementary structures:

BP trajectory dynamics
BP energy landscape geometry
Tanner graph spectral structure

Constraints:

diagnostics must not modify decoder state

diagnostics must operate on deterministic outputs

all analysis must remain observational

This layer transforms the toolkit into a deterministic decoding observatory.

Layer 5 — Predictive Modeling

Introduced in the v6.8–v6.9 development cycle.

Scope:

structural BP instability prediction

spectral failure risk scoring

instability ratio estimation

predictor validation experiments

The predictor layer attempts to estimate decoding instability before decoding runs.

Inputs include:

non-backtracking spectral radius

eigenvector localization (IPR)

Tanner graph structural signals

spectral trapping-set candidates

Outputs include:

bp_failure_risk
predicted_instability
spectral_instability_ratio

Constraints:

predictors must be deterministic

predictors must operate purely on diagnostics outputs

predictors must not modify decoding algorithms

Layer 6 — Decoder Control & Experimental Policies

Introduced in v7.0.

Scope:

spectral-guided decoder control

predictor-guided scheduling

adaptive damping policies

experimental decoding strategies

The controller layer transforms diagnostics and predictor signals into deterministic control policies.

Example policies:

risk-guided message damping

node-priority scheduling

instability-aware decoding experiments

Constraints:

controller must not modify decoder implementation

controller must remain fully deterministic

baseline decoding behavior must remain unchanged when disabled

This layer establishes the **closed-loop decoding experiment framework**.

---

Layer 3 — Channel & Noise Modeling

Scope:

Pluggable deterministic channel abstractions enabling controlled decoding
experiments under reproducible noise conditions.

Supported channel types include:

• oracle  
• bsc_syndrome  

Channel models generate deterministic LLR vectors and provide a consistent
interface between noise processes and the decoder core.

Future channel extensions may include:

• AWGN channels  
• erasure channels  
• synthetic Stim-compatible noise models  

Constraints:

Channel models must not modify decoder logic.

Channel behavior must be deterministic under fixed seed conditions.

Channel metadata must be schema-validated.

Oracle mode must remain stable and must not experience silent behavioral
changes.

Goal:

Enable realistic FER measurements while preserving the stability and
determinism of the decoder core.

---

Layer 4 — Diagnostics & Regime Analysis

Scope:

Deterministic diagnostics that characterize the behavior of belief
propagation decoding without modifying decoder semantics.

Diagnostics include:

• DPS regime diagnostics  
• belief propagation energy tracing  
• basin-switch detection  
• attractor landscape metrics  
• iteration-trace dynamics analysis  
• spectral Tanner graph diagnostics  
• eigenvector localization (IPR) analysis  
• spectral trapping-set candidate detection  

These diagnostics characterize decoder behavior across three complementary
structures:

• attractor basin geometry  
• free-energy landscape structure  
• per-iteration BP trajectory dynamics  

Constraints:

Diagnostics must remain strictly observational.

Metrics must be computed from deterministic decoder outputs.

No stochastic diagnostic sources may be introduced.

Diagnostics must not mutate decoder state or alter decoding behavior.

This layer establishes QEC as a **deterministic observatory for studying
belief propagation dynamics on QLDPC Tanner graphs**.

---

Layer 5 — Analytical & Dimensional Expansion (Opt-In Only)

Scope:

Optional analytical expansion beyond the binary qubit decoding baseline.

Potential areas include:

• GF(q) exploration  
• qudit decoding research  
• analytical gate-cost modeling  
• resource estimation tooling  
• nonbinary decoding strategies  

Constraints:

These features must remain opt-in.

Binary qubit decoding must remain the default system configuration.

Nonbinary experimentation must not alter the semantics of binary decoding.

All analytical expansions must preserve the reproducibility guarantees of
the framework.

Dimensional expansion must never destabilize the binary baseline.

---

Layer 6 — Predictive Modeling

Scope:

Deterministic prediction of decoding instability using structural signals
derived from Tanner graph analysis.

Predictive models estimate the probability of BP decoding failure **before
decoding runs**.

Inputs include:

• non-backtracking spectral radius  
• eigenvector localization metrics (IPR)  
• spectral trapping-set candidates  
• structural Tanner graph signals  

Outputs include:

• `bp_failure_risk`  
• `predicted_instability`  
• `spectral_instability_ratio`  

Constraints:

Predictors must operate purely on diagnostic outputs.

Predictors must be deterministic and reproducible.

Predictors must not modify decoder algorithms or message-passing behavior.

This layer transforms structural diagnostics into **predictive signals about
decoder stability**.

---

Layer 7 — Decoder Control & Experimental Policies

Scope:

Deterministic control policies that use predictor signals to guide decoding
experiments.

Controller capabilities include:

• predictor-guided scheduling  
• adaptive damping policies  
• spectral-aware decoding experiments  

The controller layer transforms diagnostic and predictive signals into
deterministic experimental decoding policies.

Example strategies include:

• risk-guided message damping  
• node-priority update ordering  
• instability-aware decoding experiments  

Constraints:

Controllers must not modify decoder implementation.

Controllers must remain deterministic.

Baseline decoding behavior must remain unchanged when controllers are
disabled.

This layer completes the **closed-loop decoding experiment architecture**.

---

## Current State — v7 Series

The v7 development series establishes QEC as a **spectral-aware decoding
research platform**.

The framework now supports the following experimental pipeline:


Tanner graph structure
↓
spectral diagnostics
↓
BP instability prediction
↓
decoder control policies
↓
controlled decoding experiments


The decoder core itself remains unchanged.

All research instrumentation operates outside the decoder implementation.

This architecture allows systematic investigation of the relationship
between:

• Tanner graph structure  
• spectral modes  
• belief propagation attractor geometry  
• decoding stability and performance  

---

## Near-Term Research Direction (v7.x)

The next research phase focuses on **spectral-aware decoding strategies**.

Candidate directions include:

Cluster-Aware Scheduling

Use spectral trapping-set clusters to prioritize message updates during
decoding.

Spectral Damping Profiles

Derive adaptive damping policies from eigenvector localization signals.

Instability-Aware Decoding Policies

Use predictor outputs to guide experimental decoding strategies such as:

• BP → BPGD  
• BP → OSD  
• hybrid decoding experiments  

Spectral Tanner Graph Optimization

Explore structural graph transformations that reduce decoding instability,
including spectral radius minimization and trapping-set disruption.

All experiments must remain deterministic and opt-in.

---

## Medium-Term Direction (v8)

Future research may explore deeper connections between Tanner graph
structure and decoding algorithms.

Potential investigations include:

• spectral graph design for BP stability  
• topology-aware decoding schedules  
• statistical-physics models of BP phase transitions  
• automated Tanner graph repair algorithms  

These investigations aim to understand the causal relationship between:


graph topology
↓
spectral structure
↓
BP attractor geometry
↓
decoding performance


---

## What Will Not Happen

Explicitly out of scope:

• hidden randomness in decoding  
• silent behavioral drift  
• schema changes without versioning  
• experimental code merged into decoder core paths  
• dependency bloat  
• benchmarking tools mutating decoder semantics  
• third-party code imported into core modules  

---

## Evolution Philosophy

QEC evolves by strengthening invariants first and expanding capability
second.

Every release must preserve:

• determinism  
• architectural separation  
• decoder stability  
• reproducibility guarantees  

Capability grows.  
Stability does not regress.

---

## Governance Model

This roadmap governs architectural direction.

It evolves only when:

• a new invariant is introduced  
• a new architectural layer is formalized  
• a major version transition is planned  

Release notes belong in **CHANGELOG.md**.

Strategic direction belongs here.

---

Small is beautiful.  
Determinism is holy.  
Stability is engineered.

If a result cannot be reproduced byte-for-byte, it is not a baseline.
