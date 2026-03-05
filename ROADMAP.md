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

2. Architectural Layers

QEC evolves in controlled layers.
Each layer may expand — but must not destabilize lower layers.

Layer 1 — Decoder Core (Invariant Backbone)

Scope:

Additive invariant QLDPC CSS construction

Deterministic BP variants

Controlled scheduling mechanisms

Ensemble decoding

OSD family (0 / 1 / CS)

Deterministic guided decimation (v3.4.0)

Constraints:

Must remain stable across the entire v3.x line

No benchmarking feature may modify decoding semantics

No hidden adaptive behavior

No implicit randomness

Structural interventions must be opt-in

v3.4.0 establishes the first bounded structural intervention under syndrome-only decoding while preserving baseline semantics.

This layer is the invariant backbone of QEC.

Layer 2 — Deterministic Benchmark & Interop Infrastructure

Formalized in v3.0.x → v3.1.2 and extended in v3.2.x.

Scope:

Schema-validated benchmark configs

Canonical JSON artifacts

Stable sweep hashing

Deterministic artifact hashing

Structured interop record validation

Optional reference baselines (Stim / PyMatching)

Deterministic report generation

Inversion Index (II) formalization

Constraints:

Must not alter Layer 1 semantics

Must preserve byte-identical behavior under deterministic mode

Must isolate third-party tooling

Must enforce schema invariants structurally

v3.1.2 establishes deterministic interop baseline.
v3.2.x formalizes structural diagnostics (II).

Layer 3 — Channel & Noise Modeling

Established in v3.1.3 and hardened in v3.1.4.

Scope:

Pluggable channel abstraction

oracle

bsc_syndrome

Deterministic channel parameterization

Stim-compatible synthetic channel interfaces

Future: AWGN, erasure

Constraints:

Channel layer must not mutate decoder logic

Channel behavior must be deterministic under fixed seed

Channel metadata must be schema-validated

No silent behavior changes to oracle mode

Goal:

Enable realistic FER curves without destabilizing the decoder core.

Layer 4 — Deterministic Diagnostics & Regime Analysis

Formalized in v3.2.x and significantly expanded in the v4.x series.

Scope:

Inversion Index (II)

DPS regime diagnostics

Belief propagation energy tracing

Basin-switch detection

Energy-landscape attractor metrics

Iteration-trace dynamics analysis

These diagnostics characterize decoder behavior across three complementary layers:

• attractor basin geometry  
• free-energy landscape structure  
• per-iteration BP trajectory dynamics  

Constraints:

Diagnostics must not alter decoder semantics

Metrics must be computed from deterministic decoder outputs

No stochastic diagnostic sources

No mutation of decoder state

All diagnostics must remain strictly observational.

The v4.x series establishes QEC as a deterministic experimental platform for
studying belief propagation dynamics on QLDPC energy landscapes.

Layer 5 — Analytical & Dimensional Expansion (Opt-In Only)

Future scope:

Qudit scaffolding

GF(q) exploration

Analytical gate-cost modeling

Resource estimation tooling

Nonbinary decoding research

Constraints:

Must be opt-in

Must preserve qubit defaults

Must not mutate binary decoder semantics

Must preserve reproducibility guarantees

Dimensional expansion must not destabilize binary baseline behavior.

3. Current State — v4.3.0

The v4.x series transitions QEC from a decoder engineering toolkit into a
deterministic experimental platform for studying belief propagation dynamics
on QLDPC codes.

The decoder core remains unchanged from the v3.x architecture.

All v4.x capabilities are implemented strictly in the diagnostics and
benchmarking layers.

Major Additions

Basin Switch Detection (v4.1.0)

Introduces deterministic perturbation diagnostics capable of distinguishing:

• metastable oscillation  
• shallow perturbation sensitivity  
• true basin switching  

This allows systematic identification of attractor basin transitions in the
BP energy landscape.

Energy-Landscape Metrics (v4.2.x)

The diagnostics framework now quantifies the geometry of decoding attractors
using three deterministic metrics:

Basin Stability Index (BSI)

Measures the probability that small deterministic perturbations return the
decoder to the same attractor basin.

Attractor Distance (AD)

Measures the Hamming distance between baseline and perturbed correction
vectors.

Escape Energy (EE)

Estimates the minimum perturbation magnitude required to escape the current
basin.

These metrics allow quantitative characterization of basin stability and
barrier structure in the BP landscape.

Iteration-Trace Diagnostics (v4.3.0)

v4.3 introduces analysis of the internal dynamics of belief propagation
trajectories.

Metrics include:

Belief Oscillation Index (BOI)

Energy Plateau Index (EPI)

Trapping Set Persistence (TSP)

Correction Vector Fluctuation (CVF)

Composite Iteration Stability Score

These metrics characterize oscillatory regimes, plateau behavior, trapping
sets, and correction cycling during decoding.

Architectural Guarantees

Across the entire v4.x series:

decoder core remains unchanged

message passing semantics remain unchanged

scheduling behavior unchanged

schema unchanged

baseline decoding outputs remain byte-identical

All diagnostics operate strictly outside the decoder core.

4. Near-Term Direction (v3.9.x)

Focus: decoder regime exploration under syndrome-only inference

v3.9.x continues the structural experimentation framework introduced in v3.8.x and v3.9.0.

Candidate investigations include:

DPS Sign-Flip Interventions

Goal:

Attempt to induce negative DPS scaling under controlled structural perturbations.

Potential strategies:

inference-geometry reweighting

stabilizer clustering

local field smoothing

deterministic parity-density transforms

BP initialization heuristics

All experiments must:

remain opt-in

preserve deterministic behavior

leave baseline decoder unchanged

Geometry-Aware Scheduling

Possible schedule extensions:

geometry-weighted BP updates

deterministic field annealing

convergence-aware update ordering

All scheduling changes must:

be opt-in

preserve flooding baseline behavior

maintain deterministic iteration ordering

Regime Mapping

Use the expanded DPS harness and energy diagnostics to classify decoding regimes.

Potential outputs:

energy descent profiles

convergence topology

scaling regime transitions

failure surface characterization

5. Near-Term Direction (v4.4+)

With the core diagnostics stack established in v4.1–v4.3, the next research
phase focuses on deeper analysis of BP landscape structure.

Candidate directions include:

Spin-glass–inspired BP diagnostics

Energy barrier topology estimation

Basin transition graph construction

Deterministic perturbation annealing experiments

Attractor stability scaling across code families

Trajectory phase classification

These investigations aim to understand the statistical-physics structure of
belief propagation on QLDPC Tanner graphs.

All future diagnostics must remain:

deterministic

observational

opt-in

decoder-safe

6. What Will Not Happen

Explicitly out of scope:

Hidden randomness in decoding

Silent behavior changes

Schema drift without versioning

Experimental code merged into core paths

Dependency bloat

Benchmark tooling mutating decoder semantics

Third-party code imported into core modules

7. Evolution Philosophy

QEC evolves by strengthening invariants first, expanding capability second.

Each release must satisfy:

Determinism preserved

Backward compatibility respected

Core decoding semantics unchanged (unless major bump)

Clear separation between production logic and research tooling

Reproducibility demonstrated, not assumed

Capability grows.
Stability does not regress.

8. Governance Model

This roadmap governs architectural direction.

It evolves only when:

A new invariant is introduced

A new architectural layer is formalized

A major version transition is planned

Release notes belong in CHANGELOG.md.
Strategic direction belongs here.

Small is beautiful.
Determinism is holy.
Stability is engineered.

If it cannot be reproduced byte-for-byte, it is not a baseline.
