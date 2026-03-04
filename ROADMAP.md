QEC Roadmap
Deterministic QLDPC CSS Toolkit — Architectural Governance Document

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

Layer 4 — Structural Diagnostics & Regime Analysis

Formalized in v3.2.x and extended in v3.3.x.

Scope:

Inversion Index (II)

Geometry-aware diagnostics

DPS, BSI, SSI metrics

Structural regime classification

Constraints:

Diagnostics must not alter decoder semantics

Metrics must be algebraically derived from deterministic fields

No stochastic diagnostic sources

This layer distinguishes structural channel artifacts from genuine decoder behavior.

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

3. Current State — v3.4.0

v3.4.0 establishes:

Deterministic guided decimation (Layer 1 structural extension)

Fully preserved baseline decoder semantics

No schedule mutation

No schema change

No identity/hash drift

700+ passing tests

Complete isolation from _bp_postprocess() and BP loops

Layer 1 now supports bounded structural intervention without compromising invariants.

4. Near-Term Direction (v3.5.x)

Focus: Structural decoder refinement under syndrome-only noise.

Possible candidates:

Stabilizer Inactivation (opt-in, version-scoped)

MP-aware OSD fallback chaining

Hybrid decimation + OSD sequencing

Posterior exposure as first-class API

Early plateau detection in decimation rounds

Constraints:

Must remain opt-in

Must preserve baseline decoder behavior

Must maintain determinism

Must avoid implicit schedule mutation

5. Medium-Term Direction (v3.6+)

Cross-family scaling audits

Deterministic performance certification framework

Structured comparative benchmarking across code families

Regime transition mapping under channel perturbation

All such work must:

Be version-scoped

Preserve baseline determinism

Maintain strict schema validation

Avoid semantic drift

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
