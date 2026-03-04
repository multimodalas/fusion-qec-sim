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

3. Current State — v3.9.0

v3.9.0 establishes the first deterministic inference-geometry interventions and introduces belief propagation energy diagnostics for structural regime analysis.

The release expands the deterministic experimentation framework introduced in v3.8.x.

Major Additions

Channel Geometry Interventions

Two opt-in structural interventions were introduced:

Centered Syndrome Field

Pseudo-Prior Injection

These construct decoder likelihood fields directly from parity-check structure and syndrome information.

Purpose:

analyze syndrome-only inference geometry

explore decoder behavior without oracle channel information

maintain deterministic experimentation

Both interventions are strictly adapter-layer features and do not modify baseline decoder behavior.

Belief Propagation Energy Diagnostics

v3.9.0 introduces deterministic per-iteration BP energy tracing:

E = − Σ (LLR_i · belief_i)

This enables analysis of:

BP convergence regimes

oscillatory decoding behavior

likelihood alignment across iterations

free-energy landscape structure

Energy tracing is diagnostic only and does not alter decoder outputs.

Expanded DPS Evaluation Harness

The deterministic DPS harness introduced in v3.8.1 now supports geometry-intervention modes.

Evaluation modes now include:

baseline
rpc_only
geom_v1_only
rpc_geom
centered
prior
centered_prior
geom_centered
geom_centered_prior
rpc_centered
rpc_centered_prior

All modes reuse identical deterministic error instances.

Stability Guarantees

Despite these additions:

baseline decoder outputs remain byte-identical

BP schedules are unchanged

_bp_postprocess() remains unchanged

decoder semantics are preserved

all structural features remain opt-in

Test Suite

Current test coverage:

904+ passing tests
0 failures

The test suite now includes validation for:

geometry LLR construction

BP energy tracing

deterministic mode invariance

DPS harness expansion

adapter-layer intervention safety

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

5. Medium-Term Direction (v4.0+)

v4.0 will transition from decoder engineering toward decoder regime analysis.

Primary goal:

Formalize BP free-energy landscape analysis for QLDPC codes.

Planned Capabilities

Energy-landscape mapping

per-iteration BP energy profiles

attractor basin detection

convergence topology analysis

Regime classification

stable decoding basins

oscillatory regimes

failure plateaus

Deterministic scaling analysis

DPS regime mapping across code families

structural perturbation sensitivity

topology-dependent decoding behavior

Research Direction

v4.0 moves QEC toward a deterministic experimental platform for decoding physics rather than only a decoder implementation.

Focus shifts from:

"How do we improve decoding?"

to

"What regimes does BP occupy on QLDPC energy landscapes?"

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
