QEC Roadmap

This document defines the architectural trajectory and governance principles of the QEC toolkit.

QEC evolves under a stability-first philosophy.
Experimental expansion is permitted — destabilization is not.

1. Core Principles (Non-Negotiable)

The following invariants apply to all future releases:

Determinism as Architecture

No hidden randomness

Explicit seed control

Order-independent SHA-256 sub-seed derivation

Canonical JSON serialization

Stable sweep ordering

runtime_mode="off" → byte-identical artifacts

Determinism is not a feature. It is a structural constraint.

Backward Compatibility by Default

Public API stability across minor releases

Schema evolution must be versioned

Behavioral drift requires explicit version bump

Legacy configs remain runnable

Import Hygiene

No circular dependencies

Clear separation between decoding and benchmarking layers

No leakage from experimental modules into core

Minimal Dependency Surface

No dependency expansion without architectural justification

Prefer stdlib

Prefer deterministic primitives

2. Architectural Layers

QEC evolves in controlled layers.

Layer 1 — Core Decoding Stability (Foundation)

Additive invariant QLDPC CSS construction

Deterministic BP variants

Controlled scheduling mechanisms

OSD family

Deterministic decimation

This layer must remain stable across the entire v3.x line.

No benchmarking or research feature may modify decoding semantics implicitly.

Layer 2 — Deterministic Benchmark & Validation Infrastructure

Schema-validated benchmark configs

Canonical JSON artifacts

Reproducible sweep orchestration

Determinism verification modes

Threshold/runtime analysis tooling

This layer may expand — but never alter Layer 1 behavior.

Layer 3 — Analytical & Dimensional Expansion (Opt-In)

Qudit-aware scaffolding

Analytical gate-cost modeling

Resource estimation

Controlled GF(q) exploration

Nonbinary decoding (future)

All such features must:

Be opt-in

Preserve qubit defaults

Maintain reproducibility

Avoid semantic mutation of existing decoders

3. Current State (v3.0.x Line)

The v3.0.x line is a determinism hardening cycle.

Established Foundations

Adaptive schedule formalization (v2.9.0)

Deterministic instrumentation (v2.9.1)

Schema-validated benchmarking framework (v3.0.0)

Canonicalization centralization (v3.0.1)

Fuzz-validated canonicalization determinism (v3.0.2)

v3.0.x Objectives

Strengthen determinism guarantees

Harden canonicalization invariants

Expand reproducibility testing

Preserve decoding semantics entirely

Maintain zero new dependencies

No decoder behavior changes are planned in the remainder of v3.0.x.

4. Medium-Term Direction (v3.1+)

Future releases may introduce controlled expansion:

Nonbinary Exploration (Opt-In Only)

GF(q) message passing

Qudit-aware syndrome modeling

Dimension-aware decoders

Advanced Decoding Research (Strictly Isolated)

Extended BP variants

Hybrid classical post-processing

Structured comparison harnesses

Comparative Infrastructure

Deterministic industry comparison harness

Structured reproducibility certification

Scaling audits across code families

All such work must:

Be version-scoped

Preserve qubit behavior

Remain optional

Maintain reproducibility guarantees

5. What Will Not Happen

The following are explicitly out of scope:

Hidden randomness in decoding

Silent behavior changes

Schema drift without versioning

Experimental code merged into core paths

Dependency bloat

Benchmark tooling modifying decoder logic

6. Evolution Philosophy

QEC evolves by strengthening invariants first, expanding capability second.

Each release must satisfy:

Determinism preserved

Backward compatibility respected

Core decoding semantics unchanged (unless major version bump)

Clear separation between production logic and research tooling

Reproducibility demonstrated, not assumed

Capability grows.
Stability does not regress.

7. Living Document Policy

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
