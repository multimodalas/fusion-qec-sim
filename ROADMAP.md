QEC Roadmap

Deterministic QLDPC CSS Toolkit — Architectural Governance Document

This document defines the structural trajectory, invariants, and expansion boundaries of the QEC toolkit.

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

src/qec/ → core decoding

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

Layer 1 — Core Decoding Stability (Foundation)

Scope:

Additive invariant QLDPC CSS construction

Deterministic BP variants

Controlled scheduling mechanisms

Ensemble decoding

OSD family (0 / 1 / CS)

Deterministic decimation

Constraints:

Must remain stable across the entire v3.x line

No benchmarking feature may modify decoding semantics

No hidden adaptive behavior

No randomness introduced implicitly

This layer is the invariant backbone of QEC.

Layer 2 — Deterministic Benchmark & Interop Infrastructure

Formalized in v3.0.x → v3.1.2.

Scope:

Schema-validated benchmark configs

Canonical JSON artifacts

Stable sweep hashing

Deterministic artifact hashing

Structured interop record validation

Optional reference baselines (Stim / PyMatching)

Deterministic report generation

Reproducibility certification anchor

Constraints:

Must not alter Layer 1 semantics

Must preserve byte-identical behavior under deterministic mode

Must isolate third-party tooling

Must enforce schema invariants structurally

v3.1.2 establishes the deterministic interop baseline.

Layer 3 — Channel & Noise Modeling (Next Expansion Frontier)

Established in v3.1.3 and hardened in v3.1.4.

Scope:

Pluggable channel abstraction

channel_model modes:

oracle

bsc_syndrome

AWGN (planned)

erasure (planned)

Deterministic channel parameterization

Stim-compatible synthetic channel interfaces

Hardware-injected noise adapters (future)

Constraints:

Channel layer must not mutate decoder logic

Channel behavior must be fully deterministic under fixed seed

Channel metadata must be schema-validated

No silent behavior changes to existing oracle mode

Goal:

Enable realistic FER curves without destabilizing decoder core.

Layer 4 — Analytical & Dimensional Expansion (Opt-In Only)

Scope:

Qudit-aware scaffolding

GF(q) exploration

Analytical gate-cost modeling

Resource estimation tooling

Nonbinary decoding research (future)

Constraints:

Must be opt-in

Must preserve qubit defaults

Must not mutate existing decoder semantics

Must preserve reproducibility guarantees

Dimensional expansion must not destabilize binary baseline behavior.

3. Current State — v3.1.4

The v3.1.4 release establishes:

Deterministic interop benchmarking layer (v3.1.2 baseline preserved)

Explicit channel abstraction layer under src/qec/channel/

Supported channel modes:

oracle (default, byte-identical to v3.1.2)

bsc_syndrome (syndrome-only inference)

Realistic FER curves under syndrome-only conditions

Schedule differentiation observable under realistic noise

Non-zero FER at small physical error rates

Channel abstraction hardening (centralized validation, shared constants, registry isolation)

629+ passing tests

No decoder core modifications

No schema version bump

No dependency expansion

v3.1.4 completes the stabilization of Layer 3 (Channel & Noise Modeling).

Layer 3 is now formally established and structurally hardened.

4. Near-Term Direction (v3.1.x → v3.2)

With channel abstraction now formalized, near-term work focuses on controlled channel expansion.

Channel Expansion (Additive Only)

Planned:

AWGN channel

Erasure channel

Deterministic parameterized noise families

Stim-compatible synthetic channel adapters

Hardware-injected noise interfaces (future)

Constraints:

Channel layer must not mutate decoder logic

Oracle mode must remain byte-identical

Determinism under fixed seed must be preserved

No schema drift

No silent behavior changes

Channel realism must not undermine reproducibility.

Decoder Research Under Realistic Noise

Future improvements may include:

Schedule refinement under syndrome-only inference

Improved min-sum normalization strategies

Structured post-processing comparisons

Controlled ensemble behavior evaluation

All decoder experimentation must:

Be version-scoped

Preserve deterministic guarantees

Avoid implicit behavioral drift

Maintain strict layering boundaries

5. Long-Term Direction (v3.2+)

Potential expansions:

Controlled nonbinary message passing

Cross-family scaling audits

Deterministic performance certification framework

Structured comparative benchmarking across code families

All such work must:

Be version-scoped

Preserve baseline determinism

Remain opt-in

Maintain strict schema validation

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
