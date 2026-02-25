QEC Roadmap

This document outlines the architectural trajectory of the QEC toolkit.

The roadmap prioritizes:

Determinism as a first-class invariant

Backward compatibility by default

Strict import hygiene

Zero hidden randomness

Minimal dependency surface

Controlled, opt-in research expansion

Core decoding logic is never destabilized for experimental additions.

Current State (Completed Releases)
v2.9.0 — Deterministic Adaptive Control

Formalized adaptive scheduling while preserving bit-stable defaults.

Deterministic adaptive schedule (schedule="adaptive")

Strict one-way phase switching

Deterministic tie-breaking

No behavioral drift in default modes

v2.9.1 — Deterministic Instrumentation Expansion

Expanded internal observability without modifying decoding semantics.

Opt-in residual metrics (linf, l2, energy)

JSON-safe measurement outputs

No API breakage

Bit-identical default behavior

v3.0.0 — Deterministic Benchmark Standardization

Established schema-validated, reproducible benchmarking.

Config-driven benchmark framework (src/bench/)

Canonical JSON schema (SCHEMA_VERSION)

Order-independent SHA-256 sub-seed derivation

DecoderAdapter abstraction

Structured threshold and runtime analysis

Byte-identical artifact support

Core decoding layer remained untouched.

v3.0.1 — Legacy Compatibility & High-Dimensional Readiness

Introduced forward-compatible scaffolding without altering decoder behavior.

Optional QuditSpec (dimension-aware configuration)

Deterministic analytical gate-cost modeling

Centralized canonicalization utility

Schema version preservation invariant

Backward compatibility audit suite

Import hygiene verification

No nonbinary decoding implemented

No public API changes.
No new dependencies.
Determinism preserved.

Determinism Contract (Permanent)

Across all releases:

No hidden randomness

Explicit seed control

Order-independent sub-seed derivation

Stable key ordering

Canonical JSON serialization

runtime_mode="off" → reproducible artifacts

Backward compatibility unless explicitly version-bumped

Determinism is a non-negotiable architectural invariant.

Near-Term Direction (v3.0.x Line)

Short-term releases will focus on:

Determinism hardening (test-layer reinforcement)

Canonicalization stability guarantees

Benchmark contract enforcement

Controlled analytical expansion (opt-in only)

Continued import isolation discipline

No changes to core decoding semantics are planned in the v3.0.x line.

Medium-Term Direction (v3.1+)

Future major/minor directions may include:

Controlled introduction of nonbinary decoding implementations

GF(q) message-passing support (opt-in)

Qudit-aware syndrome modeling

Extended analytical resource modeling

Expanded deterministic comparison tooling

Any such expansion will:

Remain opt-in

Preserve existing qubit behavior

Avoid destabilizing the decoding core

Maintain strict reproducibility guarantees

Architectural Philosophy

The toolkit evolves in controlled layers:

Core decoding stability

Deterministic validation and benchmarking

Analytical expansion

Dimensional and structural extensibility

Each release must expand capability without destabilizing the deterministic core.

This roadmap is a living document and will evolve as research and benchmarking results inform future priorities.
