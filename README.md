# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.0.1](https://img.shields.io/badge/release-v3.0.2-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.0.2) 

License: CC-BY-4.0

Deterministic QLDPC CSS quantum error correction framework featuring invariant-safe algebraic construction, multi-mode belief propagation, ensemble and residual scheduling, statistically rigorous FER simulation, and a schema-validated deterministic benchmarking and interop system.

Current Release
v3.1.2 — Deterministic Interop Baseline & Schema Hardening

v3.1.2 formalizes the benchmarking and interop layer as a deterministic, reproducible baseline suitable for controlled comparative research.

This release:

Establishes a deterministic benchmark artifact anchor

Enforces strict schema validation for interop records

Hardens artifact hashing and canonical JSON guarantees

Removes post-hash mutation risks

Aligns documentation with validation rules

Preserves all core decoder logic

No decoder architecture changes.
No new dependencies.
Core SCHEMA_VERSION remains 3.0.1.
INTEROP_SCHEMA_VERSION is 3.1.2.

Reproducibility Anchor

Deterministic Suite Artifact (SHA-256):

431f7573a0ba8af4784b385f528cfe99d6169eb74798eabddd146def278b6d77

Golden Vector Hash:

86babd2ec81daa165d3ce778b9eb71a3766667484e1c51a2000642ae08ec9569

Deterministic configuration:

runtime_mode="off"

deterministic_metadata=True

seed=12345

All tests passing at release time.

This artifact serves as the canonical baseline for future channel modeling and decoder differentiation work.

Determinism Guarantees

Determinism is a first-class architectural invariant.

Explicit seed control (no hidden randomness)

Order-independent SHA-256 sub-seeds

Canonical JSON serialization (sort_keys=True, compact separators)

Canonicalization idempotence (fuzz-validated)

Stable sweep ordering

Byte-identical artifacts when runtime_mode="off"

Artifact hashes computed over immutable record state

Schema validation enforces determinism metadata

Reproducibility is engineered structurally — not assumed probabilistically.

Architecture
Core Decoding Layer

Additive invariant QLDPC CSS construction

Multi-mode BP:

Sum-product

Min-sum

Normalized

Offset

Scheduling:

Flooding

Layered

Residual

Hybrid

Ensemble decoding

OSD-0 / OSD-1 / OSD-CS

Deterministic decimation

Explicit channel LLR modeling

Finite-field lifting with invariant safety

Core decoding logic is isolated from benchmarking and interop layers.

Benchmarking & Interop Layer (src/bench/)

Config-driven execution

Strict schema-validated results

Canonical JSON output

Stable sweep hashing

Artifact SHA-256 fingerprinting

Deterministic report generation

Import hygiene enforcement (third-party tools isolated)

Structured reference baselines (Stim / PyMatching optional)

Third-party tools are gated and never imported by core modules.

Interop Schema (v3.1.2)

Interop records include:

benchmark_kind (direct_comparison / reference_baseline)

code_family

representation

Determinism block with:

canonical JSON configuration

stable sweep hash

artifact hash

Structured skipped records for unavailable tools

Schema validation prevents malformed or ambiguous results.

Documentation

Release history: CHANGELOG.md

Forward direction: ROADMAP.md

Determinism contract: docs/REPRODUCIBILITY.md

Interop policy: docs/INTEROP_POLICY.md

Legal tool matrix: docs/LEGAL_THIRD_PARTY.md

Release artifacts: release_artifacts/

Design Philosophy

Small is beautiful.
Determinism is holy.
Stability is engineered.

No hidden state.
No accidental randomness.
No silent schema drift.

If it cannot be reproduced byte-for-byte, it is not a baseline.

Author

Trent Slade
QSOL-IMC
ORCID: https://orcid.org/0009-0002-4515-9237
