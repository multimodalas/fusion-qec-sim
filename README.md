# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.0.1](https://img.shields.io/badge/release-v3.0.2-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.0.2)

License: CC-BY-4.0

Deterministic QLDPC CSS quantum error correction framework with invariant-safe algebraic construction, multi-mode belief propagation, ensemble and residual scheduling, statistically rigorous FER simulation, and a schema-validated deterministic benchmarking framework.

---

Current Release
v3.0.2 — Canonicalization Determinism Hardening

Fuzz-style determinism testing for canonicalization

Idempotence validation on randomized nested metadata

JSON roundtrip stability verification

Byte-stable artifact guarantees reinforced

No decoder behavior changes

No new dependencies

Backward compatible with schema 3.0.1

v3.0.2 strengthens the artifact reproducibility contract without modifying any production decoding logic.

Determinism Guarantees

Determinism is a first-class architectural invariant.

No hidden randomness

Explicit seed control

Order-independent SHA-256 sub-seeds

Canonical JSON serialization

Canonicalization idempotence (fuzz-validated)

Stable sweep ordering

runtime_mode="off" → byte-identical artifacts

Reproducibility is enforced structurally, not probabilistically.

Architecture
Core Decoding Layer

Additive invariant QLDPC CSS construction

Multi-mode BP (sum-product, min-sum, norm, offset)

Flooding, layered, residual, hybrid schedules

Ensemble decoding

OSD-0 / OSD-1 / OSD-CS

Deterministic decimation

Benchmarking Layer (src/bench/)

Config-driven execution

Schema-validated results

Canonical JSON output

Determinism verification mode

Threshold and runtime analysis

Core decoding logic is not modified by benchmarking features.

Documentation

Full release history: CHANGELOG.md

Forward direction: ROADMAP.md

Benchmark artifacts: /bench/

Author: Trent Slade
ORCID: https://orcid.org/0009-0002-4515-9237

Small is beautiful.
Determinism is holy.
Stability is engineered.
