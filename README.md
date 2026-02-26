# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.1.3](https://img.shields.io/badge/release-v3.1.4-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.1.4)

License: CC-BY-4.0

Deterministic QLDPC CSS quantum error correction framework featuring invariant-safe algebraic construction, multi-mode belief propagation, ensemble and residual scheduling, statistically rigorous FER simulation, and a schema-validated deterministic benchmarking and interop system.

Current Release
v3.1.4 — Channel Architecture Hardening

v3.1.4 strengthens the structural integrity of the pluggable channel abstraction introduced in v3.1.3.

This is a non-behavioral hardening release.

Improvements in this release:

Centralized probability validation in ChannelModel

Shared _EPSILON constant to prevent numeric drift across channel implementations

Channel registry relocated to src/qec/channel/ (Layer 2 ownership)

Benchmark runner no longer owns channel registry logic

Explicit documentation of oracle default serialization compatibility

Behavioral Guarantees:

OracleChannel remains byte-identical to v3.1.2 artifacts

BSCSyndromeChannel behavior unchanged from v3.1.3

No decoder modifications

No schema changes

No artifact hash drift

No dependency expansion

Core SCHEMA_VERSION remains 3.0.1.
INTEROP_SCHEMA_VERSION remains 3.1.2.

All 629 tests passing at release time.

Reproducibility Anchor

The deterministic interop baseline remains anchored to v3.1.2.

Deterministic Suite Artifact (SHA-256):

431f7573a0ba8af4784b385f528cfe99d6169eb74798eabddd146def278b6d77

Golden Vector Hash:

86babd2ec81daa165d3ce778b9eb71a3766667484e1c51a2000642ae08ec9569

Deterministic configuration:

runtime_mode="off"

deterministic_metadata=True

seed=12345

This artifact remains the canonical baseline for channel modeling and decoder differentiation work.

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
