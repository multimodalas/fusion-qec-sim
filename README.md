# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.1.3](https://img.shields.io/badge/release-v3.2.1-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.2.1)

License: CC-BY-4.0

QEC — Deterministic QLDPC CSS Framework

Deterministic QLDPC CSS quantum error correction framework featuring:

Invariant-safe algebraic construction

Multi-mode belief propagation (min-sum)

Flooding, layered, hybrid-residual, and adaptive scheduling

Syndrome-only and oracle channel models

Statistically rigorous FER simulation

Schema-validated deterministic benchmarking

Structural regime diagnostics via Inversion Index

This project is engineered for reproducibility, interpretability, and controlled experimentation.

Current Release
v3.2.1 — Inversion Index Formalization & Structural Channel Diagnostics

v3.2.1 introduces the Inversion Index (II) as a deterministic diagnostic metric and completes the formal structural comparison between oracle and syndrome-only channel models.

This is a report-layer structural formalization release.

No decoder, channel, or schema behavior was modified.

What’s New in v3.2.1
Inversion Index (II)
II = SCR - Fidelity
   = syndrome_consistency_rate - (1 - FER)

The Inversion Index isolates syndrome-consistent but logically incorrect decoding outcomes.

Interpretation:

II = 0 → no systematic inversion

II > 0 → syndrome-consistent but logically wrong corrections exist

II = 1.0 → maximal inversion regime (oracle p > 0.50)

The metric is an exact algebraic derivative of existing deterministic fields.
No new stochastic sources were introduced.

Cross-Channel Structural Analysis

The release completes the formal comparison between:

Oracle channel (v3.1.4 baseline)

BSC syndrome-only channel (v3.2.0 baseline)

Key structural differences:

Property	Oracle	Syndrome-Only
Effective threshold	~0.50 (degenerate)	~0.01–0.02
Inversion regime	Yes (p > 0.50)	None
Inversion Index peak	1.0	~0.0
SCR/FER divergence	Yes	No
Distance scaling	Positive	Negative
Schedule differentiation	Masked	Real

The Inversion Index is the clearest scalar separating channel-model artifacts from genuine decoder behavior.

Statistical Noise Bound

Small non-zero II values under the syndrome-only channel are now formally bounded.

For a linear code with m independent parity checks:

P[random syndrome match] ≈ 2^(−m)
Expected matches ≈ T · 2^(−m)

Observed II values (~0.002–0.010) correspond to 1–5 events per 500 trials — consistent with random structural coincidence, not a hidden inversion mechanism.

Behavioral Guarantees (v3.2.1)

Decoder logic unchanged (Layer 1 untouched)

Channel implementations unchanged

SCHEMA_VERSION remains 3.0.1

INTEROP_SCHEMA_VERSION remains 3.1.2

No dependency expansion

No artifact hash drift

Determinism preserved

All tests passing at release time:

629 passed, 7 skipped
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

If it cannot be reproduced byte-for-byte, it is not a baseline.

Architecture Overview

Layered system:

Layer 1 — Decoder Core
Layer 2 — Channel Models
Layer 3 — Benchmark & Reporting
Interop Layer — Canonical JSON + hash verification

Strict ownership boundaries are enforced.

No layer mutates another layer’s invariants.

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
