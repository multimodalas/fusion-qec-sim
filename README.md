# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.5.0](https://img.shields.io/badge/release-v3.5.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.5.0)

License: CC-BY-4.0

QEC — Deterministic QLDPC CSS Framework

Deterministic QLDPC CSS quantum error correction framework featuring:

Invariant-safe algebraic construction

Multi-mode belief propagation (sum-product / min-sum family)

Flooding, layered, hybrid-residual, and adaptive scheduling

Deterministic guided decimation (v3.4.0)

Syndrome-only and oracle channel models

Statistically rigorous FER simulation

Schema-validated deterministic benchmarking

Structural regime diagnostics via Inversion Index

This project is engineered for reproducibility, interpretability, and controlled experimentation.

Current Release
v3.4.0 — Deterministic Guided Decimation

v3.4.0 introduces an opt-in structural intervention for syndrome-only decoding:

postprocess="guided_decimation"

This release adds deterministic belief-propagation–guided variable freezing without modifying:

BP message-passing semantics

Scheduling logic

_bp_postprocess() behavior

Schema versions

Default decoder behavior

This is a decoder-layer structural extension — not a reporting-layer formalization.

What’s New in v3.4.0
Deterministic Guided Decimation

For each decimation round:

Run BP for decimation_inner_iters

If syndrome satisfied → return immediately

Select unfrozen variable with maximal |posterior LLR|

Tie-break by lowest index (deterministic)

Zero-posterior convention → freeze positive (hard = 0)

Clamp LLR to ±decimation_freeze_llr

Repeat up to decimation_rounds

Fallback ranking (if convergence fails):

(syndrome_weight, hamming_weight, round_index)

All operations are fully deterministic.

Added Parameters

(Validated only when enabled)

decimation_rounds (default: 10)

decimation_inner_iters (default: 10)

decimation_freeze_llr (default: 1000.0)

Baseline decoder calls ignore these parameters.

Structural Guarantees (v3.4.0)

Flooding schedule loop unchanged

Layered schedule loop unchanged

_bp_postprocess() unchanged

SCHEMA_VERSION unchanged (3.0.1)

INTEROP_SCHEMA_VERSION unchanged (3.1.2)

No identity/hash drift for baseline decoders

No new dependencies

No randomness introduced

All tests passing at release time:

701 passed, 7 skipped
Inversion Index (v3.2.1)

The Inversion Index (II) remains the primary scalar diagnostic for structural channel artifacts.

II = SCR - Fidelity
   = syndrome_consistency_rate - (1 - FER)

Interpretation:

II = 0 → no systematic inversion

II > 0 → syndrome-consistent but logically incorrect corrections

II = 1.0 → maximal inversion regime (oracle p > 0.50)

The metric is algebraically derived from existing deterministic fields.
No stochastic sources are introduced.

Cross-Channel Structural Analysis
Property	Oracle Channel	Syndrome-Only
Effective threshold	~0.50	~0.01–0.02
Inversion regime	Yes (p > 0.50)	None
Inversion Index peak	1.0	~0.0
SCR/FER divergence	Yes	No
Distance scaling	Positive	Negative
Schedule differentiation	Masked	Real

The Inversion Index remains the clearest scalar separating channel-model artifacts from genuine decoder behavior.

Statistical Noise Bound

For a linear code with m independent parity checks:

P[random syndrome match] ≈ 2^(−m)
Expected matches ≈ T · 2^(−m)

Observed II values (~0.002–0.010) under the syndrome-only channel correspond to random structural coincidence, not a hidden inversion mechanism.

Architecture Overview

Layered system:

Layer 1 — Decoder Core

Layer 2 — Channel Models

Layer 3 — Benchmark & Reporting

Interop Layer — Canonical JSON + hash verification

Strict ownership boundaries are enforced.

No layer mutates another layer’s invariants.

Reproducibility Anchor

Deterministic interop baseline:

SCHEMA_VERSION = 3.0.1

INTEROP_SCHEMA_VERSION = 3.1.2

Deterministic Suite Artifact (SHA-256):

431f7573a0ba8af4784b385f528cfe99d6169eb74798eabddd146def278b6d77

Golden Vector Hash:

86babd2ec81daa165d3ce778b9eb71a3766667484e1c51a2000642ae08ec9569

Deterministic configuration:

runtime_mode="off"
deterministic_metadata=True
seed=12345

If it cannot be reproduced byte-for-byte, it is not a baseline.

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
