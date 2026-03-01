# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.5.0](https://img.shields.io/badge/release-v3.5.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.5.0)

License: CC-BY-4.0

QEC — Deterministic QLDPC CSS Framework

Deterministic QLDPC CSS quantum error correction toolkit with invariant-safe algebraic construction, multi-mode belief propagation (sum-product / min-sum family), flooding/layered/hybrid/adaptive scheduling, posterior-aware and decimation-based postprocessing, pluggable deterministic channel modeling, and rigorous FER benchmarking.

This project is engineered for:

Reproducibility

Interpretability

Structural experimentation

Controlled decoder intervention research

Current Release
v3.5.0 — Posterior-Aware OSD-1 & Structural Inversion Mitigation

v3.5.0 introduces an opt-in posterior-aware postprocess mode:

postprocess="mp_osd1"

This extends the decoder layer without modifying baseline behavior.

What’s New in v3.5.0
Deterministic MP-Aware OSD-1

Unlike standard osd1, which orders candidate flips using channel LLR magnitude, mp_osd1 uses posterior LLR magnitude (abs(L_post)) derived from belief propagation.

Implementation pattern:

Run inner BP with:

postprocess=None

llr_history=1

If syndrome satisfied → return immediately

Otherwise apply OSD-1 with reliability ordering based on posterior magnitude

Deterministic tie-breaking (ascending index)

Enforce never-degrade guarantee

No BP message-passing semantics are altered.

No scheduling logic is modified.

_bp_postprocess() remains untouched.

Schema versions remain unchanged.

Structural DPS Probe (v3.5.0 Validation)

Before tagging, a controlled structural probe was executed under:

channel_model="bsc_syndrome"

distances [5, 7]

p = 0.015

150 trials

min-sum flooding schedule

Four decoders were compared:

None

osd1

mp_osd1

guided_decimation

Distance Performance Slope (DPS)
Decoder	DPS
osd1	+0.0673
guided_decimation	+0.0552
none	+0.0513
mp_osd1	+0.0410

All decoders remain in the positive-slope (inverted) regime under syndrome-only inference.

However:

mp_osd1 produces the lowest inversion magnitude

It reduces FER growth between distances

Posterior-aware ordering outperforms channel-LLR ordering under syndrome-only conditions

Inversion is reduced but not eliminated.

This establishes posterior magnitude as a structurally superior reliability metric in this regime.

Deterministic Guided Decimation (v3.4.0)

v3.4.0 introduced:

postprocess="guided_decimation"

Deterministic belief-propagation–guided variable freezing:

For each decimation round:

Run BP for decimation_inner_iters

If syndrome satisfied → return immediately

Select unfrozen variable with maximal |posterior LLR|

Deterministic tie-breaking

Clamp LLR to ±decimation_freeze_llr

Repeat up to decimation_rounds

Fallback ranking:

(syndrome_weight, hamming_weight, round_index)

All operations are fully deterministic.

Inversion Index (v3.2.1)

The Inversion Index (II) remains the primary scalar diagnostic for structural channel artifacts.

II = SCR - Fidelity
   = syndrome_consistency_rate - (1 - FER)

Interpretation:

II = 0 → no systematic inversion

II > 0 → syndrome-consistent but logically incorrect corrections

II = 1.0 → maximal inversion regime (oracle p > 0.50)

The metric is algebraically derived from deterministic benchmark fields.

No stochastic sources are introduced.

Channel Regime Comparison
Property	Oracle	Syndrome-Only
Effective threshold	~0.50	~0.01–0.02
Inversion regime	Yes (p > 0.50)	No
Inversion Index peak	1.0	~0.0
SCR/FER divergence	Yes	No
Distance scaling	Positive	Negative
Schedule differentiation	Masked	Real

The Inversion Index cleanly separates channel-model artifacts from decoder behavior.

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

Structural Guarantees

Flooding loop unchanged

Layered loop unchanged

_bp_postprocess() unchanged

No schema changes

No dependency expansion

No hidden randomness

No identity/hash drift for baseline decoders

Determinism verified across repeated runs

All tests passing at release time.

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
