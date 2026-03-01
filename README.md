# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.6.0](https://img.shields.io/badge/release-v3.6.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.6.0)

License: CC-BY-4.0

QEC — Deterministic QLDPC CSS Framework

Deterministic QLDPC CSS quantum error correction toolkit with invariant-safe algebraic construction, multi-mode belief propagation (sum-product / min-sum family), flooding/layered/hybrid/adaptive scheduling, posterior-aware and decimation-based postprocessing, pluggable deterministic channel modeling, and rigorous FER benchmarking.

Engineered for:

Reproducibility

Interpretability

Structural experimentation

Controlled decoder intervention research

Current Release
v3.6.0 — Deterministic Posterior-Aware Combination-Sweep OSD

v3.6.0 completes the ordering-tier escalation of the decoder layer by introducing:

postprocess="mp_osd_cs"

This extends combination-sweep OSD (osd_cs) by using posterior LLR magnitude (abs(L_post)) instead of channel LLR magnitude for reliability ordering.

Baseline behavior remains bit-identical.

What’s New in v3.6.0
Deterministic Posterior-Aware OSD-CS

Unlike standard osd_cs, which orders candidate pivots using channel LLR magnitude, mp_osd_cs uses posterior magnitude derived from belief propagation.

Implementation pattern:

Run inner BP with:

postprocess=None

llr_history=1

If syndrome satisfied → return immediately

Otherwise apply OSD-CS using ordering based on abs(L_post)

Deterministic tie-breaking (ascending index)

Enforce never-degrade guarantee

Key properties:

No BP loop modifications

No scheduling changes

_bp_postprocess() untouched

No schema changes

No new parameters (reuses osd_cs_lam)

Fully deterministic

This is a strictly additive feature.

Structural Ordering Probe (v3.6.0 Validation)

A controlled lam-sensitivity probe was executed under:

channel_model="bsc_syndrome"

distances [8, 12, 16]

p ∈ [0.01, 0.02, 0.03, 0.04]

lam ∈ {1, 2, 3}

10,800 total trials

Results:

osd_cs and mp_osd_cs produced byte-identical corrections in all trials

No ordering sensitivity observed

DPS inversion persists under syndrome-only inference

Increasing lam did not alter candidate selection

Interpretation:

Under uniform channel LLR (BSC-syndrome), ordering differences do not change the selected coset leader at tested scales. Posterior-aware ordering is structurally verified but not expressive in this regime.

This isolates inversion behavior as an upstream inference or information-model phenomenon rather than an ordering deficiency.

Posterior-Aware OSD-1 (v3.5.0)

v3.5.0 introduced:

postprocess="mp_osd1"

A deterministic posterior-aware single-bit OSD variant.

Structural probe under BSC-syndrome showed:

Reduced inversion magnitude compared to osd1

Lower FER growth across distances

Posterior magnitude superior to channel LLR in syndrome-only regime

Inversion reduced but not eliminated.

Deterministic Guided Decimation (v3.4.0)
postprocess="guided_decimation"

Belief-propagation–guided deterministic variable freezing:

Iterative BP refinement

Freeze variable with maximal |posterior LLR|

Deterministic tie-breaking

Clamp LLR to fixed magnitude

Never-degrade fallback ordering

All operations deterministic.

Inversion Index (II)

Primary scalar diagnostic for structural channel artifacts:

II = SCR - Fidelity
   = syndrome_consistency_rate - (1 - FER)

Interpretation:

II = 0 → no systematic inversion

II > 0 → syndrome-consistent but logically incorrect corrections

II = 1.0 → maximal inversion regime

Derived strictly from deterministic benchmark outputs.

Channel Regime Comparison
Property	Oracle	Syndrome-Only
Effective threshold	~0.50	~0.01–0.02
Inversion regime	Yes (p > 0.50)	No
Inversion Index peak	1.0	~0.0
SCR/FER divergence	Yes	No
Distance scaling	Positive	Inverted
Schedule differentiation	Masked	Real

The Inversion Index separates channel-model artifacts from decoder behavior.

Architecture Overview

Layered system:

Layer 1 — Decoder Core
Belief propagation + deterministic postprocessing.

Layer 2 — Channel Models
Pluggable, deterministic LLR generators.

Layer 3 — Benchmark & Reporting
FER, DPS, inversion diagnostics.

Interop Layer — Canonical JSON + Hash Verification

Strict ownership boundaries enforced.
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

Full test suite passing at release

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
