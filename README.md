# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.7.0](https://img.shields.io/badge/release-v3.7.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.7.0)

License: CC-BY-4.0

QEC — Deterministic QLDPC CSS Framework

Deterministic QLDPC CSS quantum error correction toolkit with invariant-safe algebraic construction, multi-mode belief propagation (sum-product / min-sum family), flooding/layered/hybrid/adaptive scheduling, posterior-aware and decimation-based postprocessing, pluggable deterministic channel modeling, and rigorous FER benchmarking.

Engineered for:

Reproducibility

Interpretability

Structural experimentation

Controlled decoder intervention research

Current Release
v3.7.0 — Deterministic Uniformly Reweighted BP (URW) & Geometry Audit

v3.7.0 introduces Uniformly Reweighted Belief Propagation (URW) as a strictly opt-in inference-geometry variant.

mode="min_sum_urw"
urw_rho ∈ (0, 1]

URW applies a uniform scalar damping factor to check-to-variable messages:

R_{j→i} ← ρ · R_{j→i}

Where:

ρ = 1.0 → bit-identical to baseline min_sum

ρ < 1.0 → uniformly damped check influence

Fully deterministic

No adaptive behavior

No stochastic elements

Flooding and layered schedules are supported.

Structural Guarantees

Baseline decoders unchanged

Golden artifact hash unchanged

_bp_postprocess() untouched

Scheduling semantics untouched

Schema versions unchanged

Identity stability preserved

Determinism verified across repeated runs

Full test suite passing

URW is strictly opt-in and introduces zero drift when disabled.

Geometry Probe — DPS Audit (v3.7.0)

A controlled Distance Preservation Slope (DPS) sweep was executed under:

channel_model="bsc_syndrome"

distances [8, 12, 16]

p [0.01, 0.02, 0.03, 0.04]

300 trials per point

ρ ∈ [1.0 … 0.6]

Findings

No ρ value produced negative DPS.

DPS magnitude varied slightly but non-monotonically.

No stable ρ window reduced inversion meaningfully.

URW does not correct syndrome-only distance scaling inversion.

Interpretation

Uniform scalar reweighting of check influence is insufficient to alter the structural inversion regime under syndrome-only inference.

The inversion is not attributable to simple loop overcount amplification and appears structural to the information geometry of syndrome-only decoding.

v3.7.0 eliminates uniform reweighting as a viable correction mechanism.

v3.6.0 — Deterministic Posterior-Aware Combination-Sweep OSD

v3.6.0 completed the ordering-tier escalation by introducing:

postprocess="mp_osd_cs"

This extends osd_cs by using posterior LLR magnitude (abs(L_post)) instead of channel LLR magnitude for reliability ordering.

Key Properties

No BP loop modifications

No scheduling changes

_bp_postprocess() untouched

No schema changes

No new parameters (reuses osd_cs_lam)

Fully deterministic

Ordering Sensitivity Probe

Under bsc_syndrome:

osd_cs and mp_osd_cs produced byte-identical corrections

No ordering sensitivity observed

DPS inversion persisted

Increasing lam did not alter candidate selection

Interpretation:

Ordering differences are structurally verified but not expressive in the syndrome-only regime. Inversion is upstream of reliability ordering.

v3.5.0 — Posterior-Aware OSD-1

Introduced:

postprocess="mp_osd1"

Deterministic posterior-aware single-bit OSD.

Probe results under syndrome-only channel:

Reduced inversion magnitude compared to osd1

Lower FER growth across distances

Posterior magnitude superior to channel LLR ordering

Inversion reduced but not eliminated

Deterministic Guided Decimation (v3.4.0)
postprocess="guided_decimation"

Belief-propagation–guided deterministic variable freezing:

Iterative BP refinement

Freeze variable with maximal |posterior LLR|

Deterministic tie-breaking

Fixed-magnitude LLR clamp

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
Inversion regime	Yes (p>0.50)	Structural
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

Full test suite passing at release.

Design Philosophy

Small is beautiful.
Determinism is holy.
Stability is engineered.

Negative results are data.

If it cannot be reproduced byte-for-byte, it is not a baseline.

Author
Trent Slade
QSOL-IMC
ORCID: https://orcid.org/0009-0002-4515-9237
