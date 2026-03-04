# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.9.0](https://img.shields.io/badge/release-v3.9.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.9.0)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

QEC — Deterministic QLDPC CSS Framework

Deterministic QLDPC CSS quantum error correction toolkit with invariant-safe algebraic construction, multi-mode belief propagation decoding, deterministic postprocessing, pluggable channel models, and reproducible FER/DPS benchmarking.

The framework is designed for controlled decoder experimentation under strict determinism guarantees.

Core Goals

The system is engineered for:

Reproducibility — byte-identical results across runs

Interpretability — explicit algorithmic behavior

Structural experimentation — controlled topology / inference geometry changes

Deterministic benchmarking — stable FER and distance-scaling measurements

If a result cannot be reproduced byte-for-byte, it is not considered a baseline.

Current Releases
v3.9.0 — Channel Geometry Interventions & BP Energy Diagnostics

Introduces deterministic channel-geometry interventions and belief propagation energy diagnostics.

This release expands the structural experimentation framework introduced in v3.8.x and enables deeper analysis of syndrome-only inference geometry.

Channel Geometry Interventions

Two deterministic interventions are available.

Centered Field Projection

Removes the uniform syndrome bias before projection:

b = 1 − 2s
b_centered = b − mean(b)

LLR = Hᵀ b_centered

Purpose:

remove global field collapse

restore directional likelihood structure

preserve deterministic decoding

Pseudo-Prior Injection

Adds a weak deterministic prior derived from parity structure:

parity_bias = Hᵀ(1 − 2s)

LLR ← LLR + κ · parity_bias

Default:

κ = 0.25

Properties:

deterministic

no oracle leakage

opt-in only

BP Energy Diagnostics

An optional diagnostic records per-iteration BP energy traces:

E = − Σ (LLR_i · belief_i)

This allows analysis of:

convergence dynamics

oscillatory BP behavior

likelihood alignment

Energy tracing is strictly diagnostic and does not alter decoder behavior.

DPS Harness Expansion

The deterministic evaluation harness now supports geometry-intervention modes.

Mode	Centered	Prior	geom_v1	RPC
baseline	F	F	F	F
centered	T	F	F	F
prior	F	T	F	F
centered_prior	T	T	F	F
geom_centered	T	F	T	F
geom_centered_prior	T	T	T	F
rpc_centered	T	F	F	T
rpc_centered_prior	T	T	F	T

All modes reuse identical deterministic error instances.

Stability Improvements

Stabilized bp_decode return structure across optional features

Added epsilon threshold for DPS sign detection

Strengthened baseline invariance tests

v3.8.1 — Structural Geometry Evaluation

Introduces a deterministic Distance Performance Scaling (DPS) evaluation harness.

New module:

bench/dps_v381_eval.py

Features:

Deterministic RNG (seed = 42)

Pre-generated error instances reused across modes

Activation audits for structural interventions

Determinism verification

DPS slope measurement across distances

Evaluation modes:

Mode	Schedule	RPC
baseline	flooding	disabled
rpc_only	flooding	enabled
geom_v1_only	geom_v1	disabled
rpc_geom	geom_v1	enabled

Frame error rate uses syndrome-consistency semantics:

syndrome(H, correction) != s

This release established a reproducible experimental harness for structural decoder analysis.

v3.8.0 — Structural Geometry Core

Adds deterministic infrastructure for topology and inference-geometry experiments.

RPC Augmentation

Deterministic redundant parity checks generated via lexicographic row-pair XOR.

Properties:

deterministic

no feasible-set change

opt-in only

no in-place mutation

geom_v1 Schedule

Flooding-style belief propagation with deterministic check-degree scaling:

α_c = 1 / sqrt(d_c)

No adaptive logic or stochastic elements are introduced.

Adapter Integration

Structural interventions are applied at the adapter layer so that baseline decoder behavior remains unchanged when disabled.

Decoder Core

The decoding stack supports multiple belief propagation variants:

sum_product

min_sum

norm_min_sum

offset_min_sum

Scheduling modes:

flooding

layered

residual

hybrid_residual

adaptive

geom_v1

All schedules are deterministic.

Deterministic Postprocessing

Deterministic correction refinement algorithms include:

Method	Description
osd1	deterministic single-bit ordered statistics
osd_cs	combination-sweep ordered statistics
mp_osd1	posterior-aware OSD-1
mp_osd_cs	posterior-aware combination sweep
guided_decimation	BP-guided deterministic variable freezing

Postprocessing is strictly layered and does not modify BP schedules.

Channel Models

Channel models generate deterministic LLR vectors:

Model	Description
oracle	full error visibility
bsc_syndrome	syndrome-only inference
custom	pluggable deterministic models

Channel models are isolated from decoder logic.

Benchmarking & Diagnostics

Benchmark tools provide deterministic measurement of:

FER (frame error rate)

DPS (distance performance scaling)

syndrome consistency rate

inversion diagnostics

activation audit reports

BP energy convergence traces

The system separates measurement instrumentation from decoder implementation.

Architecture

Layered architecture with strict boundaries.

Layer 1 — Decoder Core
Belief propagation + deterministic postprocessing

Layer 2 — Channel Models
Deterministic LLR generation

Layer 3 — Benchmark & Diagnostics
FER / DPS / structural analysis

Interop Layer — JSON schema + hash verification

No layer mutates another layer’s invariants.

Reproducibility Anchor

Deterministic schema versions:

SCHEMA_VERSION = 3.0.1
INTEROP_SCHEMA_VERSION = 3.1.2

Deterministic suite artifact:

SHA256
431f7573a0ba8af4784b385f528cfe99d6169eb74798eabddd146def278b6d77

Golden vector hash:

86babd2ec81daa165d3ce778b9eb71a3766667484e1c51a2000642ae08ec9569

Deterministic configuration:

runtime_mode="off"
deterministic_metadata=True
seed=12345
Structural Guarantees

Baseline decoder invariants are preserved across releases:

Flooding schedule unchanged

Layered schedule unchanged

_bp_postprocess() unchanged

No stochastic elements

No hidden randomness

No schema drift

No identity/hash drift for baseline decoders

All releases require the full test suite to pass.

Design Philosophy

Small is beautiful.
Determinism is holy.
Stability is engineered.

Negative results are data.

Author

Trent Slade
QSOL-IMC

ORCID
https://orcid.org/0009-0002-4515-9237

---

# Project State & Documentation

Additional project documentation:

- **PROJECT_STATE.md** — current architectural snapshot of the system
- **ROADMAP.md** — long-term research direction and architectural governance
- **CHANGELOG.md** — full release history and version notes

These documents provide the canonical reference for the system architecture, research trajectory, and deterministic guarantees of QEC.
