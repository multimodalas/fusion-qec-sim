PROJECT_STATE.md

QSOLKCB / QEC — Deterministic QLDPC CSS Toolkit
Current Project State Snapshot

Author: Trent Slade
Organization: QSOL-IMC
Philosophy: Determinism-first engineering and structural decoding research

Purpose of This File

This document provides a quick architectural snapshot of the project.

It exists to help:

• new contributors
• future research collaborators
• automated agents (ChatGPT / Claude / tools)
• future versions of the author

quickly understand:

• what the project currently is
• what problems it is solving
• what invariants must not be broken
• where the research direction is heading

This file should be updated whenever a major release changes the research state of the system.

Authoritative Project Documents

The following files define the canonical documentation for the QEC system:

• README.md — project overview and entry point  
• PROJECT_STATE.md — current architecture and research state  
• ROADMAP.md — long-term research direction and feature planning  
• CHANGELOG.md — release history and version evolution  

Interpretation guidance:

PROJECT_STATE.md reflects the **current architectural state** of the system.  
ROADMAP.md describes the **future research direction**.  
CHANGELOG.md records **what has already been implemented**.

Contributors and automated agents should consult these documents before proposing architectural changes.

Current Stable Version

v4.3.0

Release theme:

Deterministic Decoder Dynamics Diagnostics
+
Free-Energy Landscape Analysis

v4.x establishes a deterministic framework for studying belief propagation (BP) decoding dynamics in QLDPC codes.

The toolkit now supports systematic analysis of:

• decoder attractor basins
• oscillatory BP regimes
• metastable states
• trapping sets
• convergence instabilities

under controlled deterministic experiments.

Core System

QEC is a deterministic QLDPC CSS quantum error correction toolkit designed for controlled experimentation on decoding algorithms.

The framework provides:

• deterministic QLDPC construction
• belief propagation decoding variants
• deterministic postprocessing methods
• structural intervention mechanisms
• deterministic benchmarking infrastructure
• deterministic decoder dynamics diagnostics

The system is designed to enable reproducible decoder research.

Architectural Layers

The system evolves in strictly separated layers.

Layer 1 — Decoder Core
Belief propagation + deterministic postprocessing

Layer 2 — Channel Models
Deterministic LLR construction

Layer 3 — Structural Interventions
Geometry / constraint modifications

Layer 4 — Benchmark & Diagnostics
FER / DPS / energy / trajectory analysis

Interop Layer — Schema + artifact hashing

Each layer may expand.

Lower layers must not be destabilized by higher layers.

Decoder Core Capabilities

Supported BP algorithms:

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

Deterministic postprocessing:

osd1
osd_cs
mp_osd1
mp_osd_cs
guided_decimation

All decoding operations are deterministic.

Channel Models

Channel models generate deterministic LLR vectors.

Implemented models:

oracle
bsc_syndrome

Future candidates:

AWGN
erasure
stim-compatible noise models

Channel models must not mutate decoder logic.

Structural Interventions (v3.8 → v3.9)

Structural interventions allow controlled modification of decoding geometry without altering baseline behavior.

Implemented interventions:

RPC Augmentation

Deterministic redundant parity check generation.

Properties:

• deterministic row-pair XOR
• feasible set unchanged
• opt-in

geom_v1 Schedule

Flooding BP with deterministic constraint scaling.

α_c = 1 / sqrt(deg_check)

Purpose:

stabilize heterogeneous Tanner graph structure.

Centered Syndrome Field (v3.9)

Removes global syndrome bias before projection.

b = 1 − 2s
b_centered = b − mean(b)

LLR = Hᵀ b_centered

Purpose:

restore directional likelihood structure.

Pseudo-Prior Injection (v3.9)

Injects deterministic variable prior derived from parity structure.

LLR_i ← LLR_i + κ · parity_bias_i

Default:

κ = 0.25

Purpose:

break likelihood symmetry under syndrome-only channels.

Diagnostics Infrastructure

The framework contains deterministic diagnostic tools for analyzing decoder behavior.

DPS Harness

Introduced in v3.8.1.

Measures:

Distance Performance Scaling (DPS)

Using deterministic instance reuse.

Harness modes include:

baseline
rpc_only
geom_v1_only
rpc_geom
centered
prior
centered_prior
geom_centered
geom_centered_prior
rpc_centered
rpc_centered_prior

Energy Landscape Diagnostics (v4.1 → v4.2)

The system now supports deterministic analysis of the free-energy landscape of BP decoding.

Diagnostics include:

Basin Switch Classification
Basin Stability Index (BSI)
Attractor Distance (AD)
Escape Energy (EE)

These metrics characterize:

• basin transitions
• decoder stability under perturbation
• attractor geometry

Iteration-Trace Diagnostics (v4.3)

v4.3 introduces deterministic analysis of decoder iteration dynamics.

Metrics include:

Persistent Error Indicator (PEI)
Belief Oscillation Index (BOI)
Oscillation Depth (OD)
Convergence Instability Score (CIS)
Correction Vector Fluctuation (CVF)

These diagnostics detect:

• trapping sets
• oscillatory message passing
• metastable convergence
• correction cycling

Diagnostics operate purely on iteration traces and do not modify the BP algorithm.

Current Experimental Observation

Under:

channel_model = "bsc_syndrome"

the decoder experiences distance scaling inversion.

Observed behavior:

• FER increases with code distance
• BP converges to incorrect constraint-consistent states
• likelihood landscape collapses

Evidence across releases:

Release	Intervention	Result
v3.6	posterior-aware OSD	no correction
v3.7	URW reweighting	no correction
v3.8	RPC + geom scaling	no correction
v3.9	centered field + pseudo-prior	partial improvement

Conclusion:

The problem appears information-geometric rather than algorithmic.

Research Objective

Identify the first deterministic intervention that produces:

DPS < 0

under syndrome-only inference.

Meaning:

Frame error rate decreases with distance.

Immediate Research Direction

Active exploration areas:

• geometry-aware inference fields
• constraint density amplification
• geometry-aware BP schedules
• BP initialization strategies
• free-energy landscape diagnostics
• iteration-trace regime classification

All interventions must remain:

deterministic
opt-in
baseline-preserving

Medium-Term Direction (v4)

v4 focuses on decoder regime analysis.

Primary objective:

Map the BP free-energy landscape and trajectory dynamics of QLDPC codes.

Tools now available:

• energy trajectory analysis
• attractor basin detection
• regime classification
• oscillation diagnostics
• metastability detection

Test Suite Status

1042 tests passing
7 skipped
0 failures

Baseline decoder outputs remain unchanged when all interventions are disabled.

Architectural Invariants

The following must never change without a major version bump:

BP flooding loop semantics
BP layered loop semantics
_bp_postprocess() behavior
baseline decoder outputs
deterministic artifact generation
schema compatibility

All research features must be opt-in.

Determinism Anchor

Deterministic configuration:

runtime_mode="off"
deterministic_metadata=True
seed=12345

Schema versions:

SCHEMA_VERSION = 3.0.1
INTEROP_SCHEMA_VERSION = 3.1.2

If it cannot be reproduced byte-for-byte, it is not a baseline.

Project Philosophy

Small is beautiful.
Determinism is holy.
Stability is engineered.

Negative results are data.

Author

Trent Slade
QSOL-IMC

ORCID
https://orcid.org/0009-0002-4515-9237
