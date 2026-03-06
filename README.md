# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v5.2.0](https://img.shields.io/badge/release-v5.2.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v5.2.0)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

Deterministic QLDPC CSS framework for studying belief propagation attractor geometry and decoding dynamics.

QEC is a **deterministic QLDPC CSS quantum error correction framework** for studying belief propagation decoding dynamics under controlled experimental conditions.

The toolkit provides invariant-safe algebraic code construction, multi-mode belief propagation decoding, deterministic postprocessing, pluggable channel models, and reproducible FER / distance performance scaling (DPS) benchmarking.

The framework is designed for **controlled decoder experimentation under strict determinism guarantees**.

---

## Core Goals

The system is engineered for:

- **Reproducibility** — byte-identical results across runs  
- **Interpretability** — explicit algorithmic behavior  
- **Structural experimentation** — controlled topology and inference geometry changes  
- **Deterministic benchmarking** — stable FER and distance-scaling measurements  

> If a result cannot be reproduced byte-for-byte, it is not considered a baseline.

---

## Documentation

Key project documentation:

- `PROJECT_STATE.md` — current system architecture snapshot  
- `ROADMAP.md` — research direction and upcoming work  
- `CHANGELOG.md` — full release history  

---

Current Release

v5.2.0 — Decoder Experiment Framework & Deterministic Paired Experiments

This release introduces the Decoder Experiment Framework, enabling controlled
A/B experiments between a frozen reference BP decoder and an experimental sandbox
decoder.

The goal is to allow safe exploration of decoder modifications while preserving
a permanent deterministic baseline for reproducible experiments.

A new decoder interface layer allows the benchmark harness to dynamically select
between decoder implementations without modifying the decoding API.

Release:
https://github.com/QSOLKCB/QEC/releases/tag/v5.2.0

Decoder Experiment Framework (v5.2)

Two decoder roles are now defined:

src/qec/decoder/
bp_decoder_reference.py
bp_decoder_experimental.py
decoder_interface.py

Reference Decoder

The reference decoder re-exports the original bp_decode implementation and
serves as the immutable baseline implementation.

This module must never be modified, ensuring historical experiments remain
reproducible.

Experimental Decoder

The experimental decoder provides a safe sandbox for testing decoder
modifications without affecting the baseline.

Both decoders expose the same API and can be selected dynamically by the
benchmark harness.

Deterministic Decoder Comparison

The benchmark harness now supports controlled decoder comparison experiments.

New options:

--decoder {reference, experimental}
--compare-decoders

Comparison mode runs both decoders on identical inputs during each trial,
allowing deterministic A/B experiments.

Comparison results are recorded under:

decoder_comparison
Deterministic Paired Experiments

Two additional options guarantee strict pairing of error realizations:

--paired-seed
--paired-errors

These ensure that both decoders observe identical channel error patterns
throughout the FER sweep.

This enables scientifically rigorous decoder comparisons.

Decoder Comparison Report

A convenience reporting option is now available:

--decoder-report

When comparison mode is active, the harness prints a summary table:

Decoder Comparison Report
-----------------------------------------------------------
mode                     distance         p   FER_ref   FER_exp      ΔFER
-----------------------------------------------------------
baseline                       32    0.0300    0.41      0.29     -0.12
rpc_only                       32    0.0300    0.53      0.34     -0.19

This provides immediate visibility into performance differences between
decoder implementations.

BP Attractor Geometry Diagnostics

The toolkit includes a layered deterministic diagnostics framework
for analyzing BP convergence behavior.

Version	Capability
v4.1	Basin-switch detection
v4.2	Energy landscape diagnostics
v4.3	Iteration trajectory diagnostics
v4.4	BP regime classification
v4.5	Regime transition tracing
v4.6	Phase diagram aggregation
v4.7	Freeze detection diagnostics
v4.8	Fixed-point trap analysis
v4.9	Basin-of-attraction estimation
v5.0	Attractor landscape mapping
v5.1	Free-energy barrier estimation
v5.2	Decoder experiment framework

Together these layers provide deterministic measurement of:

BP convergence regimes

attractor basin geometry

incorrect fixed-point probability

escape barrier heights

pseudocodeword boundary structure

decoder intervention effects

All diagnostics remain trace-only and opt-in, preserving deterministic
decoding guarantees.

Baseline decoding outputs remain byte-identical when diagnostics are disabled.

Free-Energy Barrier Estimation (v5.1)

v5.1 introduced deterministic escape-barrier estimation for BP attractors.

New module:

src/qec/diagnostics/bp_barrier_analysis.py

The diagnostic applies deterministic perturbations to the initial belief state
and determines the minimum perturbation magnitude required to escape the
current BP attractor basin.

Perturbation schedule:

eps = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
patterns = [+1, −1, +2, −2]

Returned metrics:

baseline_attractor
barrier_eps
escaped
num_trials

These measurements approximate free-energy barriers surrounding BP attractors.

Benchmark Harness Integration

The deterministic DPS evaluation harness supports the full diagnostics stack:

--bp-dynamics
--bp-transitions
--bp-phase-diagram
--bp-freeze-detection
--bp-fixed-point-analysis
--bp-basin-analysis
--bp-landscape-map
--bp-barrier-analysis
--compare-decoders
--paired-seed
--paired-errors

Diagnostics results are appended to benchmark artifacts without modifying
decoder behavior.

All diagnostics remain fully optional.

Determinism Guarantees

The QEC framework maintains strict deterministic execution:

no randomness introduced

no Python hash() usage

deterministic perturbation schedules

JSON-serializable artifacts

byte-identical results across repeated runs

Decoder outputs remain identical when diagnostics are disabled.

---

## Research Context

These diagnostics enable systematic investigation of **Distance Performance Scaling (DPS)** behavior under syndrome-only inference.

Experiments performed using the diagnostics stack show that BP decoding failures arise from **incorrect attractor selection** rather than metastable oscillation.

The toolkit therefore functions as a deterministic experimental platform for studying:

- BP attractor geometry
- basin stability
- pseudocodeword structure
- decoding landscape topology

in QLDPC Tanner graphs.

v4.4.0 — Deterministic BP Dynamics Regime Analysis

The v4.4.0 release introduces deterministic diagnostics for analyzing the global dynamical behavior of belief propagation (BP) on QLDPC Tanner graphs.

This release completes the first full BP observability stack in the QEC toolkit, enabling systematic analysis of decoder dynamics such as oscillatory convergence, metastability, trapping-set behavior, and chaotic inference regimes.

All diagnostics remain trace-only and opt-in, preserving the deterministic decoding guarantees of the framework.

Baseline decoding outputs remain byte-identical when diagnostics are disabled.

Release:
https://github.com/QSOLKCB/QEC/releases/tag/v4.4.0

New in v4.4.0
BP Dynamics Metric Suite

The new diagnostics module computes deterministic metrics derived from BP iteration traces:

Metric	Description
MSI	Metastability Index — detects plateau behavior with residual oscillations
CPI	Cycle Periodicity Index — deterministic detection of repeating BP states
TSL	Trapping Set Likelihood — persistent sign disagreement indicator
LEC	Local Energy Curvature — second-difference structure of the energy trace
CVNE	Correction-Vector Norm Entropy (optional)
GOS	Global Oscillation Score — aggregate oscillation signal
EDS	Energy Descent Smoothness — monotonicity of BP descent
BTI	Basin Transition Indicator — barrier crossings and state transitions

These metrics provide a deterministic view of the energy landscape and dynamical structure of BP decoding.

Deterministic Regime Classification

BP trajectories are classified into six deterministic regimes:

stable_convergence

oscillatory_convergence

metastable_state

trapping_set_regime

correction_cycling

chaotic_behavior

Each classification includes explicit metric evidence and threshold comparisons for reproducibility and interpretability.

Bench Harness Integration

The DPS benchmark harness now supports:

--bp-dynamics

When enabled:

BP traces are analyzed using the new metric suite

Results are appended to benchmark artifacts under

bp_dynamics

This feature is fully optional and does not alter baseline decoding behavior.

Determinism Guarantees

The v4.4 diagnostics layer preserves the core determinism guarantees of the QEC framework:

No randomness introduced

No Python hash() usage

Deterministic CRC32 state signatures

JSON-serializable outputs

Byte-identical artifacts across repeated runs

Determinism validation confirmed identical SHA-256 outputs across multiple benchmark executions.

Diagnostics Stack

The QEC toolkit now includes a layered diagnostics framework for studying BP dynamics:

Version	Capability
v4.1	Basin-switch detection
v4.2	Energy landscape diagnostics
v4.3	Iteration trajectory diagnostics
v4.4	BP regime classification

These tools enable systematic investigation of Distance Performance Scaling (DPS) inversion under syndrome-only inference.

v4.3.0 — Deterministic Iteration-Trace Diagnostics

v4.3.0 extends the deterministic diagnostics framework with a new layer of
**iteration-trace analysis for belief propagation decoding**.

Where the v4.2 series characterizes the *energy landscape of decoder attractors*,
v4.3.0 introduces metrics that analyze **the internal dynamics of BP during
the decoding trajectory itself**.

These diagnostics operate entirely outside the decoder core and preserve the
architectural guarantee that decoder outputs remain byte-identical when
diagnostics are disabled.

Iteration-Trace Diagnostics (v4.3.0)

The new diagnostics module analyzes per-iteration belief trajectories and
energy evolution produced during BP decoding.

Five deterministic metrics characterize decoder dynamics:

Belief Oscillation Index (BOI)

Measures the rate at which variable beliefs change sign between iterations.

High BOI indicates oscillatory BP regimes or unstable message passing.

Zero values are treated as non-negative to avoid artificial flip counts.

Energy Plateau Index (EPI)

Measures the length of low-gradient segments in the energy trajectory.

High EPI indicates plateau behavior where the decoder stalls in shallow
regions of the free-energy landscape.

Trapping Set Persistence (TSP)

Measures how long the decoder remains in a constraint-consistent but
incorrect configuration.

High persistence indicates classical trapping-set behavior.

Correction Vector Fluctuation (CVF)

Measures cycling behavior in correction vectors across iterations.

This metric is computed only when correction vectors are available and is
reported as `None` otherwise to preserve semantic correctness.

Composite Iteration Stability Score

Combines the above metrics into a deterministic stability descriptor that
characterizes the overall dynamical regime of the decoder trajectory.

Architectural Safety

All iteration diagnostics are strictly observational and never modify
decoder behavior.

The following invariants remain enforced:

decoder core unchanged  
message-passing semantics unchanged  
no randomness introduced  
schema unchanged  
baseline decoder outputs remain byte-identical

Version Lineage

v4.1.0 — Improved Basin Switch Detection  
v4.2.0 — Deterministic Landscape Metrics  
v4.2.1 — Diagnostics Refactor and Test Hardening  
v4.3.0 — Iteration-Trace Diagnostics

v4.2.1 — Diagnostics Refactor and Test Hardening

The v4.2 series extends the deterministic perturbation diagnostics introduced in v4.1.0 with quantitative energy-landscape metrics for BP decoding experiments.

These diagnostics characterize the stability and geometry of decoding attractors under small deterministic perturbations while preserving the architectural guarantee that decoder outputs remain bit-identical when diagnostics are disabled.

v4.2.1 further refines the diagnostics layer with internal optimizations and expanded test coverage.

Landscape Metrics (v4.2.0)

The diagnostics framework now computes three deterministic metrics describing the local decoding landscape.

Basin Stability Index (BSI)
Measures how often small perturbations return the decoder to the same attractor basin.

BSI = (# perturbations yielding baseline correction) / (# perturbations)

Values range from 0.0 (unstable) to 1.0 (fully stable).

Attractor Distance (AD)
Measures the Hamming distance between baseline and perturbed decoding corrections.

Reported values:

attractor_distance_max
attractor_distance_mean

These quantify how far nearby attractors lie in correction space.

Escape Energy (EE)
Estimates the minimum perturbation magnitude required to leave the current basin.

The deterministic epsilon sweep is:

[1e-3, 2e-3, 5e-3, 1e-2]

Directional escape barriers are reported:

escape_energy_plus
escape_energy_minus
escape_energy

The final escape energy is the minimum directional barrier.

v4.2.1 Improvements

v4.2.1 introduces several internal improvements to the diagnostics system:

shared perturbation decode helper to eliminate redundant BP decodes

configurable epsilon sweep (eps_values) for escape-energy diagnostics

strengthened integration tests to guarantee landscape metrics are exercised

additional test coverage for configurable epsilon schedules

These changes do not modify decoder behavior and preserve full determinism.

Decoder Safety

All diagnostics operate outside the decoder core.

The following invariants remain enforced:

decoder message passing unchanged

scheduling semantics unchanged

schema unchanged

deterministic execution preserved

baseline decoding outputs remain byte-identical

Version Lineage
v4.1.0 — Improved Basin Switch Detection
v4.2.0 — Deterministic Landscape Metrics
v4.2.1 — Diagnostics Refactor and Test Hardening
Free-Energy Landscape Diagnostics

Landscape diagnostics enable systematic experimentation with BP convergence regimes, including:

metastable oscillation

shallow perturbation sensitivity

true basin switching

attractor geometry and basin stability

perturbation-induced barrier crossings

v4.1.0 — Improved Basin Switch Detection

v4.1.0 strengthens deterministic perturbation diagnostics by distinguishing metastable oscillation, shallow sensitivity, and true basin switching.

This extends the structural experimentation framework introduced in v3.8–v3.9 and enables systematic study of BP convergence regimes, including plateau behavior, barrier crossings, and geometry-induced basin switching.

Decoder behavior remains bit-identical when diagnostics are disabled.

Channel Geometry Interventions

Two deterministic inference-geometry interventions are available.

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

Free-Energy Landscape Diagnostics

v4.0.0 introduces a diagnostics module for analyzing belief propagation energy dynamics.

src/qec/diagnostics/energy_landscape.py

The module provides deterministic analysis of BP energy traces:

compute_energy_gradient

compute_energy_curvature

detect_plateau

detect_local_minima

detect_barrier_crossings

classify_energy_landscape

detect_basin_switch

Energy is measured per BP iteration:

E = − Σ (LLR_i · belief_i)

These diagnostics allow the framework to study decoder convergence regimes, including:

plateau dynamics

metastable behavior

barrier crossings

geometry-induced basin switching

All diagnostics are strictly observational and do not alter decoding behavior.

Basin Switching Diagnostics

A deterministic perturbation experiment detects when BP converges to different free-energy basins.

A small perturbation is applied to the LLR vector:

llr_perturbed = llr + ε · sign(llr)
ε = 1e-3

If the perturbed decode converges to a different correction or final energy, the trial is classified as a basin switch.

The perturbation is deterministic and handles sign(0) safely to ensure stable results.

DPS Harness Expansion

The deterministic evaluation harness supports geometry-intervention modes and optional energy diagnostics.

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

Example Diagnostics Run
PYTHONPATH=. python bench/dps_v381_eval.py \
  --landscape \
  --trials 200 \
  --distances 5 7 \
  --p-values 0.03

Example observations:

basin switching observed in RPC + centered field modes

stable plateau dynamics in non-RPC geometry modes

deterministic behavior preserved across runs

Stability Guarantees

v4.0.0 maintains strict determinism:

decoder core unchanged

no stochastic components introduced

diagnostics operate only on decoder outputs

identical results when diagnostics are disabled

Full test suite:

945 passed
7 skipped

Previous Releases
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
