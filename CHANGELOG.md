# Changelog

All notable changes to this project are documented in this file.

This project follows Semantic Versioning (SemVer).

---

[4.3.0] — 2026-03-05
Deterministic Iteration-Trace Diagnostics

Adds iteration-trace diagnostics that analyse BP iteration logs to detect
trapping sets, oscillatory message passing, unstable convergence, and
correction vector cycling.  Diagnostics operate purely on iteration traces
and do not modify the BP decoder.

Added

- `compute_persistent_error_indicator()` (PEI): flags variable nodes whose
  LLR sign indicates an error for a configurable number of consecutive
  iterations.  Detects trapping sets.
- `compute_belief_oscillation_index()` (BOI): counts LLR sign flips across
  iterations for each variable node.  Measures oscillatory message passing.
- `compute_oscillation_depth()` (OD): measures peak-to-peak LLR amplitude
  over a trailing window.  Quantifies oscillation severity.
- `compute_convergence_instability_score()` (CIS): variance of the energy
  trace over a trailing window.  Detects unstable convergence.
- `compute_correction_vector_fluctuation()` (CVF): Euclidean norm of
  consecutive correction vector differences.  Detects correction cycling.
- `compute_iteration_trace_metrics()`: composite function returning all
  five metrics in a single call.
- DPS harness (`bench/dps_v381_eval.py`): new `--iteration-diagnostics`
  flag.  When enabled, computes iteration-trace metrics per trial and
  stores them under `result["iteration_diagnostics"]`.
- Comprehensive tests (`tests/test_iteration_trace_metrics.py`):
  determinism, no-input-mutation, oscillation detection, stable
  convergence, trapping set detection, correction cycling, and composite
  metric validation.

Unchanged

- Decoder core: no modifications to BP loops, scheduling, or iteration.
- Schema: no SCHEMA_VERSION bump.
- Canonical serialization, hashing, and identity: unchanged.
- Baseline decoding outputs: byte-identical under identical inputs.
- All existing harness output fields remain present and unchanged.

---

[4.2.1] — 2026-03-05
Diagnostics Refactor and Test Hardening

Patch release improving the diagnostics implementation introduced in
v4.2.0.  No decoder core changes.  Deterministic outputs preserved.

Changed

- Eliminates redundant BP decodes by sharing perturbation results
  via new internal helper `_run_perturbation_decodes()`.
  `classify_basin_switch()` and `compute_landscape_metrics()` now
  reuse a single set of ±epsilon decode results.
- Allows configurable epsilon sweep for escape-energy diagnostics:
  `compute_escape_energy()` accepts optional `eps_values` parameter.
  Default behavior remains identical to v4.2.0.
- Strengthens landscape integration test to guarantee metric
  validation: conditional guards replaced with explicit assertions
  ensuring `basin_classifications` and all landscape metric fields
  are always verified.

Unchanged

- Decoder core: no modifications to BP loops, scheduling, or iteration.
- Schema: no SCHEMA_VERSION bump.
- Canonical serialization, hashing, and identity: unchanged.
- Baseline decoding outputs: byte-identical under identical inputs.
- Classification logic and metric outputs: identical to v4.2.0.

---

[4.2.0] — 2026-03-05
Deterministic Landscape Metrics

Adds Basin Stability Index (BSI), Attractor Distance (AD), and Escape
Energy (EE) to energy landscape diagnostics.  Escape energy includes
directional barrier estimation inspired by spin-glass energy landscape
analysis.

This is a diagnostic-only extension.  No decoder core modifications.

Added

- `compute_basin_stability_index()`: ratio of perturbations yielding
  same correction as baseline.  Values in [0.0, 1.0].
- `compute_attractor_distance()`: Hamming distance between baseline and
  perturbed corrections.  Returns max and mean.
- `compute_escape_energy()`: deterministic epsilon sweep to find minimum
  perturbation causing a basin switch.  Probes +epsilon and -epsilon
  independently, returning directional and minimum barriers.
- `compute_landscape_metrics()`: composite function combining v4.1.0
  classification with BSI, AD, and EE in a single output dict.
- `_hamming_distance()`: deterministic Hamming distance helper.
- DPS harness (`bench/dps_v381_eval.py`) now emits landscape metrics
  (BSI, AD, EE) alongside basin classifications when `--landscape`
  mode is enabled.
- Comprehensive tests: determinism, baseline safety, Hamming distance
  correctness, escape energy sweep, and harness integration.

Unchanged

- Decoder core: no modifications to BP loops, scheduling, or iteration.
- Schema: no SCHEMA_VERSION bump.
- Canonical serialization, hashing, and identity: unchanged.
- Baseline decoding outputs: byte-identical under identical inputs.
- All existing harness output fields remain present and unchanged.
- `classify_basin_switch()` remains available and unmodified.

---

[4.1.0] — 2026-03-05
Improved Basin Switch Detection

Strengthens deterministic perturbation diagnostics by introducing a
three-regime classifier that distinguishes between metastable oscillation,
shallow perturbation sensitivity, and true basin switching.

This is a diagnostic-only improvement.  No decoder core modifications.

Added

- `classify_basin_switch()` in `src/qec/diagnostics/energy_landscape.py`:
  performs three deterministic decodes (baseline, +epsilon, -epsilon)
  and classifies the result as `metastable_oscillation`,
  `shallow_sensitivity`, `true_basin_switch`, or `none`.
- Helper functions `_count_gradient_sign_flips()` and
  `_trace_converged()` for trace analysis.
- DPS harness (`bench/dps_v381_eval.py`) now emits `basin_classifications`
  and `basin_class_counts` when `--landscape` mode is enabled.
- Comprehensive tests: determinism, baseline safety, classification
  coverage, and harness integration.

Unchanged

- Decoder core: no modifications to BP loops, scheduling, or iteration.
- Schema: no SCHEMA_VERSION bump.
- Canonical serialization, hashing, and identity: unchanged.
- Baseline decoding outputs: byte-identical under identical inputs.
- All existing harness output fields remain present and unchanged.

---

[4.0.0] — 2026-03-05
BP Free-Energy Landscape Diagnostics

Introduces a deterministic diagnostics layer for analyzing belief
propagation (BP) energy dynamics during QLDPC decoding.

This release extends the deterministic benchmarking framework with
tools for studying decoder convergence regimes, including plateau
behavior, barrier crossings, and geometry-induced basin switching.

All diagnostics are strictly observational and do not modify decoding
behavior.

Added

Energy Landscape Diagnostics Module

New module:

src/qec/diagnostics/energy_landscape.py

Provides deterministic analysis utilities for BP energy traces:

compute_energy_gradient

compute_energy_curvature

detect_plateau

detect_local_minima

detect_barrier_crossings

classify_energy_landscape

detect_basin_switch

Energy is evaluated per BP iteration:

E = − Σ (LLR_i · belief_i)

These diagnostics enable systematic analysis of BP convergence behavior.

Basin Switching Detector

Introduces a deterministic perturbation experiment to detect
free-energy basin switching in BP decoding.

A small perturbation is applied to the LLR vector:

llr_perturbed = llr + ε · sign(llr)
ε = 1e-3

If the perturbed decode converges to a different correction or final
energy, the trial is classified as a basin switch.

The perturbation is deterministic and safely handles sign(0).

DPS Harness Landscape Mode

The deterministic DPS evaluation harness now supports energy landscape
diagnostics via a CLI flag:

--landscape

When enabled the harness records:

per-iteration BP energy traces

landscape classification statistics

basin switching frequency per mode

Example run:

PYTHONPATH=. python bench/dps_v381_eval.py \
  --landscape \
  --trials 200 \
  --distances 5 7 \
  --p-values 0.03

All modes reuse identical deterministic error instances.

Improvements

Reduced Diagnostic Overhead

The basin-switch detector now reuses the existing decode result
from the harness instead of performing a redundant baseline decode.

This significantly reduces diagnostic runtime when landscape analysis
is enabled.

Fixes

Geometry Postprocessing Consistency

Fixed an issue where geometry postprocessing could be applied
asymmetrically between baseline and perturbed decodes in the basin
switch detector.

Both decodes now operate in the same LLR domain, differing only by the
deterministic perturbation.

Perturbation Stability

Improved numerical stability by ensuring deterministic perturbation
behavior when LLR == 0.

Tests

New test suite:

tests/test_energy_landscape.py

Coverage includes:

gradient and curvature computation

plateau detection

barrier detection

basin switch detection

deterministic perturbation behavior

Results

Full test suite:

945 passed
7 skipped
0 failed

Deterministic reproducibility verified.

Guarantees

No changes to core BP decoding logic

No changes to BP scheduling or message updates

Diagnostics operate only on decoder outputs

No dependency changes

Baseline decoder outputs remain byte-identical when diagnostics disabled

All new features are opt-in

Determinism preserved

---

## [3.9.1] — 2026-03-04

### Geometry Field Controls

Introduces deterministic geometry field controls for controlled testing
of likelihood magnitude effects under syndrome-only inference.

All new features are opt-in. Baseline decoder behavior remains unchanged
when features are disabled.

### Added

**Geometry Strength Scaling**

- `geometry_strength` field on `StructuralConfig` (default: `1.0`)
- Scales the constructed geometry LLR field after centered_field and
  pseudo_prior are applied
- Applied in both `BPAdapter` and DPS harness

**Deterministic Field Normalization**

- `normalize_geometry` field on `StructuralConfig` (default: `False`)
- When enabled: `llr = llr / (std(llr) + 1e-12)`
- Ensures the LLR distribution has unit variance
- Only applies when geometry interventions are active
- Normalization is applied before geometry_strength scaling

**DPS Harness Geometry Sweep Modes**

Three new evaluation modes:

- `centered_strong` — centered field + geometry_strength=2.0
- `centered_normalized` — centered field + normalize_geometry
- `centered_prior_normalized` — centered + prior + normalize_geometry

All modes reuse identical deterministic error instances.

### Tests

New test suite:

    tests/test_geometry_controls.py

Coverage includes:

- geometry_strength scaling determinism and correctness
- normalize_geometry unit-variance verification
- baseline invariance when features disabled
- DPS harness new mode execution and determinism

### Results

Full test suite:

    923 passed
    7 skipped
    0 failed

Deterministic reproducibility verified.

### Guarantees

- No changes to core decoding logic
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes
- No dependency changes
- Baseline decoder outputs remain byte-identical when features disabled
- All new features are opt-in with safe defaults
- Determinism preserved

---

[3.9.0] — 2026-03-04
Channel Geometry Interventions & BP Energy Diagnostics

Introduces deterministic channel-geometry interventions and belief propagation energy diagnostics for structural decoding experiments under syndrome-only inference.

This release expands the deterministic experimentation framework introduced in v3.8.x.

Baseline decoder behavior remains unchanged when all structural features are disabled.

Added

Channel geometry utilities:

src/qec/channel/geometry.py

Deterministic functions:

syndrome_field()

centered_syndrome_field()

pseudo_prior_bias()

apply_pseudo_prior()

These construct LLRs directly from syndrome structure for oracle-free decoding experiments.

Belief propagation energy diagnostics:

src/qec/decoder/energy.py

Provides optional per-iteration energy tracing:

E = − Σ (LLR_i · belief_i)

Energy tracing enables analysis of:

BP convergence dynamics

oscillatory decoding behavior

likelihood alignment during inference

Energy tracing is purely diagnostic and does not alter decoder outputs.

DPS Harness Expansion

The deterministic evaluation harness now includes geometry-intervention modes.

New modes:

centered

prior

centered_prior

geom_centered

geom_centered_prior

rpc_centered

rpc_centered_prior

All modes reuse identical deterministic error instances.

Baseline evaluation behavior remains unchanged.

Stability Improvements

Resolved issues identified during code review:

stabilized bp_decode() return structure across optional diagnostics

ensured consistent tuple ordering when energy_trace is enabled

added epsilon threshold to DPS sign detection to avoid floating-point noise

strengthened baseline invariance testing

Tests

New test suites:

tests/test_channel_geometry.py
tests/test_energy_trace.py

Coverage includes:

deterministic geometry field construction

pseudo-prior application

BP energy trace correctness

return-structure validation across feature combinations

baseline decoder invariance

Results

Full test suite:

904 passed
0 failed

Deterministic reproducibility verified.

## [3.8.1] — 2026-03-03

### Structural Geometry Evaluation Harness

Adds a deterministic evaluation harness for analyzing structural
decoder interventions introduced in v3.8.0.

This release introduces **measurement infrastructure only**.
No decoder algorithms were modified.

### Added

Deterministic DPS evaluation harness:


bench/dps_v381_eval.py


Capabilities:

- deterministic RNG (`seed = 42`)
- pre-generated error instances reused across modes
- four evaluation modes:
  - baseline
  - rpc_only
  - geom_v1_only
  - rpc_geom
- activation audit reporting:
  - original_rows
  - augmented_rows
  - added_rows
  - H checksum
  - syndrome checksum
  - iteration count
- deterministic slope estimation for DPS
- inversion detection marker
- determinism verification run

Frame error rate uses **syndrome-consistency semantics**:


syndrome(H, correction) != s


### Tests

New harness validation suite:


tests/test_dps_v381_harness.py


Coverage includes:

- deterministic instance reuse
- RPC activation verification
- schedule dispatch validation
- DPS slope computation
- decoder invariance confirmation

### Guarantees

- No decoder algorithm changes
- `bp_decode()` unchanged
- All BP schedules unchanged
- `_bp_postprocess()` unchanged
- No schema changes
- No dependency changes
- Deterministic outputs preserved
- Full test suite passing

---

## [3.8.0] — 2026-03-02

### Structural Geometry Infrastructure

Introduces deterministic infrastructure for controlled experiments on
decoder **topology** and **inference geometry**.

All new features are **strictly opt-in**.

Baseline decoder behavior remains unchanged when disabled.

---

### Added

#### RPC Builder

New deterministic redundant parity-check augmentation module:


src/qec/decoder/rpc.py


Provides:


build_rpc_augmented_system()


Functionality:

- deterministic lexicographic row-pair XOR generation
- redundant parity constraints
- no feasible-set change
- no mutation of original H matrix
- deterministic ordering of generated rows

Configuration objects:


RPCConfig
StructuralConfig


Tests:


tests/test_rpc_builder.py


---

#### `geom_v1` Schedule

Adds a geometry-scaled flooding schedule.


schedule="geom_v1"


Scaling rule:


α_c = 1 / sqrt(d_c)


Where `d_c` is the degree of check node `c`.

Properties:

- flooding-style schedule
- deterministic scaling
- no adaptive behavior
- no stochastic elements

Tests:


tests/test_geom_v1_schedule.py


---

#### Adapter Integration

Structural geometry features integrated into the decoder adapter layer.

File:


src/bench/adapters/bp.py


Behavior:


if structural_config.rpc.enabled:
H_used, s_used = build_rpc_augmented_system(...)
else:
H_used, s_used = H, s


This ensures structural interventions occur **outside the decoder core**.

Tests:


tests/test_adapter_rpc_integration.py


---

### Guarantees

- Flooding schedule unchanged
- Layered schedule unchanged
- Residual schedule unchanged
- `_bp_postprocess()` unchanged
- Decoder iteration logic untouched
- No schema version changes
- No dependency additions
- No stochastic behavior introduced
- Baseline decoder outputs remain bit-identical

---

## [3.7.0] — 2026-03-01

### Uniformly Reweighted BP (URW-BP)

Adds a new opt-in BP mode `mode="min_sum_urw"` that applies a uniform
scalar reweighting factor `urw_rho` to check-to-variable messages,
reducing loop overcounting in loopy Tanner graphs.

The URW reweighting scales each check-to-variable message by a constant
`rho in (0, 1]`:

    R_j→i ← urw_rho * R_j→i

This is algebraically equivalent to `min_sum` when `urw_rho=1.0`.

### Added

- `mode="min_sum_urw"` in `bp_decode()`:
  - Applies `urw_rho` as a uniform scalar multiplier to check-to-variable
    messages in the min-sum update rule
  - Supported on all schedules: flooding, layered, residual,
    hybrid_residual, adaptive
  - Compatible with damping, clipping, llr_history, residual_metrics,
    and all existing postprocessors (osd0, osd1, osd_cs, etc.)
- `urw_rho` parameter in `bp_decode()`:
  - Validated only when `mode="min_sum_urw"`: must satisfy `0 < urw_rho <= 1`
  - Default value `1.0` (no-op for non-URW modes)
- Comprehensive test suite in `tests/test_urw_bp_v370.py`:
  - Baseline invariance across all existing modes and schedules
  - `rho=1.0` bit-identity with `min_sum`
  - Determinism across repeated runs
  - Validation error tests for invalid `urw_rho`
  - Identity inclusion tests via BPAdapter

### Guarantees

- No changes to default decoder behavior
- No changes to baseline decoder identity/hash
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes (SCHEMA_VERSION and INTEROP_SCHEMA_VERSION unchanged)
- No new dependencies
- No scheduling changes
- No randomness introduced
- Determinism verified across repeated runs
- All existing modes unaffected: sum_product, min_sum, norm_min_sum,
  offset_min_sum, improved_norm, improved_offset
- All existing schedules unaffected
- All existing postprocessors unaffected
- All existing tests pass without modification

---

## [3.6.0] — 2026-03-01

### Deterministic Posterior-Aware Combination-Sweep OSD Postprocess

Adds a new opt-in postprocess mode `postprocess="mp_osd_cs"` that uses
posterior LLR magnitude (`abs(L_post)`) instead of channel LLR to order
columns for OSD-CS information-set selection and combination sweep.

This extends the mp_osd1 approach (v3.5.0) from single-bit flip to
multi-bit combination sweep, providing a higher-order posterior-aware
search without altering BP semantics or default behavior.

### Added

- `postprocess="mp_osd_cs"` in `bp_decode()`:
  - Runs inner BP with `postprocess=None` and `llr_history=1` to obtain
    posterior beliefs
  - If BP converges, returns immediately (no OSD needed)
  - Otherwise, applies OSD-CS with reliability ordering based on
    `abs(L_post)` instead of `abs(channel_llr)`
  - Combination sweep depth controlled by existing `osd_cs_lam` parameter
  - Tie-breaking: ascending variable index (deterministic)
  - Never-degrade guarantee: if OSD result fails syndrome, returns BP
    hard decision
- `mp_osd_cs_postprocess()` function in `src/decoder/osd.py`
- Comprehensive test suite in `tests/test_mp_osd_cs.py`

### Guarantees

- No changes to default decoder behavior
- No changes to baseline decoder identity/hash
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes (SCHEMA_VERSION and INTEROP_SCHEMA_VERSION unchanged)
- No new dependencies
- No new parameters (reuses existing `osd_cs_lam`)
- No randomness introduced
- Determinism verified across repeated runs
- Baseline OSD postprocess modes (osd0, osd1, osd_cs) unaffected
- MP-OSD-1 postprocess unaffected
- Guided decimation postprocess unaffected
- All existing tests pass without modification

---

## [3.5.0] — 2026-03-01

### Deterministic MP-Aware OSD-1 Postprocess

Adds a new opt-in postprocess mode `postprocess="mp_osd1"` that uses
posterior LLR magnitude (`abs(L_post)`) instead of channel LLR to order
columns for OSD-1 information-set selection.

This exploits the message-passing information to produce a more informed
reliability ranking, without altering BP semantics or default behavior.

### Added

- `postprocess="mp_osd1"` in `bp_decode()`:
  - Runs inner BP with `postprocess=None` and `llr_history=1` to obtain
    posterior beliefs
  - If BP converges, returns immediately (no OSD needed)
  - Otherwise, applies OSD-1 with reliability ordering based on
    `abs(L_post)` instead of `abs(channel_llr)`
  - Tie-breaking: ascending variable index (deterministic)
  - Never-degrade guarantee: if OSD result fails syndrome, returns BP
    hard decision
- `mp_osd1_postprocess()` function in `src/decoder/osd.py`
- Comprehensive test suite in `tests/test_mp_osd1.py`

### Guarantees

- No changes to default decoder behavior
- No changes to baseline decoder identity/hash
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes (SCHEMA_VERSION and INTEROP_SCHEMA_VERSION unchanged)
- No new dependencies
- No randomness introduced
- Determinism verified across repeated runs
- Baseline OSD postprocess modes (osd0, osd1, osd_cs) unaffected
- Guided decimation postprocess unaffected
- All existing tests pass without modification

---

## [3.4.0] — 2026-03-01

### Deterministic Belief Propagation Guided Decimation

Adds a new opt-in postprocess mode `postprocess="guided_decimation"` that
performs iterative variable freezing guided by BP posterior beliefs.

This is a minimal structural intervention designed to break degeneracy and
trapping behavior in syndrome-only BP decoding, without altering BP
semantics or default scheduling logic.

### Added

- `postprocess="guided_decimation"` in `bp_decode()`:
  - Runs BP for `decimation_inner_iters` per round (up to `decimation_rounds`)
  - After each round, selects the unfrozen variable with maximal |posterior LLR|
  - Ties broken deterministically by lowest variable index
  - Zero-posterior convention: freeze to +decimation_freeze_llr (hard = 0)
  - Freezes the selected variable by clamping its LLR to
    ±decimation_freeze_llr
  - Returns immediately when syndrome is satisfied
  - Non-convergence fallback ranks candidates by
    (syndrome_weight, hamming_weight, round_index) — fully explicit
- Three new parameters (only validated when `postprocess="guided_decimation"`):
  - `decimation_rounds` (default 10)
  - `decimation_inner_iters` (default 10)
  - `decimation_freeze_llr` (default 1000.0)
- `guided_decimation()` function in `src/decoder/decimation.py`
- Comprehensive test suite in `tests/test_guided_decimation.py`

### Guarantees

- No changes to default decoder behavior
- No changes to baseline decoder identity/hash
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes (SCHEMA_VERSION and INTEROP_SCHEMA_VERSION unchanged)
- No new dependencies
- No randomness introduced
- Determinism verified across repeated runs
- Baseline OSD postprocess modes unaffected
- All existing tests pass without modification

### Structural techniques intentionally NOT implemented (out of scope)

- Stabilizer Inactivation
- MP-aware OSD
- Check reweighting
- Sequential CN scheduling variants
- Graph surgery / Tanner graph modification
- Directional LLR bias injection
- Channel model modification
- Learned / neural components
- New schedule families
- Automatic schedule switching

---

## [3.3.1] — 2026-03-01

### v3.3.1 — Geometry Diagnostics Hardening

- SSI grouping hardened to include decoder identity
- DPS regression now fits full-precision log values
- BSI now raises `BSIConfigError` (specific subtype of `ValueError`)
- `compute_bsi` docstring clarifies handling of extra 2x-only records
- README badge corrected

No decoder behavior changes.
No schema changes.
Determinism preserved.

---

## [3.3.0] — 2026-02-28

### Geometry-Aware Syndrome-Only Diagnostics

This release adds a diagnostics-first reporting layer for explaining
distance scaling inversion under `bsc_syndrome` channel inference.

All metrics are computed post-hoc from existing benchmark results.
No decoder behavior changes.  No schema changes.  Canonical benchmark
artifacts are byte-identical when diagnostics are not invoked.

---

### Added

**Geometry Diagnostics Module (`src/bench/geometry_diagnostics.py`)**

- Distance Penalty Slope (DPS): slope of log10(FER + eps) vs distance
  per (decoder, p) group — positive slope indicates inversion
- False-Convergence Rate (FCR): P(syndrome=0 AND logical failure),
  derived algebraically as SCR - Fidelity (equivalent to Inversion
  Index from v3.2.1)
- Budget Sensitivity Index (BSI): FER(base) - FER(2x) for comparing
  iteration budget impact
- Schedule Sensitivity Index (SSI): max(FER) - min(FER) across
  schedules per (distance, p)
- Per-iteration summary computation from existing `llr_history`:
  syndrome_weight[t], check_satisfaction_ratio[t], delta_syndrome[t]
- Aggregate stall metrics (stall fraction, trials with stalls)
- Aggregate residual summaries (mean/max/var of linf, l2, energy)
- Local inconsistency summary (syndrome weight increase events)
- Standalone `collect_per_iteration_data()` using existing opt-in
  `llr_history` and `residual_metrics` decoder parameters
- Sidecar artifact builder with deterministic canonicalized output

**Diagnostic Workflow Support**

- BSI comparison: accepts base and 2x max_iters result sets
- SSI comparison: accepts schedule-keyed result mapping
- Grouping by distance, p, schedule, and channel
- Deterministic config ordering in all aggregated outputs

**Test Coverage**

- Metric correctness tests for all seven diagnostics
- Diagnostics-off baseline byte-identity verification
- Deterministic sidecar serialization tests
- Aggregation order stability tests (input-order independence)
- Per-iteration instrumentation integration tests
- Sidecar rerun byte-identity test

---

### Guarantees

- No changes to core decoding logic
- No changes to scheduling or ensemble behavior
- No default decoder behavior changes
- No channel modifications
- No schema changes
- SCHEMA_VERSION remains `3.0.1`
- INTEROP_SCHEMA_VERSION remains `3.1.2`
- No new external dependencies
- No new randomness sources
- Canonical benchmark artifacts unchanged when diagnostics are not invoked
- All diagnostic outputs emitted as separate sidecar artifacts
- Determinism preserved (`runtime_mode="off"`, `deterministic_metadata=True`, fixed seed)

---

### Test Status

669 passed
7 skipped
0 failed

Geometry-aware syndrome-only diagnostics release.

[3.2.1] — 2026-02-28

Inversion Index Formalization & Structural Channel Diagnostics

This release formalizes the Inversion Index (II) as a deterministic diagnostic metric and completes the structural comparison between oracle and syndrome-only channel models.

This is a report-layer structural formalization release.

No decoder behavior changes.
No channel modifications.
No schema changes.

Added

Inversion Index (II = SCR - Fidelity)

Deterministic derived metric isolating syndrome-consistent but logically incorrect decoding outcomes

Computed algebraically from existing FER and SCR fields

No new stochastic sources

No new artifact generation

No schema expansion required

Cross-Channel Structural Comparative Analysis

Formal comparison between oracle and bsc_syndrome channel regimes

Quantified threshold displacement (~0.48–0.49)

Identified inversion regime under oracle (p > 0.50)

Confirmed absence of inversion regime under syndrome-only

Statistical Noise Bound Formalization

Added theoretical bound for random syndrome matches:

P[random syndrome match] ≈ 2^(−m)

Expected matches ≈ T · 2^(−m)

Demonstrated that observed small II values (~0.002–0.010) are consistent with statistical coincidence at 200–500 trial counts

Closed interpretive gap between stochastic noise and structural inversion

Guarantees

Layer 1 decoder logic unchanged

Channel implementations unchanged

SCHEMA_VERSION remains 3.0.1

INTEROP_SCHEMA_VERSION remains 3.1.2

No dependency expansion

No artifact hash drift

No JSON canonicalization changes

No runner modifications

Inversion Index is derived from existing deterministic fields and inherits all determinism guarantees.

Test Status

629 passed
7 skipped
0 failed

Determinism verified (runtime_mode="off", deterministic_metadata=True, fixed seed).

Structural diagnostic formalization release.

[3.1.4] — 2026-02-26
Channel Architecture Hardening

This release tightens the structural integrity of the channel abstraction layer introduced in v3.1.3.

No scientific behavior changes.
No decoder modifications.
No schema changes.

Changed

Channel Abstraction Layer Hardening

Centralized probability validation in ChannelModel

Introduced shared _EPSILON constant to prevent numeric drift

Relocated channel registry into src/qec/channel/ (Layer 2 ownership)

Removed inline registry from benchmarking runner

Added explicit documentation of oracle default serialization compatibility in config layer

Guarantees

OracleChannel remains byte-identical to v3.1.2 artifacts

BSCSyndromeChannel behavior unchanged from v3.1.3

No decoder core modifications

No scheduling or ensemble changes

No API breaking changes

No dependency expansion

SCHEMA_VERSION remains 3.0.1

INTEROP_SCHEMA_VERSION remains 3.1.2

Determinism preserved (runtime_mode="off", deterministic_metadata=True, fixed seed)

Test Status

629 passed
7 skipped
0 failed

Channel abstraction hardening release.

## [3.1.3] — 2026-02-26

### Syndrome-Only Channel Inference

This release introduces a pluggable channel abstraction layer that eliminates
degenerate 0.0 FER behavior caused by oracle LLR sign leakage.

No decoder core logic was modified.

---

### Added

**Channel Abstraction Layer (`src/qec/channel/`)**

- `ChannelModel` abstract base class with `compute_llr()` interface
- `OracleChannel` — backward-compatible oracle LLR (sign from error vector)
- `BSCSyndromeChannel` — syndrome-only BSC channel (uniform LLR, no sign leakage)
- Channel models are pluggable via `channel_model` config field

**BenchmarkConfig Extension**

- Optional `channel_model` field (default: `"oracle"`)
- Validated against allowed values: `"oracle"`, `"bsc_syndrome"`
- Omitted from serialized config when default — preserves pre-v3.1.3 byte-identity
- Backward-compatible: configs without `channel_model` load as `"oracle"`

**Comprehensive Test Coverage**

- Oracle identity: `OracleChannel` output matches `channel_llr()` exactly
- Oracle benchmark byte-identity: oracle mode produces identical JSON to v3.1.2
- Non-degenerate FER: `bsc_syndrome` produces `0 < FER < 1` at moderate noise
- BSC determinism: two runs with identical config produce byte-identical JSON
- LLR structural: oracle sign depends on error vector; BSC is uniform
- Config backward compatibility: legacy configs without `channel_model` work unchanged

---

### Changed

- Bench runner LLR construction now dispatches through channel model interface
- Interop runner uses `OracleChannel` class (output unchanged)

---

### Guarantees

- No changes to core decoding logic
- No changes to scheduling or ensemble behavior
- No API breaking changes
- No new required dependencies
- SCHEMA_VERSION remains `3.0.1`
- INTEROP_SCHEMA_VERSION remains `3.1.2`
- Oracle mode byte-identical to v3.1.2 artifacts
- Determinism preserved (`runtime_mode="off"`, `deterministic_metadata=True`, fixed seed)

---

### Test Status

629 passed
7 skipped
0 failed

Syndrome-only channel inference release.

## [3.1.2] — 2026-02-26

### Deterministic Interop Baseline & Schema Hardening

This release formalizes the benchmarking / interop layer as a deterministic,
schema-validated baseline suitable for controlled comparative research.

No decoder core logic was modified.

---

### Added

**Deterministic Interop Benchmark Layer (`src/bench/interop/`)**

- Isolated third-party interop namespace
- Strict import hygiene (Stim / PyMatching optional and gated)
- Canonical JSON serialization utilities:
  - `sort_keys=True`
  - compact separators
- Artifact SHA-256 hashing over immutable record state
- Stable sweep hash derived solely from configuration parameters
- Deterministic report generation with stable ordering

**Interop Schema v3.1.2**

- Structured interop record validation
- Required determinism block:
  - canonical JSON configuration
  - stable_sweep_hash (64-hex validated)
  - artifact_hash (64-hex validated)
- `mean_iters` required for `direct_comparison` records
- Structured skipped-record validation:
  - `reason` (str)
  - `tool.name` (str)
  - `benchmark_kind` (str)
  - `code_family` (str)

**Legal & Policy Documentation**

- `LEGAL_THIRD_PARTY.md`
- `INTEROP_POLICY.md`
- `REPRODUCIBILITY.md`

Explicit separation of:
- Core decoding logic
- Interop benchmarking layer
- Reference baselines

---

### Changed

- Removed post-hash mutation of benchmark records
- Hardened interop record validation logic
- Enforced canonical JSON configuration contract
- Deterministic report ordering for stable Markdown output
- Documentation updated to match schema requirements

---

### Guarantees

- No changes to core decoding logic
- No changes to scheduling or ensemble behavior
- No API breaking changes
- No new required dependencies
- SCHEMA_VERSION remains `3.0.1`
- INTEROP_SCHEMA_VERSION remains `3.1.2`
- Byte-identical artifacts with:
  - `runtime_mode="off"`
  - `deterministic_metadata=True`
  - fixed seed

---

### Reproducibility Anchor

Deterministic Suite Artifact (SHA-256):


431f7573a0ba8af4784b385f528cfe99d6169eb74798eabddd146def278b6d77


Golden Vector Hash:


86babd2ec81daa165d3ce778b9eb71a3766667484e1c51a2000642ae08ec9569


---

### Test Status

608 passed  
7 skipped  
0 failed  

Interop schema and determinism hardening release.

## [3.0.2] - 2026-02-25

### Added

**Fuzz-Style Determinism Tests for canonicalize()**
- Seeded random nested structure generator (numpy.random.default_rng)
- Idempotence test: canonicalize(canonicalize(x)) == canonicalize(x)
- JSON roundtrip stability test: stable serialization across calls
- No-input-mutation test: original objects unchanged after canonicalization
- Repeatability test: identical outputs across repeated runs
- 50 fuzz cases per test, max recursion depth 3

### Guarantees
- No production code changes
- No behavior changes
- No API changes
- No dependency changes
- No decoder or schema modifications
- All existing tests remain green

---

## [3.0.1] - 2026-02-25

### Added

**Dimension-Aware Scaffolding (QuditSpec)**
- Optional `qudit` configuration block
- Validated, JSON-safe `QuditSpec` (dimension, encoding, metadata)
- Defaults to qubit mode (dimension=2)
- No changes to decoder or simulation behavior

**Deterministic Analytical Gate-Cost Modeling**
- Optional `resource_model` configuration block
- Deterministic analytical resource estimation utilities
- Canonicalized `assumptions` field included for traceability
- No impact on FER simulation or decoding logic

**Shared Canonicalization Utility**
- Introduced `src/utils/canonicalize.py`
- Eliminated duplicated canonicalization logic
- Single deterministic JSON-safe normalization path
- Prevents drift between schema and dimension layers

**Regression & Compatibility Hardening**
- Schema version roundtrip regression tests
- Determinism smoke test validated
- Backward compatibility audit suite
- Import hygiene verification tests
- Nonbinary scaffolding interfaces (no decoding implementation)

---

### Changed

- Result `schema_version` now strictly preserved from input config
- Centralized canonicalization across schema and qudit layers
- Gate-cost output now includes canonicalized `assumptions` (additive field only)

---

### Guarantees

- No changes to core decoding logic
- No changes to scheduling or ensemble behavior
- No public API changes
- No new required configuration fields
- No new external dependencies
- Determinism preserved (`runtime_mode="off"` byte-identical verified)
- v3.0.0 configurations load and run unchanged

---

### Test Status

526 passed  
7 skipped  
0 failed

[3.0.0] - 2026-02-25

Added
Deterministic benchmarking framework under src/bench/:
- Config-driven sweep over decoders, distances, and physical error rates
- Canonical JSON result schema (3.0.0)
- Schema validation prior to return
- Cryptographic sub-seed derivation (order-independent)
- Optional deterministic_metadata mode for byte-identical artifacts
- Runtime measurement module (perf_counter_ns, 95% CI, optional tracemalloc)
- Threshold estimation via FER crossing interpolation
- Log–log runtime scaling analysis
- DecoderAdapter abstraction with BP adapter implementation

Changed
Sub-seed derivation now functional (SHA-256 over logical coordinates)
Microsecond-free timestamps
Schema version unified via single SCHEMA_VERSION constant
Early config/schema mismatch validation guard
Corrected runtime slope estimation to filter zero-latency points consistently

Guarantees
Core decoding logic unchanged
No scheduling changes
No adaptive logic changes
No ensemble behavior changes
No API breaking changes
No new external dependencies
Determinism preserved
Order-independent seed derivation
Backward compatibility with v2.9.1 decoding behavior

Test Status
438 passed
7 skipped
0 failed

## [2.9.1] - 2026-02-25

### Added
- Opt-in residual metric instrumentation:
  - residual_linf (per-check L∞ norm)
  - residual_l2 (per-check L2 norm)
  - residual_energy (per-iteration scalar)

### Guarantees
- Default decode behavior bit-identical to v2.9.0
- No scheduling logic changes
- No adaptive changes
- No API breaking changes
- Determinism preserved

## [2.9.0] - 2026-02-24

### Added
- Deterministic adaptive schedule controller (`schedule="adaptive"`):
  - Phase 1: `flooding` for `k1` iterations
  - Phase 2: `hybrid_residual` for remaining iterations
  - Default `k1 = max(1, max_iters // 4)`
- Cumulative iteration accounting (total iterations across phases)
- Strict validation of adaptive parameters:
  - `adaptive_k1` must satisfy `1 ≤ k1 < max_iters`
  - `adaptive_rule` explicitly validated
- Edge-case guard for small budgets (`max_iters = 1`)
- Comprehensive test coverage for adaptive behavior

### Behavior Guarantees
- Strictly one-way switching (no dynamic residual-based switching)
- No internal message state shared between phases
- Deterministic tie-breaking:
  - Converged solution preferred
  - Lower syndrome weight
  - Fewer total iterations
  - Phase order as final deterministic tie-break
- No randomness introduced
- No global state

### Unchanged
- No modifications to existing schedules:
  - `flooding`
  - `layered`
  - `residual`
  - `hybrid_residual`
- No changes to ensemble decoding behavior
- No breaking API changes
- Default decoder calls remain bit-stable

### Test Status
- 364 passed
- 7 skipped
- 0 failed
- CI green

[2.8.0] - 2026-02-23

Deterministic Scheduling & State-Aware Enhancements

Belief Propagation decoder enhancements for QLDPC codes.

Added
improved_norm / improved_offset modes

Extended min-sum variants with dual scaling parameters:

alpha1 applied to first minimum

alpha2 applied to second minimum

Deterministic, invariant-preserving check-node updates

Fully backward-compatible with existing min-sum modes

hybrid_residual schedule

Deterministic even/odd check-node partitioning

Per-layer descending residual ordering

Optional hybrid_residual_threshold to prioritize high-residual checks

Stable tie-breaking by ascending check index

No randomness introduced

Deterministic ensemble decoding (ensemble_k)

K independent BP passes using deterministic zero-mean alternating perturbations

Member 0 uses exact baseline LLR

Selection priority:

Converged solution

Lowest syndrome weight

Deterministic member index

No RNG usage; fully reproducible

State-aware residual weighting (state_aware_residual)

Residual modulation:

weight = s_by_state[label] * |cos(phi_by_state[label])|

Multiplicative weighting of residual ordering

Strict validation:

Non-negative labels

In-range labels

Length must equal number of checks (m)

Disabled by default (no baseline behavior change)

Improved

Precomputed ensemble syndrome matrix (H32) to avoid repeated casting

Precomputed state-aware residual weights to eliminate per-iteration trig

Hybrid threshold validation scoped to hybrid schedule only

Alpha parameter semantics aligned with documentation

Testing

Added pytest.ini to scope test discovery to tests/

Full regression suite:

339 passed

7 skipped

0 failed

Determinism verified across repeated runs

[2.7.0] — 2026-02-23
Deterministic Residual Scheduling

Added

Residual-Ordered Layered Scheduling (schedule="residual")

Deterministic per-iteration reordering of check nodes based on descending maximum message residual.

Residual defined as:

max |new_msg - old_msg| per check node.

Stable lexicographic ordering via:

np.lexsort((check_indices, -residuals))

Deterministic tie-breaking by ascending check index.

Fully opt-in behavior.

Default flooding and layered schedules unchanged.

Compatibility

Works with all BP modes:

sum_product

min_sum

norm_min_sum

offset_min_sum

Fully compatible with:

damping

clipping

LLR history instrumentation (llr_history)

OSD-0, OSD-1, and OSD-CS post-processing

No change to public API.

No change to return signatures.

No new dependencies introduced.

Changed

Precomputed check_indices array to avoid per-iteration allocation during residual scheduling.

Minor documentation wording improvement:

“floating precision” → “floating-point precision”.

Verified

73 BP decoder regression tests across v2.4–v2.6 passing.

312 total project tests passing (environment-dependent mirror tests unaffected).

Deterministic repeated runs verified for residual schedule.

Flooding and layered schedules remain bit-identical to v2.6.0.

Backward compatibility maintained:

No API breakage.

No required dependency changes.

Default behavior remains bit-identical to v2.6.0.

[2.6.0] — 2026-02-23
Deterministic Decoding Hardening and Meta-Algorithm Stabilization
Added

Order-k Combination Sweep OSD (postprocess="osd_cs")

Deterministic candidate ordering via structured _candidate_key comparison.

Lexicographic ordering: Hamming weight → rounded path metric → combination index.

NumPy-native metric rounding (12 decimal places) to eliminate floating-point precision ordering drift.

Configurable sweep depth via osd_cs_lam.

Explicit never-degrade fallback: original hard decision returned if no valid candidate found.

Deterministic Decimation Meta-Decoder

Standalone module:

decimate(...)

decimation_round(...)

Features:

Threshold-based commitment with ascending index tie-breaking.

Optional peeling with deterministic ascending check-node propagation.

Scaled LLR clamping using channel-derived magnitude (no fixed magic constants).

Syndrome-verified early-return behavior (invalid fully-committed states rejected).

LLR History Instrumentation (llr_history)

Optional circular history buffer in bp_decode.

Returns (correction, iterations, history) when enabled.

Flooding and layered schedules supported.

Default return signature unchanged.

No impact on deterministic defaults.

Changed

Belief construction in decimation now strictly follows:

Sign from hard decision.

Magnitude from |clamped_llr|.

Removed redundant L_total recomputation in flooding schedule:

L_total allocated once per iteration.

Snapshot uses existing vector copy.

Eliminates extra Python-level O(m·n) pass.

Decimation early-return now verifies syndrome before accepting fully committed state.

Improved test coverage for:

OSD-CS never-degrade guarantee.

osd_cs_lam=0 equivalence with osd0 at bp_decode level.

llr_history 3-tuple compatibility in decimation meta-loop.

Test suite now environment-agnostic:

Mirror repository tests auto-skip if gh CLI is not present.

Verified

305 tests collected.
All QEC core tests passing.
Deterministic behavior preserved for all default configurations.

Backward compatibility maintained:

No API breakage.

No new required dependencies.

Default behavior remains bit-identical to v2.5.0.

## [2.5.0] — 2026-02-21

### Deterministic Statistical Rigor and Layered Decoding

### Added

Wilson score confidence intervals for Monte Carlo FER simulations (`ci_method="wilson"`):
- Continuity-corrected Wilson interval with configurable `alpha`.
- `gamma >= 0.0` continuity correction factor (set `gamma=0` to disable correction).
- Pure NumPy implementation; no new external dependencies.
- Deterministic integer-grounded computation (no float reconstruction of counts).

Deterministic early termination for FER simulations (`early_stop_epsilon`):
- Stops trials once CI width falls below user-defined threshold.
- Fully reproducible: identical seed and parameters yield identical termination points.
- Reports `actual_trials` per noise level when enabled.

Layered (serial) belief-propagation scheduling (`schedule="layered"`):
- Incremental LLR updates with maintained belief invariants.
- O(nnz(H)) per iteration.
- Typically faster convergence than flooding.
- Fully deterministic fixed check-node traversal order.

Order-1 Ordered Statistics Decoding (`postprocess="osd1"`):
- Extends OSD-0 with single least-reliable pivot bit flip.
- Deterministic tie-breaking.
- Preserves never-degrade guarantee.

### Changed

Confidence interval validation semantics:
- `gamma` now allowed to be `>= 0.0`.
- `alpha` and `gamma` validation scoped to CI-enabled runs only.
- Documentation aligned with actual contract.

Internal Wilson CI implementation updated to use stored integer frame error counts directly (eliminates float-based reconstruction).

Backward compatibility preserved:
- All new features are opt-in.
- Default parameters produce bit-identical output to v2.4.0.

### Verified

247/247 core tests passing.
No change in deterministic behavior for existing configurations.

## [2.3.0] — 2026-02-18

### Decoder Utility Formalization and Stability Refinement

### Added

Standalone decoder utility layer formalizing detection–inference–correction separation:

update_pauli_frame(frame, correction) — pure GF(2) Pauli-frame XOR update (non-mutating, validated).

syndrome(H, e) — standalone binary syndrome computation.

bp_decode(H, llr, max_iter, syndrome_vec) — standalone belief-propagation decoder operating on per-variable LLR vectors.

detect(H, e) — thin wrapper over syndrome.

infer(H, llr, max_iter, syndrome_vec) — thin wrapper over bp_decode.

channel_llr(e, p, bias) — channel LLR computation with optional scalar or per-variable bias weighting.

36 new unit tests covering:

Pauli-frame algebra

Syndrome equivalence

BP determinism and convergence

Channel LLR validation and bias behavior

Integration with decoding workflow

### Changed

channel_llr now enforces p ∈ (0, 1) to prevent undefined or numerically unstable boundary behavior.

bp_decode now precomputes integer-casted parity-check matrix and syndrome vectors for early-stopping checks, eliminating repeated per-iteration casting.

Decoder workflow is now explicitly modular while remaining backward compatible.

### Notes

No changes to construction layer.

No changes to additive lift invariants.

No changes to CSS orthogonality guarantees.

No changes to JointSPDecoder public API.

Fully backward compatible.

All tests passing (101 / 101).

## [2.2.0] — 2026-02-18

### Belief-Propagation Stability Hardening

### Added

Explicit handling of degree-1 check nodes in the JointSPDecoder belief-propagation loop.

Zero extrinsic message returned for single-neighbor check nodes.

### Changed

Corrected check-to-variable update rule in _bp_component:

Degree-1 check nodes now return 0.0 (no extrinsic information)
instead of falling through to the general tanh-product rule.

This prevents artificial LLR amplification from:

atanh(≈1) → ∞


when the product over an empty neighbor set numerically approaches unity.

### Fixed

Eliminated false confidence injection in BP decoding for sparse parity structures.

Resolved numerical instability in extremely sparse or irregular Tanner graphs.

### Notes

No changes to construction layer.

No changes to additive lift invariants.

No changes to CSS orthogonality logic.

All tests passing (65 / 65).

Decoder stability hardening release.

## [2.1.0] — 2026-02-16

### Additive Lift Invariant Hardening

### Added

Additive lift invariant formalization for shared-circulant QLDPC CSS constructions.

Deterministic structured shift mapping:

s(i, j) = (r_i + c_j) mod L


Algebraic guarantee of lifted CSS orthogonality.

Sparse-safe orthogonality verification.

Binary GF(2) rank computation without dense float conversion.

Expanded invariant test coverage (89 / 89 passing).

### Changed

Replaced per-edge random lift tables with additive invariant lift structure.

Lift implementation is now deterministic, process-independent, and order-independent.

Orthogonality now follows structurally from base-matrix commutation.

### Removed

Probabilistic orthogonality edge-case behavior from prior lift implementation.

### Notes

No architectural changes from v2.0.0.

Structural invariant hardening release.

## [2.0.0] — 2026-02-??

### Architectural Expansion of QLDPC CSS Stack

### Added

Multidimensional stabilizer stack.

Protograph-based QLDPC CSS constructions.

GF(2^e) finite-field lifting framework.

Ternary Golay [[11,1,5]]₃ implementation.

Ququart stabilizer and D4 lattice prior layer.

Deterministic seeded construction framework.

Integrated simulation and hashing bound tooling.

### Notes

Major architectural rewrite establishing the construction and decoding foundation for subsequent invariant hardening and stability refinement releases.
