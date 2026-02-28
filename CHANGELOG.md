# Changelog

All notable changes to this project are documented in this file.

This project follows semantic versioning (SemVer).

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
