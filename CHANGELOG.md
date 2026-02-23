# Changelog

All notable changes to this project will be documented in this file.

The project follows semantic versioning.
Each release reflects structural, numerical, or architectural maturity improvements in the QLDPC CSS construction and decoding stack.

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
