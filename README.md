# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Latest Release](https://img.shields.io/github/v/release/QSOLKCB/QEC?color=blue)](https://github.com/QSOLKCB/QEC/releases/latest)

License: CC-BY-4.0

Deterministic quantum error correction framework for QLDPC CSS codes with algebraic construction guarantees, numerically stable belief propagation, statistically rigorous FER simulation, and modular decoder utilities.

Release Lineage

### v2.9.1 — Deterministic Residual Instrumentation Hardening

### v2.9.0 — Deterministic Adaptive Scheduling

**Highlights**

Deterministic Adaptive Schedule Controller

- `schedule="adaptive"`
- Two-phase, strictly one-way controller:
  - Phase 1: `flooding` for k1 iterations
  - Phase 2: `hybrid_residual` for remaining iterations
- Default `k1 = max(1, max_iters // 4)`
- No dynamic backtracking
- No residual-triggered switching
- No internal message state shared between phases

Cumulative Iteration Accounting

- Public iteration count reflects total iterations across phases
- Deterministic tie-breaking:
  - Converged solution preferred
  - Lower syndrome weight
  - Fewer total iterations
  - Phase order as final deterministic tie-break

Strict Validation

- `adaptive_k1` must satisfy `1 ≤ k1 < max_iters`
- `adaptive_rule` explicitly validated
- Guarded small-budget behavior (`max_iters = 1` handled cleanly)

Design Constraints (Intentional)

- Strictly one-way switching in v2.9.0
- No residual-dynamic or feedback-based adaptation
- No ensemble integration inside adaptive
- No changes to existing schedule behavior

Determinism & Stability

- No RNG
- No new global state
- No vectorization or refactoring of core loops
- No changes to:
  - `flooding`
  - `layered`
  - `residual`
  - `hybrid_residual`
  - `ensemble_k`
- Fully reproducible across runs

Test Status

364 passed  
7 skipped  
0 failed  

CI green  
Regression verified  

---

### v2.8.0 — Deterministic Ensemble & State-Aware Scheduling

**Highlights**

Dual-parameter min-sum variants

- `mode="improved_norm"`
- `mode="improved_offset"`
- `alpha1` applies to first minimum
- `alpha2` applies to second minimum
- Consistent across flooding and layered schedules
- Deterministic mapping verified

Hybrid Residual Scheduling

- `schedule="hybrid_residual"`
- Deterministic even/odd check partition
- Within each layer: descending residual ordering
- Optional `hybrid_residual_threshold`
- Threshold validation scoped to hybrid schedule only
- Lexicographic tie-breaking preserved
- No randomness introduced
- Default behavior unchanged

Deterministic Ensemble Decoding

- `ensemble_k >= 1`
- Member 0 uses exact baseline LLR
- Additional members use deterministic zero-mean alternating perturbations
- No RNG
- No global state
- Selection priority:
  - Converged solution
  - Lowest syndrome weight
  - Lowest member index
- Precomputed H32 avoids repeated casting
- Fully reproducible across runs

State-Aware Residual Weighting

- `state_aware_residual=True`
- Residual weights:
  - `weight = s_by_state[label] * |cos(phi_by_state[label])|`
- Multiplicative modulation of residual ordering
- Strict validation:
  - Integer labels
  - Non-negative
  - In-range
  - Length must equal number of checks
- Weights precomputed once per decode call
- Disabled by default (zero behavior change when off)

Determinism & Stability

- Exactly one residual update block
- Float64 discipline preserved
- No change to default `schedule="flooding"`
- No return signature changes
- No breaking API changes
- Bit-stable for default calls
- Ensemble and state-aware paths fully deterministic

Test Status

339 passed  
7 skipped  
0 failed  

CI green  
Static review issues resolved

v2.8.0 — Deterministic Ensemble & State-Aware Scheduling
Highlights

Dual-parameter min-sum variants

mode="improved_norm"

mode="improved_offset"

alpha1 applies to first minimum

alpha2 applies to second minimum

Consistent across flooding and layered schedules

Deterministic mapping verified

Hybrid Residual Scheduling

schedule="hybrid_residual"

Deterministic even/odd check partition

Within each layer: descending residual ordering

Optional hybrid_residual_threshold

Threshold validation scoped to hybrid schedule only

Lexicographic tie-breaking preserved

No randomness introduced

Default behavior unchanged

Deterministic Ensemble Decoding

ensemble_k >= 1

Member 0 uses exact baseline LLR

Additional members use deterministic zero-mean alternating perturbations

No RNG

No global state

Selection priority:

Converged solution

Lowest syndrome weight

Lowest member index

Precomputed H32 avoids repeated casting

Fully reproducible across runs

State-Aware Residual Weighting

state_aware_residual=True

Residual weights:

weight = s_by_state[label] * |cos(phi_by_state[label])|

Multiplicative modulation of residual ordering

Strict validation:

Integer labels

Non-negative

In-range

Length must equal number of checks

Weights precomputed once per decode call

Disabled by default (zero behavior change when off)

Determinism & Stability

Exactly one residual update block

Float64 discipline preserved

No change to default schedule="flooding"

No return signature changes

No breaking API changes

Bit-stable for default calls

Ensemble and state-aware paths fully deterministic

Test Status

339 passed

7 skipped

0 failed

Regression verified twice

CI green

Static review issues resolved

v2.7.0 — Deterministic Residual Scheduling
Highlights

Deterministic residual-based layered scheduling

schedule="residual"

Per-check residual defined as:

abs(new_msg - old_msg)

Residual aggregated deterministically per check using max residual

Lexicographic ordering:

np.lexsort((check_indices, -residuals))

Tie-breaking guaranteed by ascending check index

No randomness introduced

No change to default behavior

"flooding" remains default

Bit-identical outputs for default calls

No API changes

No return signature changes

Float64 discipline preserved

Fully compatible with:

sum_product

min_sum

norm_min_sum

offset_min_sum

damping

clipping

existing postprocess logic

Performance hardening

Precomputed check_indices to avoid per-iteration allocation

No heap structures

Minimal diff implementation

All 73 BP decoder tests pass across v24, v25, v26 suites
Full suite passes (excluding environment-dependent mirror tests)

v2.7.0 — Deterministic Residual Scheduling

Highlights

Deterministic residual-based layered scheduling

schedule="residual"

Per-check residual defined as:

abs(new_msg - old_msg)

Residual aggregated deterministically per check using max residual

Lexicographic ordering:

np.lexsort((check_indices, -residuals))

Tie-breaking guaranteed by ascending check index

No randomness introduced

No change to default behavior

"flooding" remains default

Bit-identical outputs for default calls

No API changes

No return signature changes

Float64 discipline preserved

Fully compatible with:

sum_product

min_sum

norm_min_sum

offset_min_sum

damping

clipping

existing postprocess logic

Performance hardening

Precomputed check_indices to avoid per-iteration allocation

No heap structures

Minimal diff implementation

All 73 BP decoder tests pass across v24, v25, v26 suites
Full suite passes (excluding environment-dependent mirror tests)

v2.6.0 — Deterministic Decoding & Stability Hardening

Highlights

OSD-CS (Combination Sweep)

Deterministic path-metric candidate ordering

NumPy-stable metric rounding (12-decimal precision)

Guaranteed never-degrade fallback

Integrated via postprocess="osd_cs" with osd_cs_lam

Deterministic Decimation Module

Threshold-based commitment with index-ordered tie-breaking

Optional peeling (ascending check-index propagation)

Syndrome-verified early-return logic

Scaled LLR clamping (no magic constants)

LLR History Instrumentation

Optional llr_history parameter in bp_decode

Returns (correction, iterations, history) when enabled

Circular buffer implementation

No impact on default return signature

Stability & Performance Fixes

Belief construction now strictly:
sign from hard decision + magnitude from |clamped_llr|

Early-return in decimation verified against syndrome

Flooding schedule L_total recomputation removed (performance hardening)

3-tuple return compatibility hardened

Environment-agnostic test suite (mirror tests auto-skip if gh absent)

All defaults remain bit-identical to v2.5.0.

v2.5.0 — Deterministic Statistical Rigor + Layered Decoding

Highlights

Wilson score confidence intervals for FER simulation

Continuity-corrected Wilson interval (ci_method="wilson")

Configurable alpha

gamma >= 0.0 (set gamma=0 to disable correction)

Integer-grounded computation

Deterministic early termination

early_stop_epsilon stops trials once CI width is sufficiently tight

Layered (serial) BP scheduling

schedule="layered"

O(nnz(H)) per iteration

OSD-1 post-processing

postprocess="osd1"

Deterministic tie-breaking

Never-degrade guarantee preserved

All features opt-in.
Defaults remain bit-identical to v2.4.0.

v2.4.0 — Performance Hardening + Deterministic FER Harness

Highlights

Multi-mode BP decoder

sum_product

min_sum

norm_min_sum

offset_min_sum

Message damping and magnitude clipping

OSD-0 post-processing

Dict-based asymmetric channel bias

Deterministic Monte Carlo FER harness

Optional FER plotting utility

Backward-compatible API.

v2.3.0 — Decoder Utility Formalization

Explicit detection → inference → correction separation

Standalone bp_decode

Pauli-frame update abstraction

Input validation enforcement

Reduced BP early-stop overhead

v2.2.0 — Belief Propagation Stability Hardening

Correct handling of degree-1 checks

Eliminated artificial LLR amplification

Stabilized sparse Tanner graph behavior

v2.1.0 — Additive Lift Invariant Hardening

Algebraically guaranteed lifted CSS construction:

s(i, j) = (r_i + c_j) mod L

Sparse-safe GF(2) rank computation

Deterministic seeded construction

Full invariant enforcement

v2.0.0 — Architectural Expansion

Protograph-based QLDPC construction

GF(2^e) lifting

Ternary Golay [[11,1,5]]₃

Ququart stabilizer + D4 lattice prior

Deterministic construction framework

Current System State (v2.7.0)

Construction layer is algebraically enforced.

Decoder layer supports:

Multi-mode BP

Flooding, layered, and residual scheduling

OSD-0, OSD-1, and OSD-CS

Deterministic decimation rounds

Optional diagnostic instrumentation:

LLR history tracing

FER simulation harness includes:

Wilson confidence intervals

Deterministic early termination

Seeded RNG workflow

Full deterministic execution path.

Architecture Overview

Channel Model → channel_llr
Detection → syndrome / detect
Inference → bp_decode / infer
Post-processing → osd0 / osd1 / osd_cs
Decimation → decimate / decimation_round
Correction → update_pauli_frame
Construction Layer → Additive invariant QLDPC CSS lift

Citation & Authorship

Author: Trent Slade
ORCID: https://orcid.org/0009-0002-4515-9237

If you use this toolkit in academic work, please cite appropriately and reference the release version used.

Small is beautiful.
Determinism is holy.
