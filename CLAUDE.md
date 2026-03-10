# CLAUDE.md — QEC Architectural Constitution (v7 Hardened)

This document governs all AI-assisted code activity inside the QSOL QEC repository.

It applies to all actions performed by Claude, including:

- code generation
- code modification
- refactoring
- testing
- release preparation
- commits and pushes

This document is not guidance.

It is the **architectural constitution** of the repository.

Claude must obey these rules when operating inside this codebase.

---

# Core Values

The QEC framework is governed by the following non-negotiable principles:

**Determinism**  
All experiments must be reproducible byte-for-byte.

**Decoder Stability**  
The decoding algorithms are protected infrastructure.

**Strict Layering**  
Architecture must remain clean and directional.

**Schema Governance**  
Artifacts and identities must remain stable.

**Minimal Dependencies**  
Avoid unnecessary complexity.

**Reproducibility Guarantees**  
Experiments must produce identical outputs across runs.

---

# 1. Architectural Layer Model (Non-Negotiable)

The repository follows a strict dependency hierarchy.

Dependencies may only flow **downward**.

| Layer | Path | Role |
|------|------|------|
| 1 | `src/qec/decoder/` | Decoder core (protected) |
| 2 | `src/qec/channel/` | Channel and noise models |
| 3 | `src/qec/diagnostics/` | Deterministic diagnostics |
| 4 | `src/qec/predictors/` | Structural instability predictors |
| 5 | `src/qec/experiments/` | Experimental policies & controllers |
| 6 | `src/bench/` | Benchmark harness & experiment orchestration |

Dependency rules:

- Lower layers must **never import from higher layers**.
- `src/qec/decoder/` must **never import from experiments or benchmarking**.
- `src/qec/` must **never import from `src/bench/`**.
- Predictors must not depend on experiment code.
- Experiments may depend on predictors, diagnostics, and decoder APIs only.

Layer boundaries are **architectural invariants**.

Violating layer boundaries is forbidden without explicit architectural approval.

---

# 2. Determinism is Architecture

Determinism is not a feature.

It is a structural requirement.

All code must preserve deterministic execution.

Required invariants:

- No hidden randomness
- No global RNG state
- Explicit seed injection at call site
- No use of Python `hash()` for deterministic operations
- Deterministic ordering of collections
- No unordered reductions introducing floating drift

Sub-seed derivation must use:


SHA-256 or equivalent deterministic hashing


Canonical JSON serialization must remain centralized.

Sweep hashes must derive strictly from configuration state.

When:


runtime_mode = "off"


Artifacts must be **byte-identical across repeated runs**.

If a function is deterministic today, it must remain deterministic tomorrow.

---

# 3. Artifact & Identity Stability

Artifacts represent immutable experiment records.

Rules:

- Artifact hashes must reflect **final canonical state only**
- No mutation after hashing
- Serialization order must remain canonical
- Decoder identity must be stable across identical configs
- Identity must not depend on object memory layout

Baseline identity must **not include opt-in parameters** unless explicitly enabled.

Identity drift without version bump is forbidden.

---

# 4. Decoder Core Protection

The decoder is the most protected subsystem in the repository.

Default rule:

**Do not modify the decoder core.**

Protected path:


src/qec/decoder/


Prohibited without explicit instruction:

- modifying BP message passing
- altering scheduling semantics
- changing iteration order
- performance "optimizations"
- refactoring for style
- altering function signatures
- introducing new internal code paths

The decoder must remain stable across minor releases.

---

# 4a. Opt-In Structural Extensions

Structural interventions may exist only if:

- explicitly enabled via configuration
- default behavior remains bit-identical
- BP loops remain untouched
- no randomness is introduced
- baseline identity does not change
- execution remains deterministic

Structural extensions must be **additive**, not invasive.

---

# 4b. Diagnostics Layer Protection

Diagnostics are **observational instruments**.

They must never influence decoder behaviour.

Diagnostics must be:

- deterministic
- side-effect free
- opt-in
- isolated from decoding loops

Diagnostics must **not**:

- modify BP message values
- alter LLR vectors passed to the decoder
- change iteration ordering
- modify scheduling logic
- inject hooks into BP loops
- introduce conditional behaviour inside decoder internals
- mutate input arrays in-place

Diagnostics may run additional decodes, but only using:

- explicit input copies
- deterministic perturbations
- no shared mutable state

The decoder must behave as though diagnostics do not exist.

---

# 4c. Predictor Layer Protection

Predictors estimate decoding instability.

They must **not influence decoder behaviour directly**.

Predictors must:

- operate only on diagnostic outputs
- remain deterministic
- produce structural risk signals

Predictors must not:

- modify decoder inputs
- change scheduling behaviour
- inject heuristics into decoding loops
- cache decoder state across runs
- introduce stochastic estimation

Predictors generate signals such as:


bp_failure_risk
predicted_instability
spectral_instability_ratio


Predictors produce **information only**.

---

# 4d. Controller Layer Protection

Controllers are experimental policy layers.

They translate predictor signals into controlled experiments.

Controllers must:

- remain deterministic
- remain opt-in
- leave baseline decoding unchanged when disabled

Controllers must not:

- modify decoder implementation
- alter BP message passing
- patch decoder functions
- introduce randomness

Controllers operate as **wrappers around decoder calls**.

---

# 5. Channel Layer Discipline

Channel models simulate noise.

They are consumers of decoder functionality.

Channel code must:

- not mutate decoder state
- remain deterministic under fixed seeds
- maintain backward compatibility

Supported baseline channels include:


oracle
bsc_syndrome


Channel realism must never destabilize decoder invariants.

---

# 6. Schema Governance

Schema changes are high-risk.

Rules:

- Do not bump `SCHEMA_VERSION` without instruction
- Minor releases may add fields only
- No type changes
- No field removals
- Validation must remain strict

Canonicalization must use centralized serialization logic only.

Duplicate serialization pathways are forbidden.

---

# 7. Forbidden Code Regions (AI Agent Safety Map)

Certain repository regions are **protected from AI modification**.

Claude must never modify these paths without explicit instruction.

Protected regions:


src/qec/decoder/
src/qec/schema/
src/qec/serialization/
src/qec/hash/


These components define:

- decoder behaviour
- artifact identity
- schema structure
- reproducibility guarantees

Edits to these files require **explicit user authorization**.

If a requested change touches these regions, Claude must:

1. Stop
2. Explain the risk
3. Request confirmation

---

# 8. Minimal Diff Discipline

Every changed line must justify itself.

Forbidden without instruction:

- large refactors
- renaming functions or variables
- style-only edits
- import reordering
- file-wide formatting
- abstraction reshuffling
- moving code between modules

Commits must be small and single-purpose.

---

# 9. Dependency Policy

Dependency surface must remain minimal.

Rules:

- prefer stdlib
- prefer NumPy
- no new dependencies without approval
- no dependency upgrades without instruction
- no convenience frameworks

All dependencies must be version-bounded.

Architectural bloat is forbidden.

---

# 10. Versioning Discipline

The repository follows Semantic Versioning:


vMajor.Minor.Patch


Meaning:

Major  
Architectural or semantic changes

Minor  
Additive features

Patch  
Bug fixes and stability improvements

Minor releases must preserve:

- baseline decoder outputs
- identity stability
- determinism guarantees

Any default behavior change requires a **major version bump**.

---

# 11. Performance Stability

Performance is correctness for minor releases.

Rules:

- no measurable slowdown without approval
- no algorithmic complexity changes
- no silent regressions

Benchmark comparison against the previous release must be performed before tagging.

---

# 12. Test Discipline

Untested code is unshipped code.

Required:

- unit tests for all new features
- determinism tests
- regression tests
- schema validation tests

Tests must not:

- weaken assertions
- broaden tolerances
- hide drift with approximate comparisons

Correct code makes tests pass.

Tests do not adapt to drift.

---

# 13. Commit & Push Escalation

Claude may commit or push only if:

- all invariants are preserved
- protected subsystems remain untouched
- determinism is intact
- schema remains unchanged
- identity remains stable

Push escalation is required for changes touching:

- decoder internals
- scheduling logic
- serialization or hashing
- schema validation
- artifact identity
- determinism guarantees

Passing tests alone are insufficient.

---

# 14. Escalation Rule

If a proposed change affects:

- decoder semantics
- scheduling
- schema
- serialization
- hashing
- artifact identity
- determinism guarantees

Claude must **pause and request explicit instruction**.

Silence is not consent.

---

# 15. Governing Principle

When uncertain:

Preserve stability.

Avoid refactoring.

Prefer doing nothing.

Read before writing.

Maintain invariants.

Capability grows.

Stability does not regress.

If a result cannot be reproduced byte-for-byte, it is not a baseline.
