CLAUDE.md — QEC Architectural Constitution (v3.x Hardened)

This document is binding.
It governs all code generation, modification, refactor, testing, release, and push actions performed by Claude inside this repository.

This is not guidance.
It is a contract.

Core values:

Determinism

Decoder stability

Strict layering

Schema governance

Minimal dependency surface

Reproducibility guarantees

1. Layering Model (Non-Negotiable)

Dependency direction is strictly enforced.

Layer	Path	Role
1	src/qec/	Decoder core (protected)
1a	src/qec/decoder/	Opt-in structural extensions
2	src/qec/channel/	Channel abstraction
3	src/bench/	Benchmarking & interop

Rules:

Lower layers must never depend on higher layers.

src/qec/ must never import from src/bench/.

Channel code must not mutate decoder internals.

Benchmarking code must not alter decoding semantics.

Ambiguous dependency direction is forbidden until clarified.

Layer boundaries are architectural invariants.

2. Determinism is Architecture

Determinism is not a feature. It is structural.

Required invariants:

No hidden randomness.

Explicit seed injection at call site.

No global RNG state.

No use of Python hash().

Sub-seed derivation must use SHA-256 or equivalent.

Canonical JSON serialization is centralized and immutable.

Sweep hash derives strictly from config.

runtime_mode="off" → byte-identical artifacts.

No unordered reductions that introduce floating drift.

Identity and hash derivation must be pure functions of config.

If a function is deterministic today, it must remain deterministic tomorrow.

3. Artifact & Identity Stability

Artifacts are immutable records.

Artifact hash must reflect final canonical state only.

No post-hash mutation.

Serialization order must remain canonical.

Decoder identity must be stable across identical configs.

Baseline identity must not include opt-in parameters unless enabled.

Identity serialization must not depend on object memory layout.

Identity drift without version bump is forbidden.

4. Decoder Core Protection

The decoder is the most protected subsystem.

Default rule: do not modify src/qec/ internals.

Prohibited without explicit instruction:

Modifying BP message passing.

Changing scheduling semantics.

Changing iteration order.

Refactoring for style.

Performance “optimizations.”

Altering function signatures.

Introducing new internal code paths.

Opt-In Structural Extensions (Layer 1a)

Permitted only if ALL conditions hold:

Explicitly enabled via configuration.

Default behavior remains bit-identical.

BP loops remain untouched.

No randomness introduced.

No schedule mutation.

No baseline identity drift.

Fully deterministic.

Structural extension must be additive, not invasive.

5. Channel Layer Discipline

Channel models are consumers, not peers.

Must not mutate decoder state.

Must not alter scheduling or convergence logic.

"oracle" mode must remain backward-compatible.

Channel changes must be documented.

Deterministic under fixed seed.

Additive only — no renaming, no removal.

Channel realism must not destabilize decoder invariants.

6. Schema Governance

Schema is high-risk.

Do not bump SCHEMA_VERSION without instruction.

Minor releases: additive fields only.

No type changes.

No removals.

Validation must remain strict.

Canonicalization must use centralized logic only.

No duplicate serialization pathways.

No loosening validators to accommodate drift.

Schema changes require explicit architectural justification.

7. Minimal Diff Discipline

Every line changed must justify itself.

Forbidden without instruction:

Large refactors.

Renaming functions or variables.

Style-only edits.

Import reordering.

File-wide formatting.

Abstraction reshuffling.

Adding comments to untouched code.

Moving code between modules.

Small commits. Single concern per change.

8. Dependency Policy

Dependency surface must remain minimal.

No new dependencies without approval.

Prefer stdlib + NumPy.

No dependency upgrades without instruction.

No convenience frameworks.

All dependencies must be version-bounded.

No architectural bloat.

9. Versioning Discipline

Version numbers encode meaning.

Format: vArchitecture.Major.Minor

Architecture → foundational redesign.

Major → decoder or schema behavior change.

Minor → additive only.

Minor releases must:

Preserve baseline decoding outputs.

Preserve identity stability.

Preserve determinism.

Avoid performance regression.

Any default behavior change requires major bump.

10. Performance Stability

Performance is correctness in minor releases.

No measurable slowdown unless requested.

No algorithmic complexity changes.

No silent regressions.

Benchmark comparison against prior release required before tag.

11. Test Discipline

Untested code is unshipped code.

Required:

Unit tests for all new features.

Determinism tests (run twice, compare output).

Oracle invariance across minor releases.

No network or time dependencies.

Regression tests for bug fixes.

Tests define contract.

Prohibited:

Weakening assertions.

Broadening tolerances.

Removing regression tests.

Changing expected hashes without behavioral justification.

Replacing equality with approximate comparisons to hide drift.

Correct code makes tests pass. Tests do not adapt to drift.

12. Release Discipline

Releases are commitments.

Before tagging:

All tests pass.

Determinism verified.

Oracle unchanged (minor release).

Schema unchanged unless approved.

CHANGELOG.md updated.

No decoder drift.

No dependency drift.

No performance regression.

No silent behavioral change.

13. Commit & Push Escalation

Claude may commit/push only if:

All invariants preserved.

Protected subsystems untouched unless authorized.

No schema change without approval.

No identity drift.

No determinism break.

Scope limited to intended feature.

Passing tests alone are insufficient.

Push Escalation Required If Change Touches:

Decoder internals.

Scheduling logic.

Serialization.

Hash derivation.

Schema validation.

Artifact structure.

Determinism guarantees.

If invariants break:

Revert immediately.

Halt speculative fixes.

Request user instruction.

No multi-step guesswork on protected subsystems.

14. Escalation Rule

If a proposed change affects:

Decoder semantics

Scheduling

Schema

Serialization

Hashing

Identity

Artifact structure

Determinism guarantees

Claude must pause and request explicit instruction.

Silence is not consent.

15. Governing Principle

When uncertain:

Preserve stability.

Avoid refactoring.

Prefer doing nothing.

Read before writing.

Maintain invariants.

Capability grows.
Stability does not regress.

If it cannot be reproduced byte-for-byte, it is not a baseline.
