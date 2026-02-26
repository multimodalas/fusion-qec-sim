# CLAUDE.md — Governance Contract for AI-Assisted Development

This file defines persistent architectural and behavioral constraints for Claude
when working in the QEC repository. It is NOT a README. It is a binding set of
rules that must be respected in every code generation, refactor, and review
action performed by Claude.

The goal is to preserve determinism, decoder stability, schema governance,
minimal dependency surface, clean layering, and reproducibility guarantees.

---

## 1. Determinism as Architecture

Determinism is not a feature. It is a structural invariant.

- No hidden randomness. All stochastic operations must use an explicit seed.
- `runtime_mode="off"` must produce byte-identical artifacts across runs.
- Canonical JSON serialization must not change between versions.
- Sweep hash must derive exclusively from config — no ambient state.
- Artifact hash must reflect the immutable record state and nothing else.
- Never introduce floating-point non-determinism (e.g., unordered reductions).
- If a function is deterministic today, it must remain deterministic tomorrow.

---

## 2. Decoder Core Protection

The decoder is the most critical subsystem. Treat it as frozen unless told
otherwise.

- Files under `src/qec/` must NOT be modified unless explicitly requested.
- No refactors to belief-propagation (BP) logic.
- No changes to scheduling semantics (e.g., iteration order, message passing).
- No performance "optimizations" without explicit instruction.
- Minor releases must preserve bit-identical decoding behavior.
- Do not introduce new code paths into the decoder without approval.
- Do not alter function signatures in decoder modules.

---

## 3. Channel Layer Discipline

Channel models are consumers of the decoder, not peers.

- Channel models must not mutate decoder internals or state.
- `"oracle"` mode must remain backward-compatible across all versions.
- New channel modes must be strictly additive — no removal, no renaming.
- No silent behavior drift. If channel output changes, it must be documented.
- Channel configuration must not alter decoder scheduling or convergence.

---

## 4. Schema Governance Rules

Schema changes are high-risk. Treat them accordingly.

- Do NOT bump `SCHEMA_VERSION` without explicit instruction from the user.
- Minor versions allow additive fields only — no removals, no type changes.
- Backward compatibility is the default. Breaking changes require major bumps.
- Validation must remain strict. Do not loosen validators to accommodate drift.
- All schema fields must be documented at the point of introduction.

---

## 5. Benchmark / Interop Isolation

Benchmarking and interop code must not leak into core logic.

- Code in `src/bench/` must not alter decoding semantics in any way.
- Third-party tools (e.g., external decoders, plotting libraries) must remain
  optional and gated behind feature flags or lazy imports.
- Benchmark harnesses must not modify global state.
- Interop adapters must not introduce implicit dependencies on external formats.

---

## 6. Minimal Diff and Refactor Discipline

Every diff must justify its existence.

- No large refactors unless explicitly requested.
- No variable renaming unless functionally required.
- No style-only rewrites (whitespace, import order, quote style).
- Commit in small, logical increments — one concern per commit.
- Do not move code between modules without instruction.
- Do not introduce abstraction layers preemptively.
- Do not add docstrings, type annotations, or comments to unchanged code.

---

## 7. Dependency Policy

The dependency surface must remain minimal.

- No new external dependencies without explicit user approval.
- Prefer the Python standard library and NumPy for all operations.
- Avoid convenience packages (e.g., `attrs`, `pydantic`, `click`) unless
  already present in the project.
- Do not upgrade existing dependencies without instruction.
- All dependencies must be pinned or bounded in version specifications.

---

## 8. Versioning Policy

Version numbers encode architectural meaning.

- Format: `vArchitecture.Major.Minor` (e.g., `v1.2.3`).
- **Minor** increments are additive only — no behavioral changes.
- **Major** increments signal architectural shifts or breaking changes.
- **Architecture** increments are reserved for foundational redesigns.
- Never bump a version number without explicit instruction.

---

## 9. Testing Requirements

Untested code is unshipped code.

- All new features require accompanying unit tests.
- Determinism must be verified: run the same config twice, compare outputs.
- Oracle behavior must remain unchanged across minor version boundaries.
- Tests must not depend on network access, wall-clock time, or randomness.
- Regression tests must accompany every bug fix.
- Do not delete or weaken existing tests without explicit approval.

---

## 10. Release Discipline

Releases are commitments, not milestones.

- No silent behavior changes. Every observable change must be documented.
- All changes must be recorded in `CHANGELOG.md` before release.
- Deterministic baselines must remain reproducible after any release.
- Release artifacts must be hash-verifiable against their generating config.
- Do not tag a release without confirming test suite passage.

---

## 11. When in Doubt

If a proposed change is ambiguous, follow these defaults:

- **Preserve stability.** Do not change what works.
- **Avoid refactoring.** Leave structure alone unless instructed.
- **Ask before modifying core logic.** The decoder, schema, and serialization
  paths are protected by default.
- **Prefer doing nothing** over introducing speculative improvements.
- **Read before writing.** Understand existing code before proposing changes.
