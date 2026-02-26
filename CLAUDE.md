# CLAUDE.md — Architectural Governance Contract

This file defines binding architectural and behavioral constraints for Claude
when working in the QEC repository. It is NOT a README. It is a constitution
that must be respected in every code generation, refactor, review, and release
action performed by Claude.

Goals: determinism, decoder stability, strict layering, schema governance,
minimal dependency surface, and reproducibility guarantees.

---

## 1. Architectural Layering Model

The codebase enforces a strict dependency hierarchy. Violations are forbidden.

| Layer | Path             | Role                        |
|-------|------------------|-----------------------------|
| 1     | `src/qec/`       | Core decoding (protected)   |
| 2     | `src/qec/channel/`| Channel abstraction          |
| 3     | `src/bench/`     | Benchmarking and interop    |

Rules:

- Lower layers must never depend on higher layers.
- `src/qec/` must never import from `src/bench/` or interop tooling.
- `src/qec/channel/` is a consumer of Layer 1 — it must not mutate decoder
  internals, alter scheduling, or inject state into the decoder.
- `src/bench/` is a consumer only — it must not alter decoding semantics.
- No upward dependency leakage. If a dependency direction is ambiguous, it is
  forbidden until clarified.

---

## 2. Determinism as Architecture

Determinism is not a feature. It is a structural invariant.

- No hidden randomness. All stochastic operations must use an explicit seed.
- `runtime_mode="off"` must produce byte-identical artifacts across runs.
- Canonical JSON serialization must not change between versions.
- Sweep hash must derive exclusively from config — no ambient state.
- Never introduce floating-point non-determinism (e.g., unordered reductions).
- If a function is deterministic today, it must remain deterministic tomorrow.

---

## 3. Seed Discipline

Randomness is permitted only under strict controls.

- All randomness must be explicitly seeded at the call site.
- No use of global RNG state. No implicit `random.seed()` or `np.random.seed()`.
- No use of `random` or `np.random` without explicit seed injection.
- Seed derivation must remain order-independent.
- Sub-seed derivation must use SHA-256 or an equivalent deterministic mapping.
- Functions that accept a seed must propagate it to all internal stochastic
  operations without loss.
- Python's built-in `hash()` is strictly forbidden for any structural, ordering,
  or seed-derivation logic. It is salted per-session and breaks determinism.
- Always use `hashlib.sha256` (or equivalent cryptographic hash) for seed
  derivation, sweep hashing, or artifact identity.
- No reliance on object hash stability across runs or Python versions.

---

## 4. Artifact Immutability

Artifacts, once hashed, are immutable records.

- Artifact hash must reflect the final immutable record state and nothing else.
- No post-hash mutation of any artifact field.
- Deterministic metadata (timestamps excluded) must remain stable across
  identical runs.
- Report generation must not reorder data nondeterministically.
- Serialization order must be canonical and reproducible.

---

## 5. Decoder Core Protection

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

## 6. Protected Subsystems

The following subsystems require explicit user instruction before any
modification:

- `src/qec/decoder` — Decoder core and BP logic.
- `src/qec/scheduling` — Iteration order and message-passing schedule.
- `canonicalize()` — Canonical serialization function.
- Schema validation logic — Field validators, type enforcers.
- Sweep hashing logic — Config-to-hash derivation.

If a proposed change touches any of these, Claude must ask the user before
proceeding. No exceptions.

---

## 7. Channel Layer Discipline

Channel models are consumers of the decoder, not peers.

- Channel models must not mutate decoder internals or state.
- `"oracle"` mode must remain backward-compatible across all versions.
- New channel modes must be strictly additive — no removal, no renaming.
- No silent behavior drift. If channel output changes, it must be documented.
- Channel configuration must not alter decoder scheduling or convergence.

---

## 8. Schema Governance Rules

Schema changes are high-risk. Treat them accordingly.

- Do NOT bump `SCHEMA_VERSION` without explicit instruction from the user.
- Minor versions allow additive fields only — no removals, no type changes.
- Backward compatibility is the default. Breaking changes require major bumps.
- Validation must remain strict. Do not loosen validators to accommodate drift.
- All schema fields must be documented at the point of introduction.
- All JSON serialization and schema output must use the centralized
  canonicalization logic in `src/utils/canonicalize.py`.
- Do not implement ad-hoc `json.dumps(sort_keys=True)` logic elsewhere.
- Do not duplicate canonicalization behavior. Serialization format is locked
  and fuzz-validated.
- Any change to canonicalization requires explicit user approval.

---

## 9. Benchmark / Interop Isolation

Benchmarking and interop code must not leak into core logic.

- Code in `src/bench/` must not alter decoding semantics in any way.
- Third-party tools (e.g., external decoders, plotting libraries) must remain
  optional and gated behind feature flags or lazy imports.
- Benchmark harnesses must not modify global state.
- Interop adapters must not introduce implicit dependencies on external formats.
- Claude must strictly adhere to the policies defined in
  `docs/INTEROP_POLICY.md` and `docs/LEGAL_THIRD_PARTY.md`.
- No proprietary decoders, reverse-engineering, or unapproved dependencies may
  be introduced through interop or benchmarking paths.
- Third-party tools must remain optional, gated, and policy-compliant.
- Legal compliance constraints are binding architectural rules, not advisory.

---

## 10. Minimal Diff and Refactor Discipline

Every diff must justify its existence. The following are explicitly prohibited
without direct user request:

- Large refactors or structural reorganizations.
- Variable or function renaming unless functionally required.
- Style-only rewrites (whitespace, import order, quote style).
- Renaming modules or packages.
- Collapsing or splitting abstraction layers.
- "Improving structure" without explicit request.
- Reformatting entire files.
- Reordering imports globally.
- Adding docstrings, type annotations, or comments to unchanged code.
- Moving code between modules.
- Introducing abstraction layers preemptively.

Commit in small, logical increments — one concern per commit.

---

## 11. Dependency Policy

The dependency surface must remain minimal.

- No new external dependencies without explicit user approval.
- Prefer the Python standard library and NumPy for all operations.
- Avoid convenience packages (e.g., `attrs`, `pydantic`, `click`) unless
  already present in the project.
- Do not upgrade existing dependencies without instruction.
- All dependencies must be pinned or bounded in version specifications.

---

## 12. Versioning Policy

Version numbers encode architectural meaning.

- Format: `vArchitecture.Major.Minor` (e.g., `v1.2.3`).
- **Architecture** increments are reserved for foundational redesigns.
- **Major** increments signal breaking changes or decoder behavior changes.
- **Minor** increments are additive only.
- Never bump a version number without explicit instruction.

### Minor Version Constraints

- May add new features and configuration options.
- Must not change default behavior of existing configurations.
- Must not alter decoding outputs for the same config and seed.
- Must not introduce schema-breaking changes.

### Major Version Requirements

- Required for any change to decoder behavior.
- Required for any schema-breaking change.
- Required for removal or renaming of public interfaces.

---

## 13. Performance Stability

Performance is a correctness property in minor releases.

- Minor versions must not introduce measurable decoding slowdown unless
  explicitly requested.
- No algorithmic complexity changes in minor releases.
- Performance regressions must be documented and justified.
- Benchmarking comparisons against the prior release are expected before
  tagging.

---

## 14. Testing Requirements

Untested code is unshipped code.

- All new features require accompanying unit tests.
- Determinism must be verified: run the same config twice, compare outputs.
- Oracle behavior must remain unchanged across minor version boundaries.
- Tests must not depend on network access, wall-clock time, or randomness.
- Regression tests must accompany every bug fix.
- Do not delete or weaken existing tests without explicit approval.

---

## 15. Test Integrity and Specification Discipline

Tests define the contract of the system. They are not negotiable scaffolding.

- Tests must not be modified solely to make them pass. A failing test indicates
  incorrect code or an intentional behavioral change — not a test defect by
  default.
- Intentional behavioral changes that require test updates must satisfy all of
  the following:
  - Explicit user instruction authorizing the change.
  - A corresponding `CHANGELOG.md` update documenting the new behavior.
  - A clear rationale explaining why the prior behavior was incorrect or
    superseded.

The following are explicitly prohibited without direct user request:

- Weakening assertions (e.g., replacing exact checks with range checks).
- Broadening tolerances to mask numerical or behavioral drift.
- Changing expected artifact hashes without a justified behavioral change.
- Replacing strict equality with approximate comparisons.
- Removing regression tests to silence failures.
- Disabling determinism checks or hash verification to make CI pass.

Correct code makes tests pass. Tests do not adapt to accidental behavior.

---

## 16. Release Discipline

Releases are commitments, not milestones.

- No silent behavior changes. Every observable change must be documented.
- All changes must be recorded in `CHANGELOG.md` before release.
- Deterministic baselines must remain reproducible after any release.
- Release artifacts must be hash-verifiable against their generating config.
- Do not tag a release without confirming test suite passage.

---

## 17. Commit and Push Discipline

Commits and pushes are controlled actions, not routine automation.

Claude may only commit and push when ALL of the following are true:

- All tests pass without weakening or modifying assertions to force success.
- Determinism guarantees remain intact.
- No protected subsystems were modified without explicit user instruction.
- No schema versions were bumped without instruction.
- No dependency changes were introduced without approval.
- No silent behavior change occurred in default configurations.
- A scope audit confirms changes are limited to the intended feature.

Passing tests alone are necessary but not sufficient. Correctness must be
reasoned about, not assumed.

- Do not push speculative or partially audited work.
- Do not push changes justified only by "tests are green."

### Revert / Abort Protocol

If an implementation causes any of the following:

- Core decoder tests to fail.
- Determinism invariants to break.
- Schema validation to fail.
- Canonicalization invariants to drift.

Claude must immediately revert to the last known-good state (via `git revert` or
`git checkout`), halt further speculative fixes, and request explicit user
guidance. Do not attempt multi-step speculative repairs on protected subsystems.

### Push Escalation Rule

If a change affects any of the following, Claude must request explicit user
authorization before pushing:

- Decoder internals.
- Scheduling logic.
- Serialization.
- Hashing.
- Schema validation.
- Artifact structure.
- Determinism guarantees.

Push only after invariant verification and structural audit.

---

## 18. Pre-Release Checklist

Before Claude may tag or propose a release, every item must be verified:

- [ ] All tests pass.
- [ ] Determinism verified (identical config + seed produces identical output).
- [ ] Oracle behavior unchanged (if minor release).
- [ ] Schema unchanged unless explicitly approved.
- [ ] `CHANGELOG.md` updated with all observable changes.
- [ ] No decoder drift from prior release baseline.
- [ ] No new external dependencies introduced without approval.
- [ ] No performance regressions in minor releases.

---

## 19. When in Doubt

If a proposed change is ambiguous, follow these defaults:

- **Preserve stability.** Do not change what works.
- **Avoid refactoring.** Leave structure alone unless instructed.
- **Prefer doing nothing** over introducing speculative improvements.
- **Read before writing.** Understand existing code before proposing changes.

### Escalation Rule

If a change touches any of the following, Claude must ask the user before
proceeding:

- Decoder logic or scheduling.
- Schema structure or validation.
- Serialization or canonicalization.
- Hashing (sweep hash, artifact hash, sub-seed derivation).
- Artifact structure or immutability guarantees.

No silent modifications to protected subsystems. Ever.
