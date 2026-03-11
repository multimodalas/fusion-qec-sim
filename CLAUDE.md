CLAUDE.md

QSOL QEC Architectural Constitution (v8 Hardened)

This document governs all AI-assisted activity inside the QSOL QEC repository.

It applies to Claude when performing:

code generation

refactoring

testing

experiments

commits

release preparation

This document is not guidance.

It is the architectural constitution of the repository.

Claude must obey these rules when operating inside this codebase.

Core Values

The QEC framework is governed by six non-negotiable principles.

Determinism

All experiments must be reproducible byte-for-byte.

Decoder Stability

The decoder core is protected infrastructure.

Architectural Layering

The repository follows strict dependency layers.

Minimal Complexity

Prefer simple deterministic algorithms.

Scientific Transparency

Algorithms must remain explainable.

Reproducibility Guarantees

Artifacts must remain stable across runs and releases.

1. Architectural Layer Model (Non-Negotiable)

The repository follows strict directional layering.

Dependencies may only flow downward.

Layer	Path	Role
1	src/qec/decoder/	Decoder core (protected)
2	src/qec/channel/	Channel and noise models
3	src/qec/diagnostics/	Deterministic diagnostics
4	src/qec/predictors/	Structural instability predictors
5	src/qec/experiments/	Experimental policies
6	src/bench/	Benchmark harness

Rules:

Lower layers must never import higher layers

decoder must never import experiments

src/qec/ must never import src/bench/

predictors must not depend on experiments

Layer boundaries are architectural invariants.

Violating them is forbidden without explicit approval.

2. Determinism is Architecture

Determinism is a structural requirement.

All code must preserve deterministic execution.

Required invariants:

no hidden randomness

no implicit RNG state

explicit seed injection

deterministic collection ordering

no use of Python hash()

no floating-point drift due to unordered reductions

Sub-seed derivation must use:

SHA-256 deterministic hashing

Artifact serialization must be canonical.

When:

runtime_mode = "off"

outputs must be byte-identical across runs.

3. Artifact & Identity Stability

Artifacts represent immutable experiment records.

Rules:

artifacts must not mutate after hashing

serialization order must remain canonical

decoder identity must be stable across identical configs

identity must not depend on memory layout

Identity drift without version bump is forbidden.

4. Decoder Core Protection

The decoder is the most protected subsystem.

Protected path:

src/qec/decoder/

Default rule:

Do not modify the decoder core.

Prohibited without explicit instruction:

modifying BP message updates

altering scheduling semantics

changing iteration ordering

refactoring decoder internals

introducing adaptive behaviour

The decoder must remain bit-stable across minor releases.

5. Spectral Research Model

The QEC framework studies belief-propagation instability on Tanner graphs.

The research pipeline is:

Tanner graph
↓
spectral diagnostics
↓
instability localization
↓
graph repair
↓
decoder experiments

Spectral signals include:

non-backtracking spectral radius

dominant NB eigenvector

inverse participation ratio (IPR)

spectral trapping-set indicators

Claude must implement research features in the order:

measure → localize → repair → accelerate → map → mitigate

Corresponding roadmap releases:

v7.6.1 validation
v7.7.0 heatmaps
v7.8.0 graph repair
v7.9.0 incremental spectra
v8.0.0 phase diagrams
v8.1.0 ternary mitigation

Claude must not skip steps in this research sequence.

6. Diagnostics Layer Protection

Diagnostics are observational instruments.

They must never influence decoder behaviour.

Diagnostics must be:

deterministic

side-effect free

opt-in

Diagnostics must not:

modify BP messages

alter decoder inputs

change iteration ordering

mutate arrays in-place

Diagnostics may run additional decoding experiments using copied inputs only.

7. Predictor Layer Protection

Predictors estimate instability before decoding runs.

Predictors must:

operate only on diagnostics outputs

remain deterministic

produce informational signals only

Example signals:

bp_failure_risk
predicted_instability
spectral_instability_ratio

Predictors must not modify decoder inputs.

8. Controller Layer Protection

Controllers run controlled decoding experiments.

Controllers may:

alter experiment configuration

modify input LLR vectors

run multiple decode passes

Controllers must not:

modify decoder implementation

patch decoder functions

introduce stochastic behaviour

Controllers wrap decoder calls externally.

9. Sparse Linear Algebra Rules

Spectral algorithms must scale to large QLDPC graphs.

Forbidden:

dense Hashimoto matrix construction

numpy.linalg.eig on NB matrices

Required approach:

sparse operators

Krylov eigensolvers

scipy.sparse.linalg.eigs

linear operator interfaces

Memory must scale with |E|, not |E|².

10. Tanner Graph Constraint Protection

QLDPC graphs obey strict commutativity constraints:

H_X H_Z^T = 0

Graph repair algorithms must preserve these constraints.

Edge swaps must be rejected if they break stabilizer commutativity.

11. Minimal Diff Discipline

Changes must be minimal and targeted.

Forbidden without instruction:

large refactors

renaming identifiers

style-only edits

import reordering

file-wide formatting

moving code between modules

Commits must be small and single-purpose.

12. Dependency Policy

Dependency surface must remain minimal.

Rules:

prefer stdlib

prefer NumPy / SciPy

no new dependencies without approval

no framework introduction

Architectural bloat is forbidden.

13. Test Discipline

Untested code is unshipped code.

Required:

unit tests

determinism tests

regression tests

artifact validation tests

Tests must not hide drift by widening tolerances.

Correct code makes tests pass.

Tests do not adapt to drift.

14. Commit & Push Discipline

Claude may commit only when:

determinism preserved

decoder untouched

schema unchanged

identity stable

tests passing

Push escalation is required for changes touching:

decoder

schema

serialization

hashing

determinism guarantees

Passing tests alone are insufficient.

15. Escalation Rule

If a proposed change affects:

decoder semantics

scheduling

schema

serialization

hashing

artifact identity

determinism guarantees

Claude must:

Stop

Explain the risk

Request explicit instruction

Silence is not consent.

16. Governing Principle

When uncertain:

Preserve stability.

Avoid refactoring.

Prefer doing nothing.

Read before writing.

Maintain invariants.

Capability grows.
Stability does not regress.

If a result cannot be reproduced byte-for-byte, it is not a baseline.

CLAUDE.md Addition — Autonomous Safe PR Workflow

Add a section like this to CLAUDE.md.

## Autonomous PR Workflow

When modifying the repository, Claude Code must follow this workflow.

### 1. Branch Creation

Create a new branch for each task.

Example:


git checkout -b claude/<task-name>


Branches should be small and focused.

Examples:


claude/add-spectral-entropy
claude/fix-nb-localization
claude/refactor-diagnostics-api


---

### 2. Pre-Commit Validation

Before committing changes, Claude must run:


pytest


If tests fail, Claude must fix the issue before committing.

No commits should be pushed with failing tests.

---

### 3. Pull Latest Main

Before pushing a branch, Claude must update the branch with the latest
`main`.

Preferred method:


git fetch origin
git rebase origin/main


If conflicts occur:

- resolve conflicts
- remove conflict markers
- run tests again

---

### 4. Commit Style

Commits should be concise and scoped.

Example:


git commit -m "diagnostics: add spectral entropy metric"


---

### 5. Push Branch

Push the branch to origin.


git push origin claude/<task-name>


---

### 6. Pull Request Rules

Claude should open a PR with:

Title format:


module: short description


Example:


diagnostics: add spectral entropy metric


PR description should include:

- purpose
- files modified
- tests added

---

### 7. Merge Safety Rules

Claude must **not merge PRs automatically** unless:

- tests pass
- no merge conflicts exist
- changes are limited to diagnostics or experiments layers

Never merge changes that modify:

- decoder core
- BP algorithm
- parity check generation

These require human review.

---

### 8. Merge Strategy

Preferred merge strategy:


Squash and merge


This keeps the history clean.

---

### 9. Post-Merge Cleanup

After merging:


git branch -d claude/<task-name>


---

### 10. Conflict Marker Safety Check

Before committing, Claude must verify the repository does not contain
merge markers.

Search for:

<<<<<<<

If any markers exist, resolve them before committing.

Never modify `src/qec/decoder/` without explicit human instruction.
