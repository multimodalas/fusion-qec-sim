IMPLEMENTATION_RULES.md

QSOL QEC Engineering Rules for AI-Assisted Development

This document defines practical implementation discipline for AI agents operating inside the QSOL QEC repository.

It complements CLAUDE.md.

Where CLAUDE.md defines architectural law, this document defines engineering practice.

1. Read Before Writing

Before modifying code, Claude must:

Read the relevant modules

Identify architectural layer boundaries

Confirm changes do not violate CLAUDE.md

Propose minimal changes

Claude must never write code before understanding the module.

2. Minimal Change Principle

Changes must be surgical and local.

Rules:

modify the smallest number of lines possible

avoid refactoring unrelated code

avoid renaming identifiers

avoid file-wide formatting changes

do not restructure modules without explicit instruction

Large diffs increase architectural risk.

Small diffs preserve stability.

3. Deterministic Implementation Patterns

All algorithms must be deterministic.

Required patterns:

explicit seed parameters

deterministic iteration ordering

stable sorting for ranked outputs

no implicit randomness

no global state

Avoid:

random.random()
np.random without seed
unordered set iteration
dict iteration without sorting
4. Sparse Linear Algebra First

Many algorithms in this repository involve large Tanner graphs.

Dense matrix construction is forbidden for spectral operators.

Never build matrices of size:

2|E| × 2|E|

Instead use:

sparse adjacency structures

scipy.sparse

scipy.sparse.linalg

LinearOperator interfaces

Example pattern:

scipy.sparse.linalg.eigs(
    LinearOperator(...),
    k=1,
    which="LR"
)

Memory must scale with |E|, not |E|².

5. Avoid Premature Optimization

Algorithms should first be implemented in their minimal correct form.

Performance optimizations should only occur after:

correctness

determinism

reproducibility

Do not introduce complex optimization prematurely.

6. Avoid Heuristic Algorithms

This repository prioritizes mathematically justified methods.

Avoid:

machine learning heuristics

stochastic optimization

simulated annealing

reinforcement learning

Preferred methods:

spectral analysis

linear algebra

deterministic graph algorithms

provable perturbation methods

Algorithms must remain explainable.

7. Preserve Graph Constraints

QLDPC Tanner graphs must satisfy stabilizer commutativity constraints:

H_X H_Z^T = 0

Any graph modification must:

preserve row degrees

preserve column degrees

preserve commutativity constraints

If a proposed operation might break these conditions:

Claude must abort the change and request confirmation.

8. Experimental Modules Must Be External

Research experiments must live in:

src/qec/experiments/

Experiments must never:

modify decoder internals

patch decoder functions

inject hooks into BP loops

Experiments must operate as external wrappers.

9. Input Modification is Allowed (With Care)

Experimental controllers may modify:

channel LLR vectors

decoding schedules

graph structure (if valid)

But must never modify:

internal BP message updates

decoder iteration logic

check-node update rules

The decoder remains a black box.

10. Logging and Artifacts

All experiments must produce deterministic artifacts.

Artifacts must include:

configuration snapshot

deterministic seeds

experiment parameters

results

Artifacts must use canonical serialization.

11. Testing Expectations

All new features require tests.

Minimum test coverage:

deterministic behavior

numerical stability

schema validation

regression tests

Tests must verify algorithm correctness, not just execution.

12. Safe Spectral Algorithm Implementation

When implementing spectral algorithms:

Use sparse operators

Target the dominant real eigenvalue

Avoid full spectrum computation

Validate convergence tolerance

Typical pattern:

eigs(..., k=1, which="LR")

Never compute the entire spectrum unless explicitly required.

13. Graph Repair Safety

When implementing Tanner graph repair:

Rules:

preserve degree distribution

preserve stabilizer commutativity

verify graph validity after each modification

Repair loops must remain deterministic.

Forbidden methods:

stochastic edge swaps

simulated annealing

random rewiring

14. Phase Diagram Experiments

Large experiments must be structured.

Pattern:

for graph in graph_set:
    for noise in noise_levels:
        run_decoder()
        record_results()

Outputs should be simple data artifacts such as:

CSV

JSON

structured logs

Visualization is not part of the core experiment code.

15. Error Handling Discipline

Errors must be explicit.

Preferred pattern:

raise ValueError("Invalid Tanner graph configuration")

Avoid silent fallbacks.

Avoid implicit correction.

Incorrect inputs must fail loudly.

16. Documentation Discipline

Every new module must include:

purpose

inputs

outputs

algorithm description

Comments should explain why the algorithm exists, not just what it does.

17. Commit Discipline

Each commit should:

implement a single feature

include tests

preserve determinism

avoid unrelated changes

Commits must remain easy to review.

18. When Uncertain

If Claude encounters ambiguity:

Stop writing code

Explain the uncertainty

Request clarification

Never guess architectural intent.

Final Principle

This repository is a deterministic scientific instrument.

Code must prioritize:

reproducibility

transparency

stability

mathematical clarity

Capability grows.

Stability does not regress.
