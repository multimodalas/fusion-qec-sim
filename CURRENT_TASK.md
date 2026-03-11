# CURRENT_TASK.md
Active Development Target — v7.6.1

This document defines the **current implementation task** for AI agents.

Only implement features required for **v7.6.1**.

Do not implement future roadmap features.

---

# Goal

Validate that **non-backtracking eigenvector localization predicts BP instability edges**.

This release establishes the empirical foundation for the spectral Tanner-graph research program.

---

# Scientific Hypothesis

BP failures in QLDPC Tanner graphs arise from **cycle-resonant trapping sets**.

These structures create **localized eigenvectors in the non-backtracking spectrum**.

The inverse participation ratio (IPR) measures this localization.

Edges with large eigenvector magnitude should correspond to edges exhibiting **LLR oscillation or instability during BP decoding**.

---

# Features To Implement

### 1. NB Dominant Eigenpair Extraction

Compute the dominant real eigenpair of the non-backtracking operator.

Requirements:

- sparse operator implementation
- no dense matrices
- use `scipy.sparse.linalg.eigs`
- compute only `k=1`
- use `which="LR"`

Output:


nb_spectral_radius
nb_dominant_eigenvector


---

### 2. Inverse Participation Ratio (IPR)

Implement localization metric:


IPR(v) = sum(v_i^4) / (sum(v_i^2))^2


Interpretation:

- low IPR → delocalized eigenvector
- high IPR → localized trapping set

Output:


nb_ipr


---

### 3. Edge Sensitivity Ranking

Rank directed edges using spectral energy:


sensitivity(edge) = |v_i|^2 * |v_j|^2


Output:


ranked_edge_sensitivity


Edges with highest sensitivity are candidates for instability.

---

### 4. Precision@k Validation

Compare spectral ranking against empirically unstable edges.

Procedure:

1. run deterministic BP decode
2. identify edges with highest LLR variance
3. compare with spectral ranking

Metric:


precision_at_k


Target success threshold:


precision_at_k > 0.8


for k roughly equal to trapping set size.

---

# Experiments

Run validation on at least one known LDPC or QLDPC code containing trapping sets.

Example experiment:


run_decoder()
detect_llr_instability_edges()
compare_with_spectral_ranking()


Outputs must be deterministic.

---

# Output Artifacts

The experiment must produce a JSON artifact containing:


{
"nb_spectral_radius": ...,
"nb_ipr": ...,
"precision_at_k": ...,
"num_edges": ...,
"num_instability_edges": ...
}


Artifacts must be canonical JSON.

---

# Explicit Non-Goals

Do NOT implement:

- graph repair algorithms
- spectral heatmaps
- incremental spectral updates
- phase diagram experiments
- ternary decoding experiments

These belong to future releases.

---

# Architectural Constraints

Claude must obey:


CLAUDE.md
IMPLEMENTATION_RULES.md
ALGORITHM_PATTERNS.md
ROADMAP.md


The decoder core must remain untouched.

All work must occur in:


src/qec/diagnostics/
src/qec/experiments/


---

# Success Condition

v7.6.1 is complete when:

1. spectral diagnostics run deterministically
2. IPR is computed
3. edge sensitivity ranking is produced
4. Precision@k experiment validates correlation
5. artifacts are reproducible

Only then may the project move to **v7.7.0 heatmaps**.
