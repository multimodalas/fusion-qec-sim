# v6.0 — BP Stability Spectra

Spectral stability diagnostics for belief propagation decoding.

---

## 1. Motivation

Belief propagation (BP) decoding on LDPC and QLDPC codes exhibits
complex dynamical behavior: convergence, oscillation, trapping, and
divergence.  Previous diagnostics (v5.1–v5.9) characterized these
regimes empirically through energy landscapes, phase-space
trajectories, and ternary topology classification.

v6.0 adds a complementary theoretical layer: spectral stability
proxies derived from graph-theoretic operators.  These diagnostics
connect empirical decoder phase diagrams to well-studied conditions
for BP convergence and instability from the theory of message passing
on sparse graphs.

---

## 2. Non-Backtracking Matrices in BP

The non-backtracking (Hashimoto) matrix B is defined on directed
edges of the Tanner graph.  For directed edges (u → v) and (v → w):

    B_{(u→v), (v→w)} = 1  if w ≠ u

Unlike the adjacency matrix, the non-backtracking operator excludes
immediate message reversal, making its spectrum a more faithful proxy
for BP message-passing dynamics.

Key property: the spectral radius of B governs the rate of information
propagation on the graph.  When the spectral radius is large, BP
messages amplify rapidly, increasing the risk of oscillation or
divergence.

Implementation: `compute_non_backtracking_spectrum(H)` in
`src/qec/diagnostics/non_backtracking_spectrum.py`.

---

## 3. Bethe Hessian Interpretation

The Bethe Hessian matrix is defined as:

    H_B = (r² − 1)I − rA + D

where A is the Tanner graph adjacency matrix, D is the degree matrix,
and r is a regularization parameter (defaulting to an approximation
of √(NB spectral radius)).

The Bethe Hessian provides a spectral characterization of detectable
structure in sparse graphs:

- Negative eigenvalues indicate structural communities or degeneracies
  that BP may struggle to resolve.
- The minimum eigenvalue measures the severity of structural
  degeneracy.
- The count of negative eigenvalues indicates the number of
  distinguishable structural modes.

In the context of BP decoding, negative Bethe Hessian eigenvalues
may signal regimes where the decoder faces competing attractors or
degenerate solutions.

Implementation: `compute_bethe_hessian(H, r=None)` in
`src/qec/diagnostics/bethe_hessian.py`.

---

## 4. Jacobian Spectral Estimation Trick

The BP Jacobian describes how small perturbations to messages propagate
through one iteration of BP.  Its spectral radius determines local
stability: if ρ(J) < 1, the fixed point is locally stable; if
ρ(J) > 1, it is unstable.

Constructing the full Jacobian is expensive (O(E²) where E is the
number of edges).  Instead, we estimate the dominant eigenvalue using
a power-iteration heuristic on consecutive LLR differences:

    Δx_t = x_{t+1} − x_t
    λ_est ≈ ‖Δx_{t+1}‖ / ‖Δx_t‖

This ratio approximates the spectral radius when BP is near a fixed
point and the dominant eigenmode dominates the dynamics.

The estimate is averaged over the tail iterations for robustness.

Implementation: `estimate_bp_jacobian_spectral_radius(llr_history)` in
`src/qec/diagnostics/bp_jacobian_estimator.py`.

---

## 5. Integration with Phase Diagrams

The spectral diagnostics integrate with the v5.9 phase diagram
framework as optional overlays.  When enabled, trial results include:

- `spectral_radius`: NB matrix spectral radius
- `bethe_min_eigenvalue`: minimum Bethe Hessian eigenvalue
- `bp_stability_score`: combined stability proxy
- `jacobian_spectral_radius_est`: estimated Jacobian spectral radius

The phase diagram aggregator computes per-cell means:

- `mean_spectral_radius`
- `mean_bethe_min_eigenvalue`
- `mean_bp_stability_score`
- `mean_jacobian_spectral_radius_est`

These fields appear only when spectral diagnostics are enabled.
Existing v5.9 fields are fully preserved — the integration is
strictly additive.

CLI flags: `--nb-spectrum`, `--bethe-hessian`, `--bp-stability`,
`--bp-jacobian-estimator`.

---

## 6. Deterministic Design Constraints

All spectral diagnostics follow the architectural invariants:

- **No randomness**: all computations are deterministic.
- **No decoder modification**: diagnostics treat the decoder as a pure
  function.  The decoder behaves identically whether diagnostics are
  enabled or disabled.
- **No global state**: no hidden mutable state.
- **JSON-serializable outputs**: all results are JSON-safe dicts with
  native Python types.
- **Deterministic ordering**: eigenvalues are sorted by magnitude
  (NB) or value (Bethe Hessian) with deterministic tie-breaking.
- **No input mutation**: all input arrays are read-only.

---

## 7. Experimental Protocol

To run spectral stability analysis alongside phase diagrams:

```bash
python bench/dps_v381_eval.py \
  --phase-diagram \
  --nb-spectrum \
  --bethe-hessian \
  --bp-stability \
  --bp-jacobian-estimator \
  --phase-diagram-output phase_v60.json \
  --distances 3 5 7 \
  --p-values 0.01 0.03 0.05
```

Or use the standalone demo:

```bash
python scripts/run_v59_phase_diagram_demo.py
```

The demo now includes spectral diagnostics by default and prints an
ASCII phase heatmap.

---

## 8. Limitations

- **Non-backtracking spectrum is O(E²)**: the matrix size scales as
  the number of directed edges squared.  For large codes, this may be
  computationally expensive.
- **Bethe Hessian r approximation**: the default r uses an
  approximation (√(√(largest adjacency eigenvalue))) rather than the
  exact NB spectral radius.  For small codes the difference is
  negligible.
- **Jacobian estimator assumes near-fixed-point**: the power
  iteration ratio is most accurate when BP is near convergence.
  During early iterations or oscillatory regimes, the estimate may
  be noisy.
- **Stability score is a proxy**: the combined score
  (1/ρ_NB) × λ_min(H_B) is a heuristic indicator, not a rigorous
  stability criterion.
- **Observational only**: these diagnostics provide correlates of
  BP instability, not causal explanations.

---

## Research Hook — Non-Backtracking Localization (v6.1)

Eigenvectors of the non-backtracking matrix for Tanner graphs often
localize on small substructures that resemble trapping sets or
absorbing sets.

A standard way to measure eigenvector localization is the inverse
participation ratio (IPR):

    IPR(v) = Σ_i |v_i|⁴ / (Σ_i |v_i|²)²

Interpretation:
- Small IPR → eigenvector spread across graph (delocalized)
- Large IPR → eigenvector localized on few nodes

Localized modes with high IPR may indicate:
- Trapping sets
- Fragile subgraphs
- Likely BP failure regions

**Implemented in v6.1.0** — see `src/qec/diagnostics/nb_localization.py`.

```python
compute_nb_localization_metrics(parity_check_matrix)
```

Output fields:

```json
{
  "ipr_scores": [...],
  "max_ipr": float,
  "localized_modes": [...],
  "mode_support_sizes": [...],
  "localized_edge_indices": [[...]],
  "localized_variable_nodes": [[...]],
  "localized_check_nodes": [[...]],
  "top_localization_score": float,
  "per_mode_mass_on_variables": [...],
  "per_mode_mass_on_checks": [...],
  "num_directed_edges": int,
  "num_leading_modes": int
}
```

Mode selection: top-k leading eigenmodes by magnitude (default k=6).
Localization rule: relative magnitude threshold (default 0.1 × max |v|²).
IPR localization threshold: default 2/num_directed_edges (twice uniform).
Projects localized edge support back to Tanner graph variable and check nodes.

---

## Research Note — Why This May Be Especially Powerful for QLDPC

Non-backtracking localization may be especially informative for QLDPC
Tanner graphs for the following reasons:

- QLDPC Tanner graphs often contain unavoidable short cycles and
  degeneracy-related substructures due to the CSS construction and
  commutativity constraints.
- These structural features can create localized feedback regions for
  iterative decoding where BP messages reinforce incorrect beliefs.
- Non-backtracking operators are more faithful to BP message flow
  than plain adjacency spectra because they exclude immediate
  reversal — this means they capture the actual information
  propagation structure more accurately.
- Therefore, localized non-backtracking eigenmodes may reveal fragile
  quantum trapping structures more directly than ordinary Tanner
  adjacency eigenvectors.

**This is a research hypothesis / future direction, not a proven
result.**  Empirical validation is needed to confirm whether NB
eigenvector localization is a reliable predictor of QLDPC decoder
fragility.
