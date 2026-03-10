# Changelog

All notable changes to this project are documented in this file.

This project follows Semantic Versioning (SemVer).

---

[6.4.0] — 2026-03-10
Spectral Failure Risk Scoring

Added

Spectral Failure Risk Scoring (src/qec/diagnostics/spectral_failure_risk.py):

compute_spectral_failure_risk(): computes a deterministic structural risk
heuristic for candidate clusters by combining spectral localization strength,
repeated participation in localized modes, and alignment with BP dynamical
activity.  Identifies clusters most likely to influence decoder behavior based
on spectral and dynamical signals accumulated in v6.1–v6.3.

Risk score formula per cluster C:

  cluster_risk = participation_weight * alignment_score * localization_weight

Where:
  participation_weight = mean(node_participation_counts for nodes in C)
  alignment_score = per_cluster_alignment_score for C (from v6.3)
  localization_weight = mean(IPR of localized modes) (from v6.1)

Node-level risk: node_risk(v) = sum(cluster_risk for clusters containing v).

Output fields: node_risk_scores, cluster_risk_scores, cluster_risk_ranking,
max_cluster_risk, mean_cluster_risk, top_risk_clusters, num_high_risk_clusters.

Phase Diagram Risk Overlays (src/qec/diagnostics/phase_diagram.py):

Extended phase diagram cell aggregation with optional risk fields:
mean_cluster_risk, max_cluster_risk, mean_num_high_risk_clusters.
Fields are additive — appear only when risk diagnostics are enabled.
All v6.3, v6.2, v6.1, v6.0, and v5.9 fields fully preserved.

CLI Flag (bench/dps_v381_eval.py):

--spectral-failure-risk: enable spectral failure risk scoring
(implies --spectral-bp-alignment).

Evaluation Harness Integration:

Per-trial risk computation after alignment diagnostics.  Aggregates
mean_cluster_risk, max_cluster_risk, and mean_num_high_risk_clusters
across trials per (mode, p, distance) cell.

Scientific Framing

This is a deterministic structural risk heuristic combining spectral
localization (IPR), trapping-set candidate detection, and BP dynamical
alignment.  It does not claim perfect failure prediction.  It identifies
candidate clusters with the highest combined spectral-dynamical risk
signal for downstream structural analysis.

Outputs (top_risk_clusters, node_risk_scores, cluster_risk_scores)
are designed for reuse by future diagnostics (e.g. targeted damping,
modified scheduling, localized perturbation experiments).

Constraints

Decoder untouched: no changes to BP message passing, scheduling,
or convergence logic.

Determinism preserved: no randomness, no global state, all outputs
are deterministic pure functions of config.

Schema unchanged: no schema version bump.

Dependencies unchanged: stdlib + NumPy only.

Additive only: all new fields are opt-in.

---

[6.3.0] — 2026-03-10
Spectral–BP Attractor Alignment

Added

Spectral–BP Attractor Alignment Diagnostics (src/qec/diagnostics/spectral_bp_alignment.py):

compute_spectral_bp_alignment(): measures whether spectral trapping-set
candidates (v6.2) align with actual BP dynamical activity during decoding.
First explicit bridge between structural spectral diagnostics and observed
BP decoding trajectories.

Uses belief oscillation index (BOI) from iteration-trace diagnostics as the
per-node BP activity signal.  Thresholds BP-active nodes via a configurable
fraction of maximum activity score (default 10%).

Output fields: spectral_bp_alignment_score (Jaccard index),
candidate_node_overlap_fraction, bp_node_overlap_fraction, active_bp_nodes,
aligned_candidate_nodes, num_aligned_candidate_nodes,
bp_node_activity_scores, per_cluster_alignment_scores,
top_aligned_clusters, max_cluster_alignment, activity_threshold_fraction.

Phase Diagram Alignment Overlays (src/qec/diagnostics/phase_diagram.py):

Extended phase diagram cell aggregation with optional alignment fields:
mean_spectral_bp_alignment, mean_candidate_node_overlap_fraction,
mean_candidate_cluster_overlap_fraction.
Fields are additive — appear only when alignment diagnostics are enabled.
All v6.2, v6.1, v6.0, and v5.9 fields fully preserved.

CLI Flag (bench/dps_v381_eval.py):

--spectral-bp-alignment: enable spectral–BP attractor alignment diagnostics
(implies --nb-trapping-candidates --iteration-diagnostics).

Evaluation Harness Integration:

Per-trial alignment computed from BOI vectors and per-code trapping
candidates.  Aggregated per (mode, p, distance) with mean/max summary
fields: mean_spectral_bp_alignment, mean_candidate_node_overlap_fraction,
mean_candidate_cluster_overlap_fraction, max_spectral_bp_alignment.

Scientific Framing:

This diagnostic tests whether localized non-backtracking structures and
candidate fragile subgraphs correspond to actual BP activity concentrations.
High alignment supports the hypothesis that spectrally localized modes
identify decoder attractor / failure structures.  This is an alignment
diagnostic — not a proof of causality.

Constraints:

Decoder untouched.  Deterministic.  JSON-serializable.  Schema 3.0.1
unchanged.  No new dependencies.  Additive diagnostics only.

---

[6.2.0] — 2026-03-10
Spectral Trapping-Set Candidate Detection

Added

Spectral Trapping-Set Candidate Detection (src/qec/diagnostics/nb_trapping_candidates.py):

compute_nb_trapping_candidates(): identifies structural trapping-set candidates
by counting node participation across localized non-backtracking eigenmodes.
Nodes appearing in multiple localized modes are flagged as candidates for
fragile Tanner substructures (trapping sets, absorbing sets).

Output fields: node_participation_counts, candidate_variable_nodes,
candidate_check_nodes, candidate_clusters, max_node_participation,
num_candidate_nodes, num_candidate_clusters, participation_threshold.

Cluster detection: connected components among candidate nodes in the Tanner
graph using union-find with deterministic tie-breaking.

Consumes v6.1 localization outputs — does not recompute spectra.

Phase Diagram Trapping-Set Overlays (src/qec/diagnostics/phase_diagram.py):

Extended phase diagram cell aggregation with optional trapping-set fields:
mean_nb_candidate_nodes, mean_nb_max_node_participation, mean_nb_candidate_clusters.
Fields are additive — appear only when trapping-set diagnostics are enabled.
All v6.1, v6.0, and v5.9 fields fully preserved.

CLI Flag (bench/dps_v381_eval.py):

--nb-trapping-candidates: enable spectral trapping-set candidate detection
(implies --nb-localization).

Scientific Framing:

This diagnostic identifies Tanner graph regions that repeatedly participate
in localized spectral modes.  These regions are structural candidates for
trapping sets but the diagnostic does not claim perfect prediction.  Framed
as a structural probe for fragile subgraphs.

Constraints:

Decoder core untouched.
Fully deterministic (no randomness).
JSON-serializable outputs.
Additive diagnostic only.
Schema version unchanged (3.0.1).

---

[6.1.0] — 2026-03-10
Non-Backtracking Localization

Added

Non-Backtracking Localization Diagnostics (src/qec/diagnostics/nb_localization.py):

compute_nb_localization_metrics(): computes inverse participation ratio (IPR)
for leading non-backtracking eigenmodes and projects localized edge support
back onto Tanner graph variable and check nodes.  Quantifies whether leading
spectral modes are diffuse or localized on the graph structure.

Output fields: ipr_scores, max_ipr, localized_modes, mode_support_sizes,
localized_edge_indices, localized_variable_nodes, localized_check_nodes,
top_localization_score, per_mode_mass_on_variables, per_mode_mass_on_checks,
num_directed_edges, num_leading_modes.

IPR metric: IPR(v) = sum_i |v_i|^4 / (sum_i |v_i|^2)^2.
Handles complex-valued eigenvectors via magnitude.  Scale-invariant.

Mode selection: top-k leading modes by eigenvalue magnitude (default k=6).
Localization rule: relative magnitude threshold on squared eigenvector
components (default 0.1 × max).  IPR threshold defaults to 2/num_edges.

Phase Diagram Localization Overlays (src/qec/diagnostics/phase_diagram.py):

Extended phase diagram cell aggregation with optional localization fields:
mean_nb_max_ipr, mean_nb_num_localized_modes, mean_nb_top_localization_score.
Fields are additive — appear only when localization diagnostics are enabled.
All v6.0 and v5.9 fields fully preserved.

CLI Flag (bench/dps_v381_eval.py):

--nb-localization: enable non-backtracking localization diagnostics.

Demo Script (scripts/run_v59_phase_diagram_demo.py):

Updated to include localization diagnostics: nb_max_ipr, nb_num_localized_modes,
nb_top_localization_score attached per trial.

Design Goals

Bridge v6.0 spectral stability diagnostics toward structural fragility prediction.
Identify localized non-backtracking eigenmodes that may correspond to fragile
subgraph structures (trapping sets, absorbing sets).  This is a deterministic
structural diagnostic — not a proven trapping-set detector or full fragility
predictor.

Tests

Test suite: tests/test_nb_localization.py (20 tests).
Covers IPR formula correctness on synthetic vectors (uniform, delta, two-entry,
complex), scale invariance, zero vector, determinism, JSON serialization,
roundtrip stability, input mutation safety, IPR range validation, mass fraction
consistency, Tanner graph projection on toy graphs, edge index validity,
variable/check node coverage.

Phase diagram smoke tests: tests/test_v60_phase_diagram_smoke.py updated
with 2 additional tests for v6.1 localization field presence and population.

Documentation

CHANGELOG.md updated for v6.1.0.

Unchanged

Decoder core logic: untouched.
Baseline decoding outputs remain byte-identical.
All v6.0 spectral diagnostics: unchanged.
Schema version: unchanged.

---

[6.0.0] — 2026-03-10
Spectral Stability Diagnostics

Added

Non-Backtracking Spectrum Diagnostics (src/qec/diagnostics/non_backtracking_spectrum.py):

compute_non_backtracking_spectrum(): computes eigenvalues of the
non-backtracking (Hashimoto) matrix derived from the Tanner graph.
Returns spectral radius, eigenvalue list, and count.  Deterministic
ordering by magnitude with stable tie-breaking.

Bethe Hessian Spectral Analysis (src/qec/diagnostics/bethe_hessian.py):

compute_bethe_hessian(): computes the Bethe Hessian spectrum
H_B = (r²−1)I − rA + D for the Tanner graph adjacency matrix.
Returns eigenvalues (sorted ascending), minimum eigenvalue, count
of negative eigenvalues, and the regularization parameter used.
Automatic r derivation from adjacency spectrum when not specified.

BP Stability Proxy Metrics (src/qec/diagnostics/bp_stability_proxy.py):

estimate_bp_stability(): combines non-backtracking spectral radius
and Bethe Hessian minimum eigenvalue into a single deterministic
stability score: bp_stability_score = (1/spectral_radius) * min_eigenvalue.
Positive score suggests stable BP regime; negative suggests instability.

BP Jacobian Spectral Radius Estimator (src/qec/diagnostics/bp_jacobian_estimator.py):

estimate_bp_jacobian_spectral_radius(): estimates the dominant eigenvalue
of the BP Jacobian without explicit construction using power-iteration
ratios on consecutive LLR differences.  Averages over tail iterations
for robustness.

Phase Diagram Spectral Overlays (src/qec/diagnostics/phase_diagram.py):

Extended phase diagram cell aggregation with optional spectral fields:
mean_spectral_radius, mean_bethe_min_eigenvalue, mean_bp_stability_score,
mean_jacobian_spectral_radius_est.  Fields are additive — appear only
when spectral diagnostics are enabled.  v5.9 fields fully preserved.

ASCII Phase Heatmap (src/qec/diagnostics/phase_heatmap.py):

print_phase_heatmap(): prints a compact ASCII heatmap of the decoder
phase diagram with +1/0/-1 symbols for CLI inspection.

CLI Flags (bench/dps_v381_eval.py):

--nb-spectrum: enable non-backtracking spectrum diagnostics.
--bethe-hessian: enable Bethe Hessian spectral diagnostics.
--bp-stability: enable BP stability proxy (implies --nb-spectrum --bethe-hessian).
--bp-jacobian-estimator: enable BP Jacobian spectral radius estimator.

Demo Script (scripts/run_v59_phase_diagram_demo.py):

Updated to include spectral stability diagnostics and ASCII heatmap output.

Design Goals

Connect empirical decoder phase diagrams to theoretical BP stability conditions.
Provide deterministic spectral predictors of decoding instability.
Preserve observational architecture — diagnostics do not modify decoder behavior.

Tests

Test suites: tests/test_non_backtracking_spectrum.py (8 tests),
tests/test_bethe_hessian.py (9 tests), tests/test_bp_stability_proxy.py (7 tests),
tests/test_bp_jacobian_estimator.py (9 tests), tests/test_v60_phase_diagram_smoke.py (8 tests).
Covers spectral determinism, eigenvalue ordering, JSON serialization, input mutation safety,
formula correctness, backward compatibility, and end-to-end phase diagram integration.

Documentation

docs/reports/v6_bp_stability_spectra.md: motivation, theory, integration,
deterministic design constraints, experimental protocol, limitations.
Includes research hooks for v6.1 NB eigenvector localization and
QLDPC-specific research notes.

Unchanged

Decoder core logic: untouched.
Baseline decoding outputs remain byte-identical.
Schema version: unchanged.
All existing diagnostics: unchanged.
v5.9.0 phase diagram fields: fully preserved (backward compatible).
v5.9.0 tests: all passing without modification.

---

[5.9.0] — 2026-03-10
Decoder Phase Diagram Generator

Added

Decoder Phase Diagram Aggregation (src/qec/diagnostics/phase_diagram.py):

build_decoder_phase_diagram(): deterministic 2D decoder phase-diagram
aggregation over parameter grids.  Sweeps grid points, runs decoding
experiments, and aggregates ternary topology classifications with
continuous diagnostics into phase-diagram-ready JSON artifacts.

make_phase_grid(): helper to construct deterministic 2D parameter grid
specifications.

Phase fractions (success/boundary/failure), dominant phase with
deterministic tie-breaking, and Shannon phase entropy per grid cell.
Continuous observables (boundary_eps, barrier_eps, metastability,
oscillation, alignment, cluster_count) aggregated as cell means.

Phase Boundary Analysis (src/qec/diagnostics/phase_boundary_analysis.py):

analyze_phase_boundaries(): identifies boundary cells (adjacent-phase
mismatch), mixed-region cells (high phase entropy), and critical cells
(high boundary fraction, high metastability, near-zero boundary eps).
All thresholds are explicit and deterministic.

CLI Support (bench/dps_v381_eval.py):

--phase-diagram: enable decoder phase diagram generation
(implies --ternary-topology --ternary-transition-metrics).
--phase-grid-x / --phase-grid-y: axis parameter names.
--phase-grid-x-values / --phase-grid-y-values: axis values.
--phase-diagram-output: JSON output path.

Demo Script (scripts/run_v59_phase_diagram_demo.py):

Standalone deterministic phase-diagram demo that runs a small 2D sweep,
prints dominant-phase tables and summary counts, verifies determinism,
and writes JSON output.

Design Goals

Move beyond threshold curves to dynamical regime maps.
Use ternary topology classification as the primary phase label.
Pair categorical phase labels with continuous diagnostics.
Prepare the framework for future spectral stability upgrades.

Tests

Test suites: tests/test_phase_diagram.py, tests/test_phase_boundary_analysis.py,
tests/test_v59_phase_diagram_demo.py.
Covers fraction aggregation, dominant phase, phase entropy, tie-breaking,
determinism, JSON roundtrip stability, adjacency-based boundary detection,
mixed-region detection, critical-cell detection, and demo script validation.

Unchanged

Decoder core logic: untouched.
Baseline decoding outputs remain byte-identical.
Schema version: unchanged.
All existing diagnostics: unchanged.
v5.8.0 output fields: fully preserved (backward compatible).

---

[5.8.0] — 2026-03-10
Ternary Basin Analysis & Transition Metrics

Added

Ternary Transition Detection (src/qec/diagnostics/ternary_decoder_topology.py):

Extended compute_ternary_decoder_topology() with transition metrics:
  boundary_crossings: number of transitions involving state 0
  regime_switch_count: number of state changes between +1/0/-1
  first_success_iteration: first index where state == +1
  first_failure_iteration: first index where state == -1

Metastability Scoring (src/qec/diagnostics/bp_phase_space.py):

compute_metastability_score(): mean absolute difference of residual
norms over last N iterations, normalized by mean residual.
Low score → convergence, medium → plateau, high → oscillation.

Local Basin Probe (src/qec/diagnostics/basin_probe.py):

probe_local_ternary_basin(): deterministic local basin probing via
fixed-direction LLR perturbations. Returns per-direction ternary
classifications and success/failure/boundary fractions.
No random perturbations. All directions are deterministic.

Harness integration:

--ternary-transition-metrics CLI flag: adds transition metrics and
metastability score to ternary topology output.
--ternary-basin-probe CLI flag: performs deterministic local basin
probe around final LLR state.

Improved

Ternary topology classifier robustness via transition detection.
Trajectory diagnostics with boundary-crossing and regime-switch metrics.

Tests

Test suites: tests/test_ternary_transitions.py, tests/test_basin_probe.py.
Covers transition detection, metastability scoring, basin probe
determinism, classification reproducibility, JSON serialization,
input mutation safety, and backward compatibility.

Unchanged

Decoder core logic: untouched.
Baseline decoding outputs remain byte-identical.
Schema version: unchanged.
All existing diagnostics: unchanged.
v5.7.0 output fields: fully preserved (backward compatible).

---

[5.7.0] — 2026-03-10
BP Phase-Space Explorer & Ternary Topology Classification

Added

BP Phase-Space Explorer (src/qec/diagnostics/bp_phase_space.py):

bp_phase_space.py: deterministic phase-space diagnostic module
implementing compute_bp_phase_space().

Treats BP decoding as a trajectory through an observable phase space.
Records per-iteration observable decoder states and projects them
into a reduced coordinate system for analysis.

Output metrics:

trajectory_length
state_dimension
residual_norms
phase_coordinates
final_phase_coordinate
oscillation_score

Ternary Topology Classifier (src/qec/diagnostics/ternary_decoder_topology.py):

ternary_decoder_topology.py: deterministic ternary topology diagnostic
module implementing compute_ternary_decoder_topology().

Classifies decoding trajectories into:
  +1 stable success basin
   0 boundary / metastable region
  -1 failure basin

Outputs per-iteration ternary trace and final classification with
evidence values and human-readable classification reason.

Integrates with existing v5 diagnostics (v5.1 barrier, v5.3 boundary,
v5.5 alignment) when available.

Harness integration:

--bp-phase-space CLI flag added to bench/dps_v381_eval.py.
--ternary-topology CLI flag added (automatically enables --bp-phase-space).

Per-trial logging includes:

phase-space trajectory metrics
ternary trace and final state
ternary topology summary (success/boundary/failure basin fractions)

Design Goals

Treat BP decoding as a deterministic dynamical system.
Map decoder behavior in observable phase space.
Identify stable basins, boundaries, and failure basins without
modifying decoder internals.

Tests

Test suite: tests/test_bp_phase_space.py, tests/test_ternary_decoder_topology.py
Total tests: 61, all passing.

Unchanged

Decoder core logic: untouched.
Baseline decoding outputs remain byte-identical.
Schema version: unchanged.
All existing diagnostics: unchanged.

---

[5.6.0] — 2026-03-10
Spectral Trapping-Set Diagnostics

Added

Spectral Trapping-Set Diagnostic (src/qec/diagnostics/):

spectral_trapping_sets.py: deterministic spectral trapping-set
diagnostic module implementing compute_spectral_trapping_sets().

Identifies localized spectral clusters in Tanner graph eigenvectors
that may correspond to potential trapping sets or pseudocodeword-prone
regions.  Localized eigenvectors that concentrate mass on small subsets
of variable nodes indicate structural weaknesses in the Tanner graph.

Fully deterministic observational diagnostic that integrates with
the existing v5 diagnostic stack.

Output metrics:

cluster_count
largest_cluster_size
mean_cluster_size
clusters (per-mode: mode_index, cluster_size, nodes, max_importance, mean_importance)

Harness integration:

--spectral-trapping-sets CLI flag added to bench/dps_v381_eval.py.
Automatically enables --tanner-spectral-analysis as a dependency.

Per-code-instance logging includes:

cluster_count
largest_cluster_size
mean_cluster_size

Design Goals

Identify localized spectral structures that may correspond to trapping sets.
Enable correlation studies between spectral localization and BP decoding failures.
Provide a purely graph-theoretic diagnostic for structural vulnerability analysis.

---

[5.5.0] — 2026-03-10
Spectral–Boundary Alignment Diagnostics

Added

Spectral–Boundary Alignment Diagnostic (src/qec/diagnostics/):

spectral_boundary_alignment.py: deterministic spectral-boundary
alignment module implementing compute_spectral_boundary_alignment().

Measures cosine similarity between Tanner spectral eigenvectors
(from v5.4 spectral analysis) and BP decision boundary directions
(from v5.3 boundary analysis).

Fully deterministic observational diagnostic that integrates with
the existing v5 diagnostic stack.

Output metrics:

alignment_scores
max_alignment
mean_alignment
dominant_alignment_mode
mode_count

Harness integration:

--spectral-boundary-alignment CLI flag added to bench/dps_v381_eval.py.
Automatically enables --tanner-spectral-analysis, --bp-boundary-analysis,
and --bp-barrier-analysis as dependencies.

Per-run experiment logging includes:

alignment_max
alignment_mean
dominant_alignment_mode
mode_count
p (error rate)
FER (frame error rate)
boundary_eps
barrier_eps

Design Goals

Investigate whether spectral localization predicts decoder fragility.
Enable correlation studies between spectral modes and FER scaling.
Connect graph-theoretic structure to dynamical decoding vulnerability.

---

[5.4.0] — 2026-03-08
Tanner Spectral Fragility Diagnostics

Introduces deterministic spectral diagnostics for Tanner graphs,
enabling structural analysis of QLDPC parity-check matrices through
global spectral metrics and localized eigenmode analysis.

Unlike previous diagnostics in the v5.x series, this analysis operates
purely on the graph structure and does not run belief propagation
decoding.

The diagnostic measures both global connectivity properties and
localized spectral modes of the Tanner graph, providing structural
signals that may correlate with decoding fragility and pseudocodeword
behavior.

Added

Tanner spectral analysis diagnostic (src/qec/diagnostics/):

tanner_spectral_analysis.py: deterministic Tanner graph spectral
analysis module implementing compute_tanner_spectral_analysis().

Constructs the bipartite Tanner adjacency matrix from the parity-check
matrix and computes global spectral metrics including:

largest_eigenvalue
adjacency_spectral_gap
laplacian_second_eigenvalue
spectral_ratio

Eigenmode localization diagnostics:

Computes Inverse Participation Ratio (IPR) for the leading adjacency
eigenmodes to measure spectral mode localization across the graph.

Localization is reported for both the full graph spectrum and the
variable-node component:

mode_iprs
variable_mode_iprs

Localized spectral node mapping:

Identifies the variable nodes contributing most strongly to the most
localized spectral mode.

Outputs include:

localized_variable_nodes
localized_variable_weights
localized_variable_fraction

The localized_variable_fraction metric measures the proportion of
spectral mass contained in the top localized nodes, providing a simple
indicator of structural fragility concentration.

Benchmark harness integration (bench/dps_v381_eval.py):

--tanner-spectral-analysis: runs Tanner spectral diagnostics once per
code instance and appends results to benchmark artifacts under
tanner_spectral_analysis.

Improved interpretability:

Normalized variable eigenvector components prior to weight extraction
to ensure consistent localization metrics across graph instances.

Added localized_variable_fraction metric quantifying spectral mass
concentration within the top localized variable nodes.

Tests

New test suite (tests/test_tanner_spectral_analysis.py):

deterministic outputs across runs
correct node and edge counts
eigenvalue ordering validation
valid IPR ranges
localized node ordering
JSON serialization compatibility

Total tests: 33, all passing.

Unchanged

Decoder core logic (src/qec_qldpc_codes.py): untouched.
Construction layer (src/qec/construction/): untouched.
Belief propagation decoding algorithms: unchanged.
Benchmark harness behavior unchanged when diagnostics are disabled.

All diagnostics remain fully optional and observational.

Baseline decoding outputs remain byte-identical when diagnostics are disabled.

[5.3.0] — 2026-03-07
Deterministic BP Decision Boundary Analysis

Introduces deterministic boundary analysis for belief propagation (BP)
decoding, enabling estimation of the distance in LLR space to the
nearest competing BP attractor basin.

This diagnostic complements the v5.1 barrier analysis, which measures
the difficulty of escaping the current attractor basin. Boundary
analysis instead measures how close the current decoding state lies to
a competing attractor.

Together these diagnostics enable experimental study of BP attractor
geometry, including basin stability, escape barriers, and decision
boundary proximity.

Added

BP boundary analysis diagnostic (src/qec/diagnostics/):

bp_boundary_analysis.py: deterministic boundary probing module
implementing compute_bp_boundary_analysis().

The diagnostic probes structured perturbation directions derived from:

parity-check constraints  
least-reliable bit structure  
global belief alignment

and performs deterministic binary search along each direction to
determine the minimal perturbation magnitude required to change the
decoder outcome.

Returned metrics include:

baseline_attractor  
boundary_eps  
boundary_direction  
boundary_crossed  
num_directions

where boundary_eps represents the smallest perturbation magnitude that
changes the decoder attractor.

Deterministic perturbation directions:

Three structured direction families are evaluated in deterministic order:

parity-check hyperplane directions  
least-reliable bit directions  
global sign direction

Binary search boundary detection:

For each direction d, the diagnostic searches along:

λ' = λ + ε·d

to determine the minimal ε producing a different decoder outcome.

The smallest ε across all directions defines the estimated boundary
distance.

Benchmark harness integration (bench/dps_v381_eval.py):

--bp-boundary-analysis: runs boundary analysis per trial and records
results under bp_boundary_analysis.

Aggregate statistics are recorded under bp_boundary_summary including:

mean_boundary_eps  
boundary_cross_probability  
num_trials

Improved robustness:

LLR inputs are copied prior to perturbation to prevent accidental
in-place mutation.

Attractor comparison uses tolerance-based equality to avoid spurious
boundary crossings due to floating-point noise.

Tests

New test suite (tests/test_bp_boundary_analysis.py):

deterministic outputs across runs  
successful boundary detection  
no-crossing scenarios  
edge cases (zero LLR, empty parity matrix)  
JSON serialization compatibility

Total tests: 17, all passing.

Unchanged

Decoder core logic (src/qec_qldpc_codes.py): untouched.
Construction layer (src/qec/construction/): untouched.
Belief propagation decoding algorithms: unchanged.
Benchmark harness behavior unchanged when diagnostics are disabled.

All diagnostics remain fully optional and observational.

Baseline decoding outputs remain byte-identical when diagnostics are disabled.

[5.2.0] — 2026-03-06
Decoder Experiment Framework & Deterministic Paired Experiments

Introduces the Decoder Experiment Framework, enabling controlled A/B
experiments between a frozen reference BP decoder and an experimental
sandbox decoder.

This infrastructure allows decoder modifications to be tested while
preserving a permanent deterministic baseline implementation. All
baseline experiments remain byte-identical.

Added

Decoder experiment framework (src/qec/decoder/):

bp_decoder_reference.py: frozen reference decoder re-exporting the
existing bp_decode implementation. Serves as the immutable baseline
decoder for reproducible experiments.

bp_decoder_experimental.py: experimental decoder sandbox. Initially
identical to the reference decoder and intended for future algorithm
experimentation.

decoder_interface.py: decoder registry and selection interface
(DECODER_REGISTRY, get_decoder()) enabling dynamic decoder selection
without modifying the decoding API.

Harness decoder selection (bench/dps_v381_eval.py):

--decoder {reference, experimental}: selects which decoder implementation
to run. Default is reference, preserving existing behavior.

--compare-decoders: executes both decoders on identical inputs during
each trial, enabling deterministic A/B comparison experiments.

Deterministic paired experiment controls:

--paired-seed: guarantees both decoders share the same deterministic
seed sequence during the FER sweep.

--paired-errors: copies the LLR vector before passing to each decoder,
ensuring both decoders observe identical error realizations and
preventing in-place mutation from affecting paired trials.

Decoder comparison reporting:

--decoder-report: prints a console FER comparison table summarizing
decoder performance differences (FER_ref, FER_exp, ΔFER) during
comparison experiments.

Unchanged

Decoder core logic (src/qec_qldpc_codes.py): untouched.
Construction layer (src/qec/construction/): untouched.
Diagnostics modules (src/qec/diagnostics/): unchanged.
Experiment schema and output formats: unchanged.
Baseline decoding behavior remains byte-identical when new flags are disabled.

All existing tests pass without modification.

[5.1.0] — 2026-03-06
Free-Energy Barrier Estimation

Introduces deterministic barrier estimation around BP attractors.  Measures
the minimum perturbation required to escape the current attractor basin,
estimating the free-energy barrier height around BP fixed points.
Diagnostics only.  Decoder behavior unchanged.

Added

- `compute_bp_barrier_analysis()`: deterministic barrier estimation around BP
  attractors via ordered perturbation probing.  Returns baseline attractor
  classification, barrier epsilon (smallest perturbation causing escape),
  escape status, and trial count.
- Deterministic perturbation patterns `[+1, -1, +2, -2]` with configurable
  epsilon schedule (default `[1e-4, 5e-4, 1e-3, 2e-3, 5e-3]`).
- Early termination: stops at the first perturbation that changes attractor
  classification, returning the barrier epsilon.
- Integration with attractor landscape diagnostics (`--bp-landscape-map`
  dependency).
- DPS harness (`bench/dps_v381_eval.py`): new `--bp-barrier-analysis` flag.
  When enabled, computes barrier analysis per trial and stores results under
  `bp_barrier_analysis` with aggregate `bp_barrier_summary` including
  `mean_barrier_eps`, `escape_probability`, and `num_trials`.
- Comprehensive tests (`tests/test_bp_barrier_analysis.py`): correct attractor
  barrier detection, incorrect attractor barrier detection, no escape case,
  determinism across runs, JSON serializability, custom eps schedule.

Unchanged

- Decoder core (`src/qec/decoder/`): untouched.
- Construction layer (`src/qec/construction/`): untouched.
- Existing diagnostics modules: untouched.
- Schema version: unchanged.
- All existing tests pass without modification.

---

[5.0.0] — 2026-03-06
BP Attractor Landscape Mapping

Introduces deterministic mapping of BP attractor landscapes and automatic
pseudocodeword detection.  Diagnostics only.  Decoder behavior unchanged.

Added

- `compute_bp_landscape_map()`: maps the decoder attractor landscape by
  sampling deterministic perturbations of the initial belief state.  Measures
  attractor enumeration, basin size distribution, largest basin fraction,
  correct/incorrect/degenerate attractor fractions, and pseudocodeword
  detection.
- Attractor identification via CRC32 of the final LLR sign pattern, grouping
  runs that converge to the same fixed point.
- Automatic pseudocodeword detection: incorrect fixed-point attractors that
  remain stable under small perturbations are flagged as pseudocodewords.
- DPS harness (`bench/dps_v381_eval.py`): new `--bp-landscape-map` flag.
  When enabled, computes landscape mapping per trial and stores results under
  `bp_landscape_map` with aggregate `bp_landscape_summary` including
  `mean_num_attractors`, `mean_largest_basin_fraction`, and
  `total_pseudocodewords`.
- Comprehensive tests (`tests/test_bp_landscape_mapping.py`): single attractor,
  multiple attractors, pseudocodeword detection, determinism, JSON
  serializability, attractor ID computation.

Unchanged

- Decoder core (`src/qec/decoder/`): untouched.
- Construction layer (`src/qec/construction/`): untouched.
- Existing diagnostics modules: untouched.
- Schema version: unchanged.
- All existing tests pass without modification.
- Baseline decoding output: byte-identical when diagnostics disabled.

---

[4.9.0] — 2026-03-06
Basin-of-Attraction and Boundary Analysis

Introduces deterministic perturbation experiments to estimate the basin
geometry of BP fixed points.  Diagnostics only.  Decoder behavior unchanged.

Added

- `compute_bp_basin_analysis()`: estimates basin-of-attraction geometry via
  deterministic perturbation of the initial LLR vector.  Measures correct
  attractor probability, incorrect attractor probability, degenerate attractor
  probability, and basin boundary distance.
- Basin boundary estimation: finds minimum perturbation magnitude that changes
  the fixed-point classification, approximating the distance to the nearest
  pseudocodeword basin boundary.
- DPS harness (`bench/dps_v381_eval.py`): new `--bp-basin-analysis` flag.
  When enabled, computes basin analysis per trial and stores results under
  `bp_basin_analysis` with aggregate `bp_basin_summary` including
  `mean_basin_correct_probability`, `mean_basin_incorrect_probability`,
  `mean_basin_degenerate_probability`, `mean_boundary_eps`, and
  `min_boundary_eps`.
- Comprehensive tests (`tests/test_bp_basin_analysis.py`): correct basin,
  mixed basin, boundary detection, determinism, JSON serializability,
  custom epsilon values.

Unchanged

- Decoder core (`src/qec/decoder/`): untouched.
- Construction layer (`src/qec/construction/`): untouched.
- Existing diagnostics modules: untouched.
- Schema version: unchanged.
- All existing tests pass without modification.
- Baseline decoding output: byte-identical when diagnostics disabled.

---

[4.8.0] — 2026-03-06
Deterministic Fixed-Point Trap Analysis

Introduces diagnostics for classifying BP fixed-point outcomes.
Diagnostics only.  Decoder behavior unchanged.

Added

- `compute_bp_fixed_point_analysis()`: classifies BP decoding outcomes as
  `correct_fixed_point`, `incorrect_fixed_point`, `degenerate_fixed_point`,
  or `no_convergence` using deterministic analysis of energy traces, LLR
  entropy, LLR variance, and final syndrome weight.
- Degenerate fixed-point detection via LLR entropy and variance thresholds
  to identify symmetric attractors.
- Energy convergence detection via tail-window stability analysis.
- DPS harness (`bench/dps_v381_eval.py`): new `--bp-fixed-point-analysis`
  flag.  When enabled, computes fixed-point classification per trial and
  stores results under `bp_fixed_point_analysis` with aggregate
  `bp_fixed_point_summary` including `correct_fixed_point_probability`,
  `incorrect_fixed_point_probability`, `degenerate_fixed_point_probability`,
  and `mean_iterations_to_fixed_point`.
- Comprehensive tests (`tests/test_bp_fixed_point_analysis.py`): determinism,
  correct/incorrect/degenerate/no-convergence classification, edge cases,
  JSON serializability, custom parameters.

Unchanged

- Decoder core (`src/qec/decoder/`): untouched.
- Construction layer (`src/qec/construction/`): untouched.
- Existing diagnostics modules: untouched.
- Schema version: unchanged.
- All existing tests pass without modification.
- Baseline decoding output: byte-identical when diagnostics disabled.

---

[4.7.0] — 2026-03-05
Deterministic BP Freeze Detection

Adds deterministic early metastability / freeze detection for BP decoding
dynamics.  Diagnostics only.  Decoder behavior unchanged.

Added

- `compute_bp_freeze_detection()`: detects early metastability in BP
  decoding dynamics by computing a composite freeze score from MSI, EDS,
  GOS, and CPI metrics over a sliding window.
- Freeze is declared when `freeze_score > threshold` AND the regime is
  `metastable_state`.  Returns `freeze_detected`, `freeze_iteration`,
  `freeze_score`, and `freeze_regime`.
- DPS harness (`bench/dps_v381_eval.py`): new `--bp-freeze-detection` flag.
  When enabled, computes freeze detection per trial and stores results
  under `bp_freeze_detection` with aggregate `bp_freeze_summary`.
- Comprehensive tests (`tests/test_bp_freeze_detection.py`): determinism,
  stable/metastable traces, edge cases, JSON serializability.

Unchanged

- Decoder core (`src/qec/decoder/`): untouched.
- Construction layer (`src/qec/construction/`): untouched.
- Existing diagnostics modules: untouched.
- Schema version: unchanged.
- All existing tests pass without modification.
- Baseline decoding output: byte-identical when diagnostics disabled.

---

[4.6.0] — 2026-03-05
Deterministic BP Phase Diagram Analysis

Adds deterministic aggregation of BP regime traces across parameter sweeps
to compute phase diagram statistics for metastability analysis.
Diagnostics only.  Decoder behavior unchanged.

Added

- `compute_bp_phase_diagram()`: aggregates regime-trace diagnostics across
  decoding runs to produce deterministic BP phase statistics as a function
  of code distance and noise rate.
- Phase point output: per-(distance, noise) statistics including
  `metastable_probability`, `mean_freeze_score`, `mean_switch_rate`,
  `mean_max_dwell`, `event_rate`, and `regime_frequencies`.
- Configurable metastable threshold (default 0.5).
- DPS harness (`bench/dps_v381_eval.py`): new `--bp-phase-diagram` flag.
  When enabled, computes phase diagram analysis across all modes and stores
  results under `bp_phase_diagram`.
- Comprehensive tests (`tests/test_bp_phase_diagram.py`): determinism,
  correct aggregation, empty input handling, bench integration smoke tests.

Improved

- Clarified CLI help text for `--bp-phase-diagram` (implicitly enables
  `--bp-transitions`; users need not specify both).
- Added input validation for malformed phase-diagram inputs
  (`TypeError` / `ValueError` on missing keys).
- Improved type hints: `TypedDict` structures (`RegimeTraceSummary`,
  `RegimeTraceResult`, `RunResult`, `BpPhaseDiagram`).
- Added edge-case test for empty regime traces.

Unchanged

- Decoder core (`src/qec/decoder/`): untouched.
- Construction layer (`src/qec/construction/`): untouched.
- Schema version: unchanged.
- Baseline decoding output: byte-identical when diagnostics disabled.

---

[4.5.0] — 2026-03-05
Deterministic BP Regime Transition Analysis

Adds deterministic per-iteration regime trace analysis that tracks how
BP regimes evolve over time.  Diagnostics only.  Decoder behavior
unchanged.

Added

- `compute_bp_regime_trace()`: constructs a per-iteration regime trace
  using sliding-window classification, detects regime transitions,
  measures dwell times, identifies instanton-like transition events,
  and produces transition statistics.
- Regime trace output: per-iteration regime label sequence.
- Transition detection: records iteration index, source/target regime,
  and instanton-like event flag for each regime change.
- Dwell-time metrics: contiguous run lengths per regime.
- Transition count matrix: deterministic `from->to` counts, sorted
  lexicographically.
- Summary statistics: `switch_rate`, `max_dwell`, `freeze_score`,
  `num_events`.
- Instanton-like event detection: flags transitions where energy change
  exceeds `event_factor * median(|ΔE|)`.
- DPS harness (`bench/dps_v381_eval.py`): new `--bp-transitions` flag.
  When enabled, computes regime transition analysis per trial and stores
  results under `bp_regime_trace`, `bp_transition_summary`, and
  `bp_transition_counts`.
- Comprehensive tests (`tests/test_bp_regime_trace.py`): determinism,
  stable regime, oscillatory regime, metastable plateau, chaotic regime,
  dwell-time consistency, edge cases, bench integration smoke tests.

Improved

- Added strict validation ensuring `llr_trace` and `energy_trace` lengths
  match — raises `ValueError` on mismatch instead of silent misalignment.
- Prevented regime-trace parameter (`event_factor`) leakage into
  `bp_dynamics` metrics API.
- Strengthened event-detection tests to assert actual event occurrence.
- Added mismatched-trace edge-case test.

Unchanged

- Decoder core (`src/qec/decoder/`) untouched.
- Construction (`src/qec/construction/`) untouched.
- Schema version unchanged.
- Baseline decoding outputs byte-identical when diagnostics disabled.
- All prior diagnostics (v4.1–v4.4) unchanged and fully backward
  compatible.

---

[4.4.0] — 2026-03-05
Deterministic BP Dynamics Regime Analysis

Adds a deterministic BP dynamics metric suite and regime classifier that
analyzes iteration traces to classify decoder dynamics into one of six
regimes.  Diagnostics-only, decoder-safe: does not modify BP decoder
internals.

Added

- `compute_bp_dynamics_metrics()`: computes eight deterministic metrics
  from BP traces — Metastability Index (MSI), Cycle Periodicity Index
  (CPI), Trapping Set Likelihood (TSL), Local Energy Curvature (LEC),
  Correction-Vector Norm Entropy (CVNE), Global Oscillation Score (GOS),
  Energy Descent Smoothness (EDS), Basin Transition Indicator (BTI).
- `classify_bp_regime()`: deterministic first-match rule ladder classifier
  producing one of six regimes: `stable_convergence`,
  `oscillatory_convergence`, `metastable_state`, `trapping_set_regime`,
  `correction_cycling`, `chaotic_behavior`.
- CVNE metrics return `None` when correction vectors are unavailable
  (consistent with v4.3.0 correction-vector semantics).
- DPS harness (`bench/dps_v381_eval.py`): new `--bp-dynamics` flag.
  When enabled, computes BP dynamics metrics per trial and stores
  them under `result["bp_dynamics"]` with aggregate `bp_regime_counts`.
- Comprehensive tests (`tests/test_bp_dynamics.py`): determinism,
  zero-sign handling, optional correction vectors, regime branch
  coverage (all six regimes), trace normalization, edge cases, bench
  integration smoke tests.

Unchanged

- Decoder core (`src/qec/decoder/`) untouched.
- Construction (`src/qec/construction/`) untouched.
- Schema version unchanged.
- Baseline decoding outputs byte-identical when diagnostics disabled.
- v4.3.0 iteration-trace semantics preserved (BOI sign logic, CVF None
  pattern).

---

[4.3.0] — 2026-03-05
Deterministic Iteration-Trace Diagnostics

Adds iteration-trace diagnostics that analyse BP iteration logs to detect
trapping sets, oscillatory message passing, unstable convergence, and
correction vector cycling.  Diagnostics operate purely on iteration traces
and do not modify the BP decoder.

Added

- `compute_persistent_error_indicator()` (PEI): flags variable nodes whose
  LLR sign indicates an error for a configurable number of consecutive
  iterations.  Detects trapping sets.
- `compute_belief_oscillation_index()` (BOI): counts LLR sign flips across
  iterations for each variable node.  Measures oscillatory message passing.
- `compute_oscillation_depth()` (OD): measures peak-to-peak LLR amplitude
  over a trailing window.  Quantifies oscillation severity.
- `compute_convergence_instability_score()` (CIS): variance of the energy
  trace over a trailing window.  Detects unstable convergence.
- `compute_correction_vector_fluctuation()` (CVF): Euclidean norm of
  consecutive correction vector differences.  Detects correction cycling.
- `compute_iteration_trace_metrics()`: composite function returning all
  five metrics in a single call.
- DPS harness (`bench/dps_v381_eval.py`): new `--iteration-diagnostics`
  flag.  When enabled, computes iteration-trace metrics per trial and
  stores them under `result["iteration_diagnostics"]`.
- Comprehensive tests (`tests/test_iteration_trace_metrics.py`):
  determinism, no-input-mutation, oscillation detection, stable
  convergence, trapping set detection, correction cycling, and composite
  metric validation.

Unchanged

- Decoder core: no modifications to BP loops, scheduling, or iteration.
- Schema: no SCHEMA_VERSION bump.
- Canonical serialization, hashing, and identity: unchanged.
- Baseline decoding outputs: byte-identical under identical inputs.
- All existing harness output fields remain present and unchanged.

---

[4.2.1] — 2026-03-05
Diagnostics Refactor and Test Hardening

Patch release improving the diagnostics implementation introduced in
v4.2.0.  No decoder core changes.  Deterministic outputs preserved.

Changed

- Eliminates redundant BP decodes by sharing perturbation results
  via new internal helper `_run_perturbation_decodes()`.
  `classify_basin_switch()` and `compute_landscape_metrics()` now
  reuse a single set of ±epsilon decode results.
- Allows configurable epsilon sweep for escape-energy diagnostics:
  `compute_escape_energy()` accepts optional `eps_values` parameter.
  Default behavior remains identical to v4.2.0.
- Strengthens landscape integration test to guarantee metric
  validation: conditional guards replaced with explicit assertions
  ensuring `basin_classifications` and all landscape metric fields
  are always verified.

Unchanged

- Decoder core: no modifications to BP loops, scheduling, or iteration.
- Schema: no SCHEMA_VERSION bump.
- Canonical serialization, hashing, and identity: unchanged.
- Baseline decoding outputs: byte-identical under identical inputs.
- Classification logic and metric outputs: identical to v4.2.0.

---

[4.2.0] — 2026-03-05
Deterministic Landscape Metrics

Adds Basin Stability Index (BSI), Attractor Distance (AD), and Escape
Energy (EE) to energy landscape diagnostics.  Escape energy includes
directional barrier estimation inspired by spin-glass energy landscape
analysis.

This is a diagnostic-only extension.  No decoder core modifications.

Added

- `compute_basin_stability_index()`: ratio of perturbations yielding
  same correction as baseline.  Values in [0.0, 1.0].
- `compute_attractor_distance()`: Hamming distance between baseline and
  perturbed corrections.  Returns max and mean.
- `compute_escape_energy()`: deterministic epsilon sweep to find minimum
  perturbation causing a basin switch.  Probes +epsilon and -epsilon
  independently, returning directional and minimum barriers.
- `compute_landscape_metrics()`: composite function combining v4.1.0
  classification with BSI, AD, and EE in a single output dict.
- `_hamming_distance()`: deterministic Hamming distance helper.
- DPS harness (`bench/dps_v381_eval.py`) now emits landscape metrics
  (BSI, AD, EE) alongside basin classifications when `--landscape`
  mode is enabled.
- Comprehensive tests: determinism, baseline safety, Hamming distance
  correctness, escape energy sweep, and harness integration.

Unchanged

- Decoder core: no modifications to BP loops, scheduling, or iteration.
- Schema: no SCHEMA_VERSION bump.
- Canonical serialization, hashing, and identity: unchanged.
- Baseline decoding outputs: byte-identical under identical inputs.
- All existing harness output fields remain present and unchanged.
- `classify_basin_switch()` remains available and unmodified.

---

[4.1.0] — 2026-03-05
Improved Basin Switch Detection

Strengthens deterministic perturbation diagnostics by introducing a
three-regime classifier that distinguishes between metastable oscillation,
shallow perturbation sensitivity, and true basin switching.

This is a diagnostic-only improvement.  No decoder core modifications.

Added

- `classify_basin_switch()` in `src/qec/diagnostics/energy_landscape.py`:
  performs three deterministic decodes (baseline, +epsilon, -epsilon)
  and classifies the result as `metastable_oscillation`,
  `shallow_sensitivity`, `true_basin_switch`, or `none`.
- Helper functions `_count_gradient_sign_flips()` and
  `_trace_converged()` for trace analysis.
- DPS harness (`bench/dps_v381_eval.py`) now emits `basin_classifications`
  and `basin_class_counts` when `--landscape` mode is enabled.
- Comprehensive tests: determinism, baseline safety, classification
  coverage, and harness integration.

Unchanged

- Decoder core: no modifications to BP loops, scheduling, or iteration.
- Schema: no SCHEMA_VERSION bump.
- Canonical serialization, hashing, and identity: unchanged.
- Baseline decoding outputs: byte-identical under identical inputs.
- All existing harness output fields remain present and unchanged.

---

[4.0.0] — 2026-03-05
BP Free-Energy Landscape Diagnostics

Introduces a deterministic diagnostics layer for analyzing belief
propagation (BP) energy dynamics during QLDPC decoding.

This release extends the deterministic benchmarking framework with
tools for studying decoder convergence regimes, including plateau
behavior, barrier crossings, and geometry-induced basin switching.

All diagnostics are strictly observational and do not modify decoding
behavior.

Added

Energy Landscape Diagnostics Module

New module:

src/qec/diagnostics/energy_landscape.py

Provides deterministic analysis utilities for BP energy traces:

compute_energy_gradient

compute_energy_curvature

detect_plateau

detect_local_minima

detect_barrier_crossings

classify_energy_landscape

detect_basin_switch

Energy is evaluated per BP iteration:

E = − Σ (LLR_i · belief_i)

These diagnostics enable systematic analysis of BP convergence behavior.

Basin Switching Detector

Introduces a deterministic perturbation experiment to detect
free-energy basin switching in BP decoding.

A small perturbation is applied to the LLR vector:

llr_perturbed = llr + ε · sign(llr)
ε = 1e-3

If the perturbed decode converges to a different correction or final
energy, the trial is classified as a basin switch.

The perturbation is deterministic and safely handles sign(0).

DPS Harness Landscape Mode

The deterministic DPS evaluation harness now supports energy landscape
diagnostics via a CLI flag:

--landscape

When enabled the harness records:

per-iteration BP energy traces

landscape classification statistics

basin switching frequency per mode

Example run:

PYTHONPATH=. python bench/dps_v381_eval.py \
  --landscape \
  --trials 200 \
  --distances 5 7 \
  --p-values 0.03

All modes reuse identical deterministic error instances.

Improvements

Reduced Diagnostic Overhead

The basin-switch detector now reuses the existing decode result
from the harness instead of performing a redundant baseline decode.

This significantly reduces diagnostic runtime when landscape analysis
is enabled.

Fixes

Geometry Postprocessing Consistency

Fixed an issue where geometry postprocessing could be applied
asymmetrically between baseline and perturbed decodes in the basin
switch detector.

Both decodes now operate in the same LLR domain, differing only by the
deterministic perturbation.

Perturbation Stability

Improved numerical stability by ensuring deterministic perturbation
behavior when LLR == 0.

Tests

New test suite:

tests/test_energy_landscape.py

Coverage includes:

gradient and curvature computation

plateau detection

barrier detection

basin switch detection

deterministic perturbation behavior

Results

Full test suite:

945 passed
7 skipped
0 failed

Deterministic reproducibility verified.

Guarantees

No changes to core BP decoding logic

No changes to BP scheduling or message updates

Diagnostics operate only on decoder outputs

No dependency changes

Baseline decoder outputs remain byte-identical when diagnostics disabled

All new features are opt-in

Determinism preserved

---

## [3.9.1] — 2026-03-04

### Geometry Field Controls

Introduces deterministic geometry field controls for controlled testing
of likelihood magnitude effects under syndrome-only inference.

All new features are opt-in. Baseline decoder behavior remains unchanged
when features are disabled.

### Added

**Geometry Strength Scaling**

- `geometry_strength` field on `StructuralConfig` (default: `1.0`)
- Scales the constructed geometry LLR field after centered_field and
  pseudo_prior are applied
- Applied in both `BPAdapter` and DPS harness

**Deterministic Field Normalization**

- `normalize_geometry` field on `StructuralConfig` (default: `False`)
- When enabled: `llr = llr / (std(llr) + 1e-12)`
- Ensures the LLR distribution has unit variance
- Only applies when geometry interventions are active
- Normalization is applied before geometry_strength scaling

**DPS Harness Geometry Sweep Modes**

Three new evaluation modes:

- `centered_strong` — centered field + geometry_strength=2.0
- `centered_normalized` — centered field + normalize_geometry
- `centered_prior_normalized` — centered + prior + normalize_geometry

All modes reuse identical deterministic error instances.

### Tests

New test suite:

    tests/test_geometry_controls.py

Coverage includes:

- geometry_strength scaling determinism and correctness
- normalize_geometry unit-variance verification
- baseline invariance when features disabled
- DPS harness new mode execution and determinism

### Results

Full test suite:

    923 passed
    7 skipped
    0 failed

Deterministic reproducibility verified.

### Guarantees

- No changes to core decoding logic
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes
- No dependency changes
- Baseline decoder outputs remain byte-identical when features disabled
- All new features are opt-in with safe defaults
- Determinism preserved

---

[3.9.0] — 2026-03-04
Channel Geometry Interventions & BP Energy Diagnostics

Introduces deterministic channel-geometry interventions and belief propagation energy diagnostics for structural decoding experiments under syndrome-only inference.

This release expands the deterministic experimentation framework introduced in v3.8.x.

Baseline decoder behavior remains unchanged when all structural features are disabled.

Added

Channel geometry utilities:

src/qec/channel/geometry.py

Deterministic functions:

syndrome_field()

centered_syndrome_field()

pseudo_prior_bias()

apply_pseudo_prior()

These construct LLRs directly from syndrome structure for oracle-free decoding experiments.

Belief propagation energy diagnostics:

src/qec/decoder/energy.py

Provides optional per-iteration energy tracing:

E = − Σ (LLR_i · belief_i)

Energy tracing enables analysis of:

BP convergence dynamics

oscillatory decoding behavior

likelihood alignment during inference

Energy tracing is purely diagnostic and does not alter decoder outputs.

DPS Harness Expansion

The deterministic evaluation harness now includes geometry-intervention modes.

New modes:

centered

prior

centered_prior

geom_centered

geom_centered_prior

rpc_centered

rpc_centered_prior

All modes reuse identical deterministic error instances.

Baseline evaluation behavior remains unchanged.

Stability Improvements

Resolved issues identified during code review:

stabilized bp_decode() return structure across optional diagnostics

ensured consistent tuple ordering when energy_trace is enabled

added epsilon threshold to DPS sign detection to avoid floating-point noise

strengthened baseline invariance testing

Tests

New test suites:

tests/test_channel_geometry.py
tests/test_energy_trace.py

Coverage includes:

deterministic geometry field construction

pseudo-prior application

BP energy trace correctness

return-structure validation across feature combinations

baseline decoder invariance

Results

Full test suite:

904 passed
0 failed

Deterministic reproducibility verified.

## [3.8.1] — 2026-03-03

### Structural Geometry Evaluation Harness

Adds a deterministic evaluation harness for analyzing structural
decoder interventions introduced in v3.8.0.

This release introduces **measurement infrastructure only**.
No decoder algorithms were modified.

### Added

Deterministic DPS evaluation harness:


bench/dps_v381_eval.py


Capabilities:

- deterministic RNG (`seed = 42`)
- pre-generated error instances reused across modes
- four evaluation modes:
  - baseline
  - rpc_only
  - geom_v1_only
  - rpc_geom
- activation audit reporting:
  - original_rows
  - augmented_rows
  - added_rows
  - H checksum
  - syndrome checksum
  - iteration count
- deterministic slope estimation for DPS
- inversion detection marker
- determinism verification run

Frame error rate uses **syndrome-consistency semantics**:


syndrome(H, correction) != s


### Tests

New harness validation suite:


tests/test_dps_v381_harness.py


Coverage includes:

- deterministic instance reuse
- RPC activation verification
- schedule dispatch validation
- DPS slope computation
- decoder invariance confirmation

### Guarantees

- No decoder algorithm changes
- `bp_decode()` unchanged
- All BP schedules unchanged
- `_bp_postprocess()` unchanged
- No schema changes
- No dependency changes
- Deterministic outputs preserved
- Full test suite passing

---

## [3.8.0] — 2026-03-02

### Structural Geometry Infrastructure

Introduces deterministic infrastructure for controlled experiments on
decoder **topology** and **inference geometry**.

All new features are **strictly opt-in**.

Baseline decoder behavior remains unchanged when disabled.

---

### Added

#### RPC Builder

New deterministic redundant parity-check augmentation module:


src/qec/decoder/rpc.py


Provides:


build_rpc_augmented_system()


Functionality:

- deterministic lexicographic row-pair XOR generation
- redundant parity constraints
- no feasible-set change
- no mutation of original H matrix
- deterministic ordering of generated rows

Configuration objects:


RPCConfig
StructuralConfig


Tests:


tests/test_rpc_builder.py


---

#### `geom_v1` Schedule

Adds a geometry-scaled flooding schedule.


schedule="geom_v1"


Scaling rule:


α_c = 1 / sqrt(d_c)


Where `d_c` is the degree of check node `c`.

Properties:

- flooding-style schedule
- deterministic scaling
- no adaptive behavior
- no stochastic elements

Tests:


tests/test_geom_v1_schedule.py


---

#### Adapter Integration

Structural geometry features integrated into the decoder adapter layer.

File:


src/bench/adapters/bp.py


Behavior:


if structural_config.rpc.enabled:
H_used, s_used = build_rpc_augmented_system(...)
else:
H_used, s_used = H, s


This ensures structural interventions occur **outside the decoder core**.

Tests:


tests/test_adapter_rpc_integration.py


---

### Guarantees

- Flooding schedule unchanged
- Layered schedule unchanged
- Residual schedule unchanged
- `_bp_postprocess()` unchanged
- Decoder iteration logic untouched
- No schema version changes
- No dependency additions
- No stochastic behavior introduced
- Baseline decoder outputs remain bit-identical

---

## [3.7.0] — 2026-03-01

### Uniformly Reweighted BP (URW-BP)

Adds a new opt-in BP mode `mode="min_sum_urw"` that applies a uniform
scalar reweighting factor `urw_rho` to check-to-variable messages,
reducing loop overcounting in loopy Tanner graphs.

The URW reweighting scales each check-to-variable message by a constant
`rho in (0, 1]`:

    R_j→i ← urw_rho * R_j→i

This is algebraically equivalent to `min_sum` when `urw_rho=1.0`.

### Added

- `mode="min_sum_urw"` in `bp_decode()`:
  - Applies `urw_rho` as a uniform scalar multiplier to check-to-variable
    messages in the min-sum update rule
  - Supported on all schedules: flooding, layered, residual,
    hybrid_residual, adaptive
  - Compatible with damping, clipping, llr_history, residual_metrics,
    and all existing postprocessors (osd0, osd1, osd_cs, etc.)
- `urw_rho` parameter in `bp_decode()`:
  - Validated only when `mode="min_sum_urw"`: must satisfy `0 < urw_rho <= 1`
  - Default value `1.0` (no-op for non-URW modes)
- Comprehensive test suite in `tests/test_urw_bp_v370.py`:
  - Baseline invariance across all existing modes and schedules
  - `rho=1.0` bit-identity with `min_sum`
  - Determinism across repeated runs
  - Validation error tests for invalid `urw_rho`
  - Identity inclusion tests via BPAdapter

### Guarantees

- No changes to default decoder behavior
- No changes to baseline decoder identity/hash
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes (SCHEMA_VERSION and INTEROP_SCHEMA_VERSION unchanged)
- No new dependencies
- No scheduling changes
- No randomness introduced
- Determinism verified across repeated runs
- All existing modes unaffected: sum_product, min_sum, norm_min_sum,
  offset_min_sum, improved_norm, improved_offset
- All existing schedules unaffected
- All existing postprocessors unaffected
- All existing tests pass without modification

---

## [3.6.0] — 2026-03-01

### Deterministic Posterior-Aware Combination-Sweep OSD Postprocess

Adds a new opt-in postprocess mode `postprocess="mp_osd_cs"` that uses
posterior LLR magnitude (`abs(L_post)`) instead of channel LLR to order
columns for OSD-CS information-set selection and combination sweep.

This extends the mp_osd1 approach (v3.5.0) from single-bit flip to
multi-bit combination sweep, providing a higher-order posterior-aware
search without altering BP semantics or default behavior.

### Added

- `postprocess="mp_osd_cs"` in `bp_decode()`:
  - Runs inner BP with `postprocess=None` and `llr_history=1` to obtain
    posterior beliefs
  - If BP converges, returns immediately (no OSD needed)
  - Otherwise, applies OSD-CS with reliability ordering based on
    `abs(L_post)` instead of `abs(channel_llr)`
  - Combination sweep depth controlled by existing `osd_cs_lam` parameter
  - Tie-breaking: ascending variable index (deterministic)
  - Never-degrade guarantee: if OSD result fails syndrome, returns BP
    hard decision
- `mp_osd_cs_postprocess()` function in `src/decoder/osd.py`
- Comprehensive test suite in `tests/test_mp_osd_cs.py`

### Guarantees

- No changes to default decoder behavior
- No changes to baseline decoder identity/hash
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes (SCHEMA_VERSION and INTEROP_SCHEMA_VERSION unchanged)
- No new dependencies
- No new parameters (reuses existing `osd_cs_lam`)
- No randomness introduced
- Determinism verified across repeated runs
- Baseline OSD postprocess modes (osd0, osd1, osd_cs) unaffected
- MP-OSD-1 postprocess unaffected
- Guided decimation postprocess unaffected
- All existing tests pass without modification

---

## [3.5.0] — 2026-03-01

### Deterministic MP-Aware OSD-1 Postprocess

Adds a new opt-in postprocess mode `postprocess="mp_osd1"` that uses
posterior LLR magnitude (`abs(L_post)`) instead of channel LLR to order
columns for OSD-1 information-set selection.

This exploits the message-passing information to produce a more informed
reliability ranking, without altering BP semantics or default behavior.

### Added

- `postprocess="mp_osd1"` in `bp_decode()`:
  - Runs inner BP with `postprocess=None` and `llr_history=1` to obtain
    posterior beliefs
  - If BP converges, returns immediately (no OSD needed)
  - Otherwise, applies OSD-1 with reliability ordering based on
    `abs(L_post)` instead of `abs(channel_llr)`
  - Tie-breaking: ascending variable index (deterministic)
  - Never-degrade guarantee: if OSD result fails syndrome, returns BP
    hard decision
- `mp_osd1_postprocess()` function in `src/decoder/osd.py`
- Comprehensive test suite in `tests/test_mp_osd1.py`

### Guarantees

- No changes to default decoder behavior
- No changes to baseline decoder identity/hash
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes (SCHEMA_VERSION and INTEROP_SCHEMA_VERSION unchanged)
- No new dependencies
- No randomness introduced
- Determinism verified across repeated runs
- Baseline OSD postprocess modes (osd0, osd1, osd_cs) unaffected
- Guided decimation postprocess unaffected
- All existing tests pass without modification

---

## [3.4.0] — 2026-03-01

### Deterministic Belief Propagation Guided Decimation

Adds a new opt-in postprocess mode `postprocess="guided_decimation"` that
performs iterative variable freezing guided by BP posterior beliefs.

This is a minimal structural intervention designed to break degeneracy and
trapping behavior in syndrome-only BP decoding, without altering BP
semantics or default scheduling logic.

### Added

- `postprocess="guided_decimation"` in `bp_decode()`:
  - Runs BP for `decimation_inner_iters` per round (up to `decimation_rounds`)
  - After each round, selects the unfrozen variable with maximal |posterior LLR|
  - Ties broken deterministically by lowest variable index
  - Zero-posterior convention: freeze to +decimation_freeze_llr (hard = 0)
  - Freezes the selected variable by clamping its LLR to
    ±decimation_freeze_llr
  - Returns immediately when syndrome is satisfied
  - Non-convergence fallback ranks candidates by
    (syndrome_weight, hamming_weight, round_index) — fully explicit
- Three new parameters (only validated when `postprocess="guided_decimation"`):
  - `decimation_rounds` (default 10)
  - `decimation_inner_iters` (default 10)
  - `decimation_freeze_llr` (default 1000.0)
- `guided_decimation()` function in `src/decoder/decimation.py`
- Comprehensive test suite in `tests/test_guided_decimation.py`

### Guarantees

- No changes to default decoder behavior
- No changes to baseline decoder identity/hash
- No changes to `_bp_postprocess()` or BP iteration loops
- No schema changes (SCHEMA_VERSION and INTEROP_SCHEMA_VERSION unchanged)
- No new dependencies
- No randomness introduced
- Determinism verified across repeated runs
- Baseline OSD postprocess modes unaffected
- All existing tests pass without modification

### Structural techniques intentionally NOT implemented (out of scope)

- Stabilizer Inactivation
- MP-aware OSD
- Check reweighting
- Sequential CN scheduling variants
- Graph surgery / Tanner graph modification
- Directional LLR bias injection
- Channel model modification
- Learned / neural components
- New schedule families
- Automatic schedule switching

---

## [3.3.1] — 2026-03-01

### v3.3.1 — Geometry Diagnostics Hardening

- SSI grouping hardened to include decoder identity
- DPS regression now fits full-precision log values
- BSI now raises `BSIConfigError` (specific subtype of `ValueError`)
- `compute_bsi` docstring clarifies handling of extra 2x-only records
- README badge corrected

No decoder behavior changes.
No schema changes.
Determinism preserved.

---

## [3.3.0] — 2026-02-28

### Geometry-Aware Syndrome-Only Diagnostics

This release adds a diagnostics-first reporting layer for explaining
distance scaling inversion under `bsc_syndrome` channel inference.

All metrics are computed post-hoc from existing benchmark results.
No decoder behavior changes.  No schema changes.  Canonical benchmark
artifacts are byte-identical when diagnostics are not invoked.

---

### Added

**Geometry Diagnostics Module (`src/bench/geometry_diagnostics.py`)**

- Distance Penalty Slope (DPS): slope of log10(FER + eps) vs distance
  per (decoder, p) group — positive slope indicates inversion
- False-Convergence Rate (FCR): P(syndrome=0 AND logical failure),
  derived algebraically as SCR - Fidelity (equivalent to Inversion
  Index from v3.2.1)
- Budget Sensitivity Index (BSI): FER(base) - FER(2x) for comparing
  iteration budget impact
- Schedule Sensitivity Index (SSI): max(FER) - min(FER) across
  schedules per (distance, p)
- Per-iteration summary computation from existing `llr_history`:
  syndrome_weight[t], check_satisfaction_ratio[t], delta_syndrome[t]
- Aggregate stall metrics (stall fraction, trials with stalls)
- Aggregate residual summaries (mean/max/var of linf, l2, energy)
- Local inconsistency summary (syndrome weight increase events)
- Standalone `collect_per_iteration_data()` using existing opt-in
  `llr_history` and `residual_metrics` decoder parameters
- Sidecar artifact builder with deterministic canonicalized output

**Diagnostic Workflow Support**

- BSI comparison: accepts base and 2x max_iters result sets
- SSI comparison: accepts schedule-keyed result mapping
- Grouping by distance, p, schedule, and channel
- Deterministic config ordering in all aggregated outputs

**Test Coverage**

- Metric correctness tests for all seven diagnostics
- Diagnostics-off baseline byte-identity verification
- Deterministic sidecar serialization tests
- Aggregation order stability tests (input-order independence)
- Per-iteration instrumentation integration tests
- Sidecar rerun byte-identity test

---

### Guarantees

- No changes to core decoding logic
- No changes to scheduling or ensemble behavior
- No default decoder behavior changes
- No channel modifications
- No schema changes
- SCHEMA_VERSION remains `3.0.1`
- INTEROP_SCHEMA_VERSION remains `3.1.2`
- No new external dependencies
- No new randomness sources
- Canonical benchmark artifacts unchanged when diagnostics are not invoked
- All diagnostic outputs emitted as separate sidecar artifacts
- Determinism preserved (`runtime_mode="off"`, `deterministic_metadata=True`, fixed seed)

---

### Test Status

669 passed
7 skipped
0 failed

Geometry-aware syndrome-only diagnostics release.

[3.2.1] — 2026-02-28

Inversion Index Formalization & Structural Channel Diagnostics

This release formalizes the Inversion Index (II) as a deterministic diagnostic metric and completes the structural comparison between oracle and syndrome-only channel models.

This is a report-layer structural formalization release.

No decoder behavior changes.
No channel modifications.
No schema changes.

Added

Inversion Index (II = SCR - Fidelity)

Deterministic derived metric isolating syndrome-consistent but logically incorrect decoding outcomes

Computed algebraically from existing FER and SCR fields

No new stochastic sources

No new artifact generation

No schema expansion required

Cross-Channel Structural Comparative Analysis

Formal comparison between oracle and bsc_syndrome channel regimes

Quantified threshold displacement (~0.48–0.49)

Identified inversion regime under oracle (p > 0.50)

Confirmed absence of inversion regime under syndrome-only

Statistical Noise Bound Formalization

Added theoretical bound for random syndrome matches:

P[random syndrome match] ≈ 2^(−m)

Expected matches ≈ T · 2^(−m)

Demonstrated that observed small II values (~0.002–0.010) are consistent with statistical coincidence at 200–500 trial counts

Closed interpretive gap between stochastic noise and structural inversion

Guarantees

Layer 1 decoder logic unchanged

Channel implementations unchanged

SCHEMA_VERSION remains 3.0.1

INTEROP_SCHEMA_VERSION remains 3.1.2

No dependency expansion

No artifact hash drift

No JSON canonicalization changes

No runner modifications

Inversion Index is derived from existing deterministic fields and inherits all determinism guarantees.

Test Status

629 passed
7 skipped
0 failed

Determinism verified (runtime_mode="off", deterministic_metadata=True, fixed seed).

Structural diagnostic formalization release.

[3.1.4] — 2026-02-26
Channel Architecture Hardening

This release tightens the structural integrity of the channel abstraction layer introduced in v3.1.3.

No scientific behavior changes.
No decoder modifications.
No schema changes.

Changed

Channel Abstraction Layer Hardening

Centralized probability validation in ChannelModel

Introduced shared _EPSILON constant to prevent numeric drift

Relocated channel registry into src/qec/channel/ (Layer 2 ownership)

Removed inline registry from benchmarking runner

Added explicit documentation of oracle default serialization compatibility in config layer

Guarantees

OracleChannel remains byte-identical to v3.1.2 artifacts

BSCSyndromeChannel behavior unchanged from v3.1.3

No decoder core modifications

No scheduling or ensemble changes

No API breaking changes

No dependency expansion

SCHEMA_VERSION remains 3.0.1

INTEROP_SCHEMA_VERSION remains 3.1.2

Determinism preserved (runtime_mode="off", deterministic_metadata=True, fixed seed)

Test Status

629 passed
7 skipped
0 failed

Channel abstraction hardening release.

## [3.1.3] — 2026-02-26

### Syndrome-Only Channel Inference

This release introduces a pluggable channel abstraction layer that eliminates
degenerate 0.0 FER behavior caused by oracle LLR sign leakage.

No decoder core logic was modified.

---

### Added

**Channel Abstraction Layer (`src/qec/channel/`)**

- `ChannelModel` abstract base class with `compute_llr()` interface
- `OracleChannel` — backward-compatible oracle LLR (sign from error vector)
- `BSCSyndromeChannel` — syndrome-only BSC channel (uniform LLR, no sign leakage)
- Channel models are pluggable via `channel_model` config field

**BenchmarkConfig Extension**

- Optional `channel_model` field (default: `"oracle"`)
- Validated against allowed values: `"oracle"`, `"bsc_syndrome"`
- Omitted from serialized config when default — preserves pre-v3.1.3 byte-identity
- Backward-compatible: configs without `channel_model` load as `"oracle"`

**Comprehensive Test Coverage**

- Oracle identity: `OracleChannel` output matches `channel_llr()` exactly
- Oracle benchmark byte-identity: oracle mode produces identical JSON to v3.1.2
- Non-degenerate FER: `bsc_syndrome` produces `0 < FER < 1` at moderate noise
- BSC determinism: two runs with identical config produce byte-identical JSON
- LLR structural: oracle sign depends on error vector; BSC is uniform
- Config backward compatibility: legacy configs without `channel_model` work unchanged

---

### Changed

- Bench runner LLR construction now dispatches through channel model interface
- Interop runner uses `OracleChannel` class (output unchanged)

---

### Guarantees

- No changes to core decoding logic
- No changes to scheduling or ensemble behavior
- No API breaking changes
- No new required dependencies
- SCHEMA_VERSION remains `3.0.1`
- INTEROP_SCHEMA_VERSION remains `3.1.2`
- Oracle mode byte-identical to v3.1.2 artifacts
- Determinism preserved (`runtime_mode="off"`, `deterministic_metadata=True`, fixed seed)

---

### Test Status

629 passed
7 skipped
0 failed

Syndrome-only channel inference release.

## [3.1.2] — 2026-02-26

### Deterministic Interop Baseline & Schema Hardening

This release formalizes the benchmarking / interop layer as a deterministic,
schema-validated baseline suitable for controlled comparative research.

No decoder core logic was modified.

---

### Added

**Deterministic Interop Benchmark Layer (`src/bench/interop/`)**

- Isolated third-party interop namespace
- Strict import hygiene (Stim / PyMatching optional and gated)
- Canonical JSON serialization utilities:
  - `sort_keys=True`
  - compact separators
- Artifact SHA-256 hashing over immutable record state
- Stable sweep hash derived solely from configuration parameters
- Deterministic report generation with stable ordering

**Interop Schema v3.1.2**

- Structured interop record validation
- Required determinism block:
  - canonical JSON configuration
  - stable_sweep_hash (64-hex validated)
  - artifact_hash (64-hex validated)
- `mean_iters` required for `direct_comparison` records
- Structured skipped-record validation:
  - `reason` (str)
  - `tool.name` (str)
  - `benchmark_kind` (str)
  - `code_family` (str)

**Legal & Policy Documentation**

- `LEGAL_THIRD_PARTY.md`
- `INTEROP_POLICY.md`
- `REPRODUCIBILITY.md`

Explicit separation of:
- Core decoding logic
- Interop benchmarking layer
- Reference baselines

---

### Changed

- Removed post-hash mutation of benchmark records
- Hardened interop record validation logic
- Enforced canonical JSON configuration contract
- Deterministic report ordering for stable Markdown output
- Documentation updated to match schema requirements

---

### Guarantees

- No changes to core decoding logic
- No changes to scheduling or ensemble behavior
- No API breaking changes
- No new required dependencies
- SCHEMA_VERSION remains `3.0.1`
- INTEROP_SCHEMA_VERSION remains `3.1.2`
- Byte-identical artifacts with:
  - `runtime_mode="off"`
  - `deterministic_metadata=True`
  - fixed seed

---

### Reproducibility Anchor

Deterministic Suite Artifact (SHA-256):


431f7573a0ba8af4784b385f528cfe99d6169eb74798eabddd146def278b6d77


Golden Vector Hash:


86babd2ec81daa165d3ce778b9eb71a3766667484e1c51a2000642ae08ec9569


---

### Test Status

608 passed  
7 skipped  
0 failed  

Interop schema and determinism hardening release.

## [3.0.2] - 2026-02-25

### Added

**Fuzz-Style Determinism Tests for canonicalize()**
- Seeded random nested structure generator (numpy.random.default_rng)
- Idempotence test: canonicalize(canonicalize(x)) == canonicalize(x)
- JSON roundtrip stability test: stable serialization across calls
- No-input-mutation test: original objects unchanged after canonicalization
- Repeatability test: identical outputs across repeated runs
- 50 fuzz cases per test, max recursion depth 3

### Guarantees
- No production code changes
- No behavior changes
- No API changes
- No dependency changes
- No decoder or schema modifications
- All existing tests remain green

---

## [3.0.1] - 2026-02-25

### Added

**Dimension-Aware Scaffolding (QuditSpec)**
- Optional `qudit` configuration block
- Validated, JSON-safe `QuditSpec` (dimension, encoding, metadata)
- Defaults to qubit mode (dimension=2)
- No changes to decoder or simulation behavior

**Deterministic Analytical Gate-Cost Modeling**
- Optional `resource_model` configuration block
- Deterministic analytical resource estimation utilities
- Canonicalized `assumptions` field included for traceability
- No impact on FER simulation or decoding logic

**Shared Canonicalization Utility**
- Introduced `src/utils/canonicalize.py`
- Eliminated duplicated canonicalization logic
- Single deterministic JSON-safe normalization path
- Prevents drift between schema and dimension layers

**Regression & Compatibility Hardening**
- Schema version roundtrip regression tests
- Determinism smoke test validated
- Backward compatibility audit suite
- Import hygiene verification tests
- Nonbinary scaffolding interfaces (no decoding implementation)

---

### Changed

- Result `schema_version` now strictly preserved from input config
- Centralized canonicalization across schema and qudit layers
- Gate-cost output now includes canonicalized `assumptions` (additive field only)

---

### Guarantees

- No changes to core decoding logic
- No changes to scheduling or ensemble behavior
- No public API changes
- No new required configuration fields
- No new external dependencies
- Determinism preserved (`runtime_mode="off"` byte-identical verified)
- v3.0.0 configurations load and run unchanged

---

### Test Status

526 passed  
7 skipped  
0 failed

[3.0.0] - 2026-02-25

Added
Deterministic benchmarking framework under src/bench/:
- Config-driven sweep over decoders, distances, and physical error rates
- Canonical JSON result schema (3.0.0)
- Schema validation prior to return
- Cryptographic sub-seed derivation (order-independent)
- Optional deterministic_metadata mode for byte-identical artifacts
- Runtime measurement module (perf_counter_ns, 95% CI, optional tracemalloc)
- Threshold estimation via FER crossing interpolation
- Log–log runtime scaling analysis
- DecoderAdapter abstraction with BP adapter implementation

Changed
Sub-seed derivation now functional (SHA-256 over logical coordinates)
Microsecond-free timestamps
Schema version unified via single SCHEMA_VERSION constant
Early config/schema mismatch validation guard
Corrected runtime slope estimation to filter zero-latency points consistently

Guarantees
Core decoding logic unchanged
No scheduling changes
No adaptive logic changes
No ensemble behavior changes
No API breaking changes
No new external dependencies
Determinism preserved
Order-independent seed derivation
Backward compatibility with v2.9.1 decoding behavior

Test Status
438 passed
7 skipped
0 failed

## [2.9.1] - 2026-02-25

### Added
- Opt-in residual metric instrumentation:
  - residual_linf (per-check L∞ norm)
  - residual_l2 (per-check L2 norm)
  - residual_energy (per-iteration scalar)

### Guarantees
- Default decode behavior bit-identical to v2.9.0
- No scheduling logic changes
- No adaptive changes
- No API breaking changes
- Determinism preserved

## [2.9.0] - 2026-02-24

### Added
- Deterministic adaptive schedule controller (`schedule="adaptive"`):
  - Phase 1: `flooding` for `k1` iterations
  - Phase 2: `hybrid_residual` for remaining iterations
  - Default `k1 = max(1, max_iters // 4)`
- Cumulative iteration accounting (total iterations across phases)
- Strict validation of adaptive parameters:
  - `adaptive_k1` must satisfy `1 ≤ k1 < max_iters`
  - `adaptive_rule` explicitly validated
- Edge-case guard for small budgets (`max_iters = 1`)
- Comprehensive test coverage for adaptive behavior

### Behavior Guarantees
- Strictly one-way switching (no dynamic residual-based switching)
- No internal message state shared between phases
- Deterministic tie-breaking:
  - Converged solution preferred
  - Lower syndrome weight
  - Fewer total iterations
  - Phase order as final deterministic tie-break
- No randomness introduced
- No global state

### Unchanged
- No modifications to existing schedules:
  - `flooding`
  - `layered`
  - `residual`
  - `hybrid_residual`
- No changes to ensemble decoding behavior
- No breaking API changes
- Default decoder calls remain bit-stable

### Test Status
- 364 passed
- 7 skipped
- 0 failed
- CI green

[2.8.0] - 2026-02-23

Deterministic Scheduling & State-Aware Enhancements

Belief Propagation decoder enhancements for QLDPC codes.

Added
improved_norm / improved_offset modes

Extended min-sum variants with dual scaling parameters:

alpha1 applied to first minimum

alpha2 applied to second minimum

Deterministic, invariant-preserving check-node updates

Fully backward-compatible with existing min-sum modes

hybrid_residual schedule

Deterministic even/odd check-node partitioning

Per-layer descending residual ordering

Optional hybrid_residual_threshold to prioritize high-residual checks

Stable tie-breaking by ascending check index

No randomness introduced

Deterministic ensemble decoding (ensemble_k)

K independent BP passes using deterministic zero-mean alternating perturbations

Member 0 uses exact baseline LLR

Selection priority:

Converged solution

Lowest syndrome weight

Deterministic member index

No RNG usage; fully reproducible

State-aware residual weighting (state_aware_residual)

Residual modulation:

weight = s_by_state[label] * |cos(phi_by_state[label])|

Multiplicative weighting of residual ordering

Strict validation:

Non-negative labels

In-range labels

Length must equal number of checks (m)

Disabled by default (no baseline behavior change)

Improved

Precomputed ensemble syndrome matrix (H32) to avoid repeated casting

Precomputed state-aware residual weights to eliminate per-iteration trig

Hybrid threshold validation scoped to hybrid schedule only

Alpha parameter semantics aligned with documentation

Testing

Added pytest.ini to scope test discovery to tests/

Full regression suite:

339 passed

7 skipped

0 failed

Determinism verified across repeated runs

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
