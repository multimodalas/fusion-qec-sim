"""
v8.4.0 — Generation Public API.

Exposes all public generation functions from a single module.
Import from here for stable access to the generation layer.

Layer 3 — Generation.
Does not import or modify the decoder (Layer 1).
"""

from __future__ import annotations

# ── v8.4.0: Tanner graph generation ────────────────────────────────
from src.qec.generation.tanner_graph_generator import (
    generate_tanner_graph_candidates,
)
from src.qec.generation.candidate_evaluation import (
    evaluate_tanner_graph_candidate,
)
from src.qec.generation.candidate_ranking import (
    rank_tanner_graph_candidates,
)
from src.qec.generation.export_generated_graph import (
    export_generated_graph,
)
