# Interop Policy

This document defines the boundary between core QEC and the benchmarking
interop layer, and the rules governing third-party tool integration.

## Core vs Bench Boundary (Hard Rule)

```
src/                    ← CORE: stdlib + numpy only. No third-party QEC tools.
  ├── __init__.py
  ├── qec_qldpc_codes.py
  ├── decoder/
  ├── simulation/
  ├── utils/
  └── bench/
       ├── config.py     ← Core bench infra (no third-party imports)
       ├── schema.py      ← Core bench infra (no third-party imports)
       ├── runner.py      ← Core bench infra (no third-party imports)
       ├── report.py      ← Core bench infra (no third-party imports)
       ├── compare.py     ← Core bench infra (no third-party imports)
       ├── adapters/      ← Core bench infra (no third-party imports)
       └── interop/       ← INTEROP LAYER: optional third-party tools here ONLY
            ├── imports.py
            ├── runners.py
            ├── env.py
            └── serialize.py
```

**Rule**: No module outside `bench/interop/` may import `stim`, `pymatching`,
or any other optional third-party QEC tool. This is enforced by tests.

## Import Hygiene Required Pattern

All third-party imports in `bench/interop/` MUST use the following pattern:

```python
try:
    import stim
    HAS_STIM = True
except ImportError:
    stim = None  # type: ignore[assignment]
    HAS_STIM = False
```

Functions that require an optional dependency must check availability and
return a structured "tool unavailable" result rather than raising:

```python
def run_stim_baseline(config):
    if not HAS_STIM:
        return {"status": "skipped", "reason": "stim not installed"}
    # ... actual implementation ...
```

## Determinism Contract for Interop

1. All random number generation uses `numpy.random.default_rng(seed)` with
   explicit seeds.
2. Sweep ordering is deterministic: sorted by tool name, then code family,
   then parameters.
3. When `runtime_mode="off"`, timing fields are omitted and all output
   artifacts must be byte-identical across repeated runs.
4. JSON serialization uses `sort_keys=True, separators=(",", ":")`.

## Comparability Taxonomy (REQUIRED Labels)

Every benchmark record MUST include a `benchmark_kind` field with one of:

| Value                  | Meaning                                        |
|------------------------|------------------------------------------------|
| `direct_comparison`    | Same code family, same representation, directly comparable results |
| `reference_baseline`   | Different tool/representation; context-only, NOT a direct comparison |

**Rules**:
- Stim/PyMatching results on surface codes or repetition codes are
  `reference_baseline` (different representation than QEC-native QLDPC CSS).
- QEC-native results on QLDPC CSS codes are `direct_comparison` when
  comparing QEC decoder variants against each other.
- Reports MUST visually separate these two categories. Mixing them in a
  single table is prohibited.
- No benchmark record may claim "superiority" of one tool over another
  across different `benchmark_kind` categories.
