# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![Release v3.0.1](https://img.shields.io/badge/release-v3.0.1-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v3.0.1)

License: CC-BY-4.0

Deterministic QLDPC CSS quantum error correction framework with invariant-safe algebraic construction, multi-mode belief propagation, ensemble and residual scheduling, statistically rigorous FER simulation, and a schema-validated deterministic benchmarking framework.

---

## Current Release

**v3.0.1 — Legacy Compatibility & High-Dimensional Readiness**

- Optional dimension-aware `QuditSpec`
- Deterministic analytical gate-cost modeling
- Centralized canonicalization
- Schema version preservation invariant
- Backward compatibility guaranteed
- No new dependencies
- No decoder behavior changes

---

## Determinism Guarantees

- No hidden randomness
- Explicit seed control
- Order-independent SHA-256 sub-seeds
- Canonical JSON serialization
- Stable sweep ordering
- `runtime_mode="off"` → reproducible artifacts

Determinism is a first-class architectural invariant.

---

## Architecture

Core decoding layer:
- Additive invariant QLDPC CSS construction
- Multi-mode BP (sum-product, min-sum, norm, offset)
- Flooding, layered, residual, hybrid schedules
- Ensemble decoding
- OSD-0 / OSD-1 / OSD-CS
- Deterministic decimation

Benchmarking layer (isolated under `src/bench/`):
- Config-driven execution
- Schema-validated results
- Canonical JSON output
- Threshold and runtime analysis

Core decoding logic is not modified by benchmarking features.

---

## Documentation

- Full release history: **CHANGELOG.md**
- Forward direction: **ROADMAP.md**

---

Author: Trent Slade  
ORCID: https://orcid.org/0009-0002-4515-9237  

Small is beautiful.  
Determinism is holy.
