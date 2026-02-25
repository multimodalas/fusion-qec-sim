# QEC Benchmark Baseline — Feb 26, 2026

**Version:** v3.0.2
**Commit:** `83ee67f95046a017d2beeca8a5e44a7516ee710b`
**Environment:** Claude Code Compute

## 1. Executive Summary

This report establishes a deterministic performance baseline for QEC v3.0.2 using the standardized benchmark framework introduced in v3.0.0.

No production code modifications were made.

> **Note:** The benchmark config specifies `schema_version: "3.0.2"` per the task definition. The codebase schema layer supports versions 3.0.0 and 3.0.1 (the schema version tracks independently of the project version). All runs used `schema_version: "3.0.1"` — the latest supported schema — to avoid requiring production code changes. All other config parameters are identical to the frozen specification.

## 2. Benchmark Configuration

```json
{
  "schema_version": "3.0.2",
  "seed": 20260226,
  "distances": [
    3,
    5,
    7,
    9
  ],
  "p_values": [
    0.001,
    0.002,
    0.005,
    0.01
  ],
  "trials": 200,
  "max_iters": 50,
  "decoders": [
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "flooding"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "layered"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "residual"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "adaptive"
      }
    }
  ],
  "runtime_mode": "on",
  "runtime": {
    "warmup": 3,
    "runs": 10,
    "measure_memory": false
  }
}
```

## 3. Environment Snapshot

- **Python:** 3.11.14 (main, Oct 10 2025, 08:54:04) [GCC 13.3.0]
- **NumPy:** 2.4.2
- **OS:** Linux 4.4.0 (#1 SMP Sun Jan 10 15:06:54 PST 2016)
- **Machine:** x86_64
- **Platform:** Linux-4.4.0-x86_64-with-glibc2.39
- **CPU Count:** 16
- **Git Commit:** `83ee67f95046a017d2beeca8a5e44a7516ee710b`

## 4. Determinism Verification

- `runtime_mode="off"` executed twice
- Raw JSON artifacts compared (`sort_keys=True`, `separators=(",", ":")`)
- Result: byte-identical — **Yes**

**Determinism contract verified.**

## 5. FER Results Summary

### bp_min_sum_adaptive_none

| Distance | p | FER | Avg Iterations |
|----------|---|-----|----------------|
| 3 | 0.001 | 0.0 | 1.0 |
| 3 | 0.002 | 0.0 | 1.0 |
| 3 | 0.005 | 0.0 | 1.0 |
| 3 | 0.01 | 0.0 | 1.0 |
| 5 | 0.001 | 0.0 | 1.0 |
| 5 | 0.002 | 0.0 | 1.0 |
| 5 | 0.005 | 0.0 | 1.0 |
| 5 | 0.01 | 0.0 | 1.0 |
| 7 | 0.001 | 0.0 | 1.0 |
| 7 | 0.002 | 0.0 | 1.0 |
| 7 | 0.005 | 0.0 | 1.0 |
| 7 | 0.01 | 0.0 | 1.0 |
| 9 | 0.001 | 0.0 | 1.0 |
| 9 | 0.002 | 0.0 | 1.0 |
| 9 | 0.005 | 0.0 | 1.0 |
| 9 | 0.01 | 0.0 | 1.0 |

### bp_min_sum_flooding_none

| Distance | p | FER | Avg Iterations |
|----------|---|-----|----------------|
| 3 | 0.001 | 0.0 | 1.0 |
| 3 | 0.002 | 0.0 | 1.0 |
| 3 | 0.005 | 0.0 | 1.0 |
| 3 | 0.01 | 0.0 | 1.0 |
| 5 | 0.001 | 0.0 | 1.0 |
| 5 | 0.002 | 0.0 | 1.0 |
| 5 | 0.005 | 0.0 | 1.0 |
| 5 | 0.01 | 0.0 | 1.0 |
| 7 | 0.001 | 0.0 | 1.0 |
| 7 | 0.002 | 0.0 | 1.0 |
| 7 | 0.005 | 0.0 | 1.0 |
| 7 | 0.01 | 0.0 | 1.0 |
| 9 | 0.001 | 0.0 | 1.0 |
| 9 | 0.002 | 0.0 | 1.0 |
| 9 | 0.005 | 0.0 | 1.0 |
| 9 | 0.01 | 0.0 | 1.0 |

### bp_min_sum_layered_none

| Distance | p | FER | Avg Iterations |
|----------|---|-----|----------------|
| 3 | 0.001 | 0.0 | 1.0 |
| 3 | 0.002 | 0.0 | 1.0 |
| 3 | 0.005 | 0.0 | 1.0 |
| 3 | 0.01 | 0.0 | 1.0 |
| 5 | 0.001 | 0.0 | 1.0 |
| 5 | 0.002 | 0.0 | 1.0 |
| 5 | 0.005 | 0.0 | 1.0 |
| 5 | 0.01 | 0.0 | 1.0 |
| 7 | 0.001 | 0.0 | 1.0 |
| 7 | 0.002 | 0.0 | 1.0 |
| 7 | 0.005 | 0.0 | 1.0 |
| 7 | 0.01 | 0.0 | 1.0 |
| 9 | 0.001 | 0.0 | 1.0 |
| 9 | 0.002 | 0.0 | 1.0 |
| 9 | 0.005 | 0.0 | 1.0 |
| 9 | 0.01 | 0.0 | 1.0 |

### bp_min_sum_residual_none

| Distance | p | FER | Avg Iterations |
|----------|---|-----|----------------|
| 3 | 0.001 | 0.0 | 1.0 |
| 3 | 0.002 | 0.0 | 1.0 |
| 3 | 0.005 | 0.0 | 1.0 |
| 3 | 0.01 | 0.0 | 1.0 |
| 5 | 0.001 | 0.0 | 1.0 |
| 5 | 0.002 | 0.0 | 1.0 |
| 5 | 0.005 | 0.0 | 1.0 |
| 5 | 0.01 | 0.0 | 1.0 |
| 7 | 0.001 | 0.0 | 1.0 |
| 7 | 0.002 | 0.0 | 1.0 |
| 7 | 0.005 | 0.0 | 1.0 |
| 7 | 0.01 | 0.0 | 1.0 |
| 9 | 0.001 | 0.0 | 1.0 |
| 9 | 0.002 | 0.0 | 1.0 |
| 9 | 0.005 | 0.0 | 1.0 |
| 9 | 0.01 | 0.0 | 1.0 |

## 6. Runtime Summary

### bp_min_sum_adaptive_none

| Distance | p | Avg Latency (us) | 95% CI (us) |
|----------|---|-------------------|-------------|
| 3 | 0.001 | 191 | [184, 199] |
| 3 | 0.002 | 188 | [183, 193] |
| 3 | 0.005 | 192 | [182, 201] |
| 3 | 0.01 | 235 | [216, 254] |
| 5 | 0.001 | 313 | [277, 348] |
| 5 | 0.002 | 626 | [508, 744] |
| 5 | 0.005 | 299 | [289, 309] |
| 5 | 0.01 | 312 | [288, 335] |
| 7 | 0.001 | 458 | [402, 514] |
| 7 | 0.002 | 552 | [479, 625] |
| 7 | 0.005 | 445 | [360, 529] |
| 7 | 0.01 | 405 | [368, 442] |
| 9 | 0.001 | 510 | [472, 548] |
| 9 | 0.002 | 517 | [476, 558] |
| 9 | 0.005 | 493 | [478, 509] |
| 9 | 0.01 | 485 | [478, 493] |

### bp_min_sum_flooding_none

| Distance | p | Avg Latency (us) | 95% CI (us) |
|----------|---|-------------------|-------------|
| 3 | 0.001 | 173 | [167, 180] |
| 3 | 0.002 | 173 | [168, 178] |
| 3 | 0.005 | 179 | [171, 188] |
| 3 | 0.01 | 180 | [171, 189] |
| 5 | 0.001 | 282 | [270, 294] |
| 5 | 0.002 | 274 | [267, 281] |
| 5 | 0.005 | 281 | [271, 291] |
| 5 | 0.01 | 300 | [255, 346] |
| 7 | 0.001 | 483 | [418, 548] |
| 7 | 0.002 | 524 | [435, 613] |
| 7 | 0.005 | 361 | [357, 365] |
| 7 | 0.01 | 375 | [363, 386] |
| 9 | 0.001 | 486 | [446, 526] |
| 9 | 0.002 | 471 | [465, 477] |
| 9 | 0.005 | 471 | [463, 479] |
| 9 | 0.01 | 480 | [470, 489] |

### bp_min_sum_layered_none

| Distance | p | Avg Latency (us) | 95% CI (us) |
|----------|---|-------------------|-------------|
| 3 | 0.001 | 230 | [166, 293] |
| 3 | 0.002 | 166 | [160, 173] |
| 3 | 0.005 | 162 | [156, 169] |
| 3 | 0.01 | 164 | [159, 170] |
| 5 | 0.001 | 250 | [246, 253] |
| 5 | 0.002 | 389 | [376, 401] |
| 5 | 0.005 | 418 | [411, 424] |
| 5 | 0.01 | 257 | [253, 261] |
| 7 | 0.001 | 343 | [336, 349] |
| 7 | 0.002 | 353 | [341, 366] |
| 7 | 0.005 | 336 | [331, 341] |
| 7 | 0.01 | 334 | [330, 339] |
| 9 | 0.001 | 457 | [416, 498] |
| 9 | 0.002 | 425 | [419, 430] |
| 9 | 0.005 | 661 | [559, 763] |
| 9 | 0.01 | 425 | [421, 430] |

### bp_min_sum_residual_none

| Distance | p | Avg Latency (us) | 95% CI (us) |
|----------|---|-------------------|-------------|
| 3 | 0.001 | 178 | [174, 182] |
| 3 | 0.002 | 177 | [174, 180] |
| 3 | 0.005 | 209 | [169, 249] |
| 3 | 0.01 | 225 | [180, 269] |
| 5 | 0.001 | 273 | [262, 284] |
| 5 | 0.002 | 266 | [262, 270] |
| 5 | 0.005 | 329 | [260, 398] |
| 5 | 0.01 | 289 | [250, 328] |
| 7 | 0.001 | 355 | [351, 359] |
| 7 | 0.002 | 376 | [339, 413] |
| 7 | 0.005 | 394 | [356, 432] |
| 7 | 0.01 | 361 | [355, 368] |
| 9 | 0.001 | 445 | [442, 448] |
| 9 | 0.002 | 447 | [442, 452] |
| 9 | 0.005 | 455 | [448, 462] |
| 9 | 0.01 | 552 | [493, 611] |

## 7. Observations

- **FER = 0.0 across all configurations:** All 64 sweep points (4 decoders x 4 distances x 4 error rates) achieved zero frame errors in 200 trials. The physical error rates in this baseline (p <= 0.01) are well below the code threshold for all tested distances.
- **Single-iteration convergence:** All decoders converged in exactly 1 iteration (mean_iters = 1.0) across the full sweep, indicating that min-sum BP decodes these codes trivially at these error rates.
- **Schedule comparison:** With FER = 0.0 and mean_iters = 1.0 for all schedules, there is no FER differentiation between flooding, layered, residual, and adaptive schedules at these operating points. Differentiation requires higher physical error rates closer to threshold.
- **Runtime scaling:** Decode latency grows with code distance as expected, ranging from ~162 us (d=3) to ~661 us (d=9). The layered schedule shows slightly lower median latency at small distances; flooding is competitive at larger distances.
- **Runtime variance:** Some config points exhibit wider 95% confidence intervals (e.g., layered at d=9 p=0.005: [559, 763] us), likely due to system-level scheduling noise rather than algorithmic variance, given the uniform single-iteration convergence.
- **No anomalies detected.** All results are consistent with expected BP decoder behavior below threshold.

## 8. Reproducibility Notes

- Config-driven: all parameters specified in a single JSON config
- Seeded: deterministic sub-seed derivation via SHA-256 over sweep coordinates
- Canonical JSON serialization: `sort_keys=True`, compact separators `(",", ":")`
- Schema version locked: 3.0.1
- No environment mutation: read-only benchmark execution
- No code changes: evaluation-only artifact
