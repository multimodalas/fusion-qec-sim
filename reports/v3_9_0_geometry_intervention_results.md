# v3.9.0 — Geometry Intervention Results

## Overview

v3.9.0 introduces two deterministic channel-geometry interventions
and a per-iteration BP energy trace diagnostic:

1. **Centered syndrome-field projection** — removes uniform syndrome bias
2. **Parity-derived pseudo-prior injection** — weak deterministic variable prior
3. **BP energy trace** — measures LLR-belief correlation per iteration

## Experimental Setup

- Seed: 42
- Distances: [3, 5, 7]
- P values: [0.01, 0.015, 0.02]
- Trials per (distance, p): 200
- Max iterations: 50
- BP mode: min_sum
- Modes evaluated: 11

## Determinism Verification

- Baseline invariant: **PASS**
- Full determinism check: **PASS**
- All modes produce identical results across repeated runs.

## DPS Results

| Mode | p=0.01 | p=0.015 | p=0.02 |
| --- | --- | --- | --- |
| baseline | -0.000000 ** | -0.000000 ** | -0.000000 ** |
| rpc_only | -0.000000 ** | -0.000000 ** | -0.000000 ** |
| geom_v1_only | -0.000000 ** | -0.000000 ** | -0.000000 ** |
| rpc_geom | -0.000000 ** | -0.000000 ** | -0.000000 ** |
| centered | 0.070474 | 0.064843 | 0.040074 |
| prior | 0.070474 | 0.064843 | 0.040074 |
| centered_prior | 0.070474 | 0.064843 | 0.040074 |
| geom_centered | 0.070474 | 0.064843 | 0.040074 |
| geom_centered_prior | 0.070474 | 0.064843 | 0.040074 |
| rpc_centered | 0.070474 | 0.064843 | 0.040074 |
| rpc_centered_prior | 0.070474 | 0.064843 | 0.040074 |

## FER Results

| Mode | p | d | FER | Frame Errors | Avg Iters |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.01 | 3 | 0.0000 | 0 | 1.00 |
| baseline | 0.01 | 5 | 0.0000 | 0 | 1.00 |
| baseline | 0.01 | 7 | 0.0000 | 0 | 1.00 |
| baseline | 0.015 | 3 | 0.0000 | 0 | 1.00 |
| baseline | 0.015 | 5 | 0.0000 | 0 | 1.00 |
| baseline | 0.015 | 7 | 0.0000 | 0 | 1.00 |
| baseline | 0.02 | 3 | 0.0000 | 0 | 1.00 |
| baseline | 0.02 | 5 | 0.0000 | 0 | 1.00 |
| baseline | 0.02 | 7 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.01 | 3 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.01 | 5 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.01 | 7 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.015 | 3 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.015 | 5 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.015 | 7 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.02 | 3 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.02 | 5 | 0.0000 | 0 | 1.00 |
| rpc_only | 0.02 | 7 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.01 | 3 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.01 | 5 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.01 | 7 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.015 | 3 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.015 | 5 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.015 | 7 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.02 | 3 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.02 | 5 | 0.0000 | 0 | 1.00 |
| geom_v1_only | 0.02 | 7 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.01 | 3 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.01 | 5 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.01 | 7 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.015 | 3 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.015 | 5 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.015 | 7 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.02 | 3 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.02 | 5 | 0.0000 | 0 | 1.00 |
| rpc_geom | 0.02 | 7 | 0.0000 | 0 | 1.00 |
| centered | 0.01 | 3 | 0.2900 | 58 | 15.21 |
| centered | 0.01 | 5 | 0.4400 | 88 | 22.56 |
| centered | 0.01 | 7 | 0.5550 | 111 | 28.20 |
| centered | 0.015 | 3 | 0.4100 | 82 | 21.09 |
| centered | 0.015 | 5 | 0.6000 | 120 | 30.40 |
| centered | 0.015 | 7 | 0.7450 | 149 | 37.51 |
| centered | 0.02 | 3 | 0.5600 | 112 | 28.44 |
| centered | 0.02 | 5 | 0.6600 | 132 | 33.34 |
| centered | 0.02 | 7 | 0.8100 | 162 | 40.69 |
| prior | 0.01 | 3 | 0.2900 | 58 | 15.21 |
| prior | 0.01 | 5 | 0.4400 | 88 | 22.56 |
| prior | 0.01 | 7 | 0.5550 | 111 | 28.20 |
| prior | 0.015 | 3 | 0.4100 | 82 | 21.09 |
| prior | 0.015 | 5 | 0.6000 | 120 | 30.40 |
| prior | 0.015 | 7 | 0.7450 | 149 | 37.51 |
| prior | 0.02 | 3 | 0.5600 | 112 | 28.44 |
| prior | 0.02 | 5 | 0.6600 | 132 | 33.34 |
| prior | 0.02 | 7 | 0.8100 | 162 | 40.69 |
| centered_prior | 0.01 | 3 | 0.2900 | 58 | 15.21 |
| centered_prior | 0.01 | 5 | 0.4400 | 88 | 22.56 |
| centered_prior | 0.01 | 7 | 0.5550 | 111 | 28.20 |
| centered_prior | 0.015 | 3 | 0.4100 | 82 | 21.09 |
| centered_prior | 0.015 | 5 | 0.6000 | 120 | 30.40 |
| centered_prior | 0.015 | 7 | 0.7450 | 149 | 37.51 |
| centered_prior | 0.02 | 3 | 0.5600 | 112 | 28.44 |
| centered_prior | 0.02 | 5 | 0.6600 | 132 | 33.34 |
| centered_prior | 0.02 | 7 | 0.8100 | 162 | 40.69 |
| geom_centered | 0.01 | 3 | 0.2900 | 58 | 15.21 |
| geom_centered | 0.01 | 5 | 0.4400 | 88 | 22.56 |
| geom_centered | 0.01 | 7 | 0.5550 | 111 | 28.20 |
| geom_centered | 0.015 | 3 | 0.4100 | 82 | 21.09 |
| geom_centered | 0.015 | 5 | 0.6000 | 120 | 30.40 |
| geom_centered | 0.015 | 7 | 0.7450 | 149 | 37.51 |
| geom_centered | 0.02 | 3 | 0.5600 | 112 | 28.44 |
| geom_centered | 0.02 | 5 | 0.6600 | 132 | 33.34 |
| geom_centered | 0.02 | 7 | 0.8100 | 162 | 40.69 |
| geom_centered_prior | 0.01 | 3 | 0.2900 | 58 | 15.21 |
| geom_centered_prior | 0.01 | 5 | 0.4400 | 88 | 22.56 |
| geom_centered_prior | 0.01 | 7 | 0.5550 | 111 | 28.20 |
| geom_centered_prior | 0.015 | 3 | 0.4100 | 82 | 21.09 |
| geom_centered_prior | 0.015 | 5 | 0.6000 | 120 | 30.40 |
| geom_centered_prior | 0.015 | 7 | 0.7450 | 149 | 37.51 |
| geom_centered_prior | 0.02 | 3 | 0.5600 | 112 | 28.44 |
| geom_centered_prior | 0.02 | 5 | 0.6600 | 132 | 33.34 |
| geom_centered_prior | 0.02 | 7 | 0.8100 | 162 | 40.69 |
| rpc_centered | 0.01 | 3 | 0.2900 | 58 | 15.21 |
| rpc_centered | 0.01 | 5 | 0.4400 | 88 | 22.56 |
| rpc_centered | 0.01 | 7 | 0.5550 | 111 | 28.20 |
| rpc_centered | 0.015 | 3 | 0.4100 | 82 | 21.09 |
| rpc_centered | 0.015 | 5 | 0.6000 | 120 | 30.40 |
| rpc_centered | 0.015 | 7 | 0.7450 | 149 | 37.51 |
| rpc_centered | 0.02 | 3 | 0.5600 | 112 | 28.44 |
| rpc_centered | 0.02 | 5 | 0.6600 | 132 | 33.34 |
| rpc_centered | 0.02 | 7 | 0.8100 | 162 | 40.69 |
| rpc_centered_prior | 0.01 | 3 | 0.2900 | 58 | 15.21 |
| rpc_centered_prior | 0.01 | 5 | 0.4400 | 88 | 22.56 |
| rpc_centered_prior | 0.01 | 7 | 0.5550 | 111 | 28.20 |
| rpc_centered_prior | 0.015 | 3 | 0.4100 | 82 | 21.09 |
| rpc_centered_prior | 0.015 | 5 | 0.6000 | 120 | 30.40 |
| rpc_centered_prior | 0.015 | 7 | 0.7450 | 149 | 37.51 |
| rpc_centered_prior | 0.02 | 3 | 0.5600 | 112 | 28.44 |
| rpc_centered_prior | 0.02 | 5 | 0.6600 | 132 | 33.34 |
| rpc_centered_prior | 0.02 | 7 | 0.8100 | 162 | 40.69 |

## Energy Trace Analysis

| Mode | Energy Start | Energy End | Delta Energy | Monotonic | Oscillations |
| --- | --- | --- | --- | --- | --- |
| baseline | -2150.89 | -2150.89 | 0.00 | 100% | 0.00 |
| rpc_only | -9989.68 | -9989.68 | 0.00 | 100% | 0.00 |
| geom_v1_only | -1613.17 | -1613.17 | 0.00 | 100% | 0.00 |
| rpc_geom | -4384.60 | -4384.60 | 0.00 | 100% | 0.00 |
| centered | -2.95 | -2.95 | 0.00 | 100% | 0.00 |
| prior | -176.58 | -176.58 | 0.00 | 100% | 0.00 |
| centered_prior | -16.13 | -16.13 | 0.00 | 100% | 0.00 |
| geom_centered | -7.59 | -7.59 | 0.00 | 100% | 0.00 |
| geom_centered_prior | -19.10 | -19.10 | 0.00 | 100% | 0.00 |
| rpc_centered | 5.30 | -7097554390991545626164396101619540819968.00 | -7097554390991545626164396101619540819968.00 | 68% | 14.33 |
| rpc_centered_prior | -2413.64 | -99586778863982188253849914140903390314496.00 | -99586778863982188253849914140903390314496.00 | 72% | 13.04 |

## Intervention Comparison

### Centered Field Projection

- p=0.01: DPS delta = +0.070474
- p=0.015: DPS delta = +0.064843
- p=0.02: DPS delta = +0.040074

### Pseudo-Prior Injection

- p=0.01: DPS delta = +0.070474
- p=0.015: DPS delta = +0.064843
- p=0.02: DPS delta = +0.040074

### Combined (Centered + Prior)

- p=0.01: DPS delta = +0.070474
- p=0.015: DPS delta = +0.064843
- p=0.02: DPS delta = +0.040074

## Conclusion

**DPS sign flip detected** in the following configurations:

- baseline at p=0.01: DPS = -0.000000
- baseline at p=0.015: DPS = -0.000000
- baseline at p=0.02: DPS = -0.000000
- rpc_only at p=0.01: DPS = -0.000000
- rpc_only at p=0.015: DPS = -0.000000
- rpc_only at p=0.02: DPS = -0.000000
- geom_v1_only at p=0.01: DPS = -0.000000
- geom_v1_only at p=0.015: DPS = -0.000000
- geom_v1_only at p=0.02: DPS = -0.000000
- rpc_geom at p=0.01: DPS = -0.000000
- rpc_geom at p=0.015: DPS = -0.000000
- rpc_geom at p=0.02: DPS = -0.000000

Release readiness: **SAFE TO TAG v3.9.0**
