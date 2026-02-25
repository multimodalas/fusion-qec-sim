# Reproducibility Requirements

This document defines the reproducibility guarantees for QEC benchmark artifacts.

## Byte-Identical Artifacts (runtime_mode="off")

When `runtime_mode="off"` (the default), benchmark output artifacts MUST be
byte-identical across repeated runs given the same configuration. This means:

1. **No timestamps** in output: `created_utc` is set to a fixed epoch value
   (`"1970-01-01T00:00:00+00:00"`) when `deterministic_metadata=True`.
2. **No runtime measurements**: timing fields are `null` or omitted.
3. **Deterministic RNG**: all random number generation uses
   `numpy.random.default_rng(seed)` with explicit, config-specified seeds.
4. **Order-independent sub-seeds**: per-record seeds are derived via SHA-256
   from logical coordinates (decoder, distance, p), not loop iteration order.

## Canonical JSON Settings

All JSON serialization in benchmark output uses:

```python
json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

This ensures:
- Keys are alphabetically sorted at every nesting level.
- No trailing whitespace or variable indentation.
- Compact separators eliminate formatting variance.

## Stable Sweep Ordering Rules

Benchmark sweeps iterate in a fixed, deterministic order:

1. **Tools**: sorted alphabetically by tool name.
2. **Code families**: sorted alphabetically.
3. **Distances**: sorted numerically (ascending).
4. **Physical error rates (p_values)**: sorted numerically (ascending).
5. **Decoder variants**: sorted by adapter name, then by parameter dict
   (JSON-serialized with `sort_keys=True`).

## Artifact Hash Definition

Each benchmark output record includes a `stable_sweep_hash` computed as:

```python
import hashlib, json
payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
stable_sweep_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

The full artifact hash covers the entire serialized output:

```python
artifact_bytes = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
```

## Environment Capture Requirements

Each benchmark output includes an `environment` block with:

| Field            | Source                          | Deterministic? |
|------------------|---------------------------------|----------------|
| `platform`       | `platform.platform()`          | Machine-dependent |
| `python_version` | `sys.version`                  | Machine-dependent |
| `numpy_version`  | `numpy.__version__`            | Machine-dependent |
| `qec_version`    | `src.__version__`              | From code      |
| `git_commit`     | `git rev-parse HEAD` (if available) | From repo |
| `tool_versions`  | Dict of optional tool versions | From installed packages |

When `deterministic_metadata=True`, machine-dependent fields are replaced
with fixed placeholder values to enable byte-identical comparison.

## Verification

To verify reproducibility:

```bash
# Run twice with same config
python -m src.bench --config config.json --output run1.json
python -m src.bench --config config.json --output run2.json

# Compare artifact hashes
python -c "
import hashlib, json
for f in ['run1.json', 'run2.json']:
    data = open(f).read()
    print(f'{f}: {hashlib.sha256(data.encode()).hexdigest()}')
"
```

Both hashes must be identical when `runtime_mode="off"` and
`deterministic_metadata=True`.
