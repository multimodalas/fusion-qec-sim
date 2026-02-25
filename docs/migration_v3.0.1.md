Migration Notes — v3.0.1

Overview

v3.0.1 ("Legacy Compatibility & High-Dimensional Readiness") is a strictly
additive release.  No action is required to migrate from v3.0.0.

All existing v2.x and v3.0 public APIs remain stable.  No function signatures
have changed.  No default behaviour has been altered.

What Changed

1. Schema version updated from "3.0.0" to "3.0.1".
   Both versions are accepted by the validator.

2. Two optional config fields were added (ignored when absent):
   - "qudit": optional dimension specification (default: qubit, dimension=2).
   - "resource_model": optional analytical gate-cost estimation config.

3. New isolated modules were added (never imported by core decoding):
   - src/qudit/ — QuditSpec dimension specification layer.
   - src/analysis/ — Analytical gate-cost estimation helpers.
   - src/nonbinary/ — Scaffolding interfaces for future nonbinary work.

4. When resource_model.enabled is true, a "resource_estimates" key
   appears in the summaries section of benchmark output.

Backward Compatibility

- v3.0.0 configs run unchanged (new fields default to absent/None).
- v3.0.0 result objects still pass schema validation.
- runtime_mode="off" still produces byte-identical JSON.
- No decoder behaviour or output has changed.
- No new dependencies (stdlib + numpy only).

Required Action

None.
