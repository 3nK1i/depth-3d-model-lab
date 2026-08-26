# I4.1.5 — Decoupled Global/Local Sampling Counterfactual

## Purpose

I4.1.4 established a mixed failure: the Horse capture regime is difficult **and** local temporal density materially affects dense-consistency success. I4.1.5 therefore evaluates an additive sampling architecture without changing production behavior.

The counterfactual separates three responsibilities:

1. **Global SfM anchors** — bounded full-span samples used by the single authoritative sparse reconstruction.
2. **Localized dense candidates** — extra source observations sampled at higher local density and, in a later production implementation, explicitly localized into the existing SfM frame. They are not a second reconstruction/map.
3. **Final fusion views** — the existing bounded final qualified set; the fusion cap remains unchanged.

## Candidate plans

- `global_anchors_only`: current one-set baseline.
- `global_anchors_plus_local_neighbors`: global anchors plus bounded ±local temporal neighbors.
- `uniform_localized_dense_pool`: separate full-span local-density candidate pool, bounded independently from global anchors.

I4.1.5 estimates pair viability using the empirical temporal pass-rate buckets produced by I4.1.4. Unsupported buckets remain unscored. These estimates are **not** DA3 or dense-pair results and cannot authorize reconstruction quality by themselves.

## Guardrails

- One authoritative PyCOLMAP/SfM coordinate frame.
- I4.1.3 full-span global sampling remains intact.
- Extra dense candidates require explicit pose localization into the existing SfM model before any future production use.
- No second camera trajectory or independent geometry authority.
- No dense-consistency threshold changes.
- No F/G filtering or persistence changes.
- Final fusion cap remains bounded.
- No metric, measurement, mesh, or I4.2 authority.

## Decision

A positive counterfactual authorizes only **I4.1.6 implementation work**, not production promotion. I4.1.6 must add the mechanism behind compatibility boundaries, re-run regression suites, exercise real GPU/runtime behavior, and preserve clean short-video behavior. The Horse sequence remains an I5 robustness case; a cleaner static object/connector orbit remains required for final I4 qualification when capture-regime evidence is strong.
