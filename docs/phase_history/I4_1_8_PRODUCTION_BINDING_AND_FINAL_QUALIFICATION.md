# I4.1.8 — Production Binding + Retained-Core Final I4.1 Qualification

## Purpose

I4.1.8 consumes the qualified I4.1.7B clean static-orbit evidence and binds the validated `global_anchors_plus_local_neighbors` plan into a bounded production reconstruction path. The already-qualified path is deterministically capped to the preserved final fusion cap; no new target is searched for and no dense/orbit threshold is relaxed.

The bound scene then runs the existing production kernels:

1. bounded Dense RGB voxel fusion;
2. formal F confidence / consistency / artifact filtering;
3. G self-contained canonical Dense RGB persistence;
4. ordinary I2 automated real-scene quality qualification;
5. retained-Core robust object-target recomputation;
6. final locked I4.1 orbit requalification on the actual bound production camera subset.

## Heavy-runtime boundary

I4.1.8 reuses the completed I4.1.7 evidence. It does **not** rerun PyCOLMAP, DA3, localization or dense-pair A/B. All selected observations remain in the single authoritative global SfM frame.

## Production binding policy

- source plan must be explicitly authorized by I4.1.7B;
- source selected path must be strictly monotonic and contain at least 10 views;
- the exact final fusion cap recorded by I4.1.6B is preserved (40 in the qualifying box campaign);
- downselection is deterministic uniform sampling over the already-qualified path and preserves both path endpoints;
- production default behavior is not silently changed elsewhere in the application.

## Final qualification

The provisional I4.1.7B trajectory-circle target is **not** final authority. After F, I4.1.8 computes the final target from retained Core geometry and re-applies the unchanged I4.1 gates:

- registered SfM fraction >= 80%;
- bound dense alignment fraction >= 80%;
- selected production views >= 10;
- selected azimuth coverage >= 150 degrees;
- selected occupied sectors >= 5/12;
- opposed-view angle >= 120 degrees;
- target-facing fraction >= 75% at <= 60 degrees;
- standoff p90/p10 <= 2.5;
- ordinary bound-scene I2 automated quality PASS.

If retained Core geometry causes any final gate to fail, I4.1.8 remains DEGRADED. The provisional target is never carried forward to make the final result pass.

## Authority boundaries

Always preserved:

- arbitrary monocular SfM gauge;
- `metric_scale_claimed=False`;
- `measurement_authoritative=False`;
- `certified_metrology=False`;
- `surface_mesh_generated=False`;
- no dense-consistency or orbit threshold relaxation.

A clean automated result becomes `READY_FOR_MANUAL_ORBIT_REVIEW`. I4.2 remains unauthorized until the bound Dense RGB scene, retained-Core target and production camera orbit pass the final manual review.
