# Phase 8.6D Validation — Geometry Quality + Uncertainty

## Scope

Phase 8.6D adds a dependency-light quality layer over the Phase 8.6C dense fused point cloud. It evaluates how trustworthy each observed point is and reports conservative local geometric uncertainty without changing the underlying dense-fusion contract.

This step does **not** add mesh generation, semantics, Gaussian Splatting, NeRF, world alignment, certified metrology, or Reconstruction UI enablement.

## Quality evidence

Always-available signals:

- distinct supporting-view count per fused voxel;
- fused depth confidence;
- source pose-layer provenance (`tracking`, `optimized_keyframe`, `optimized_propagated`);
- estimated vs calibrated camera-geometry provenance;
- input depth coverage and fusion-capacity statistics;
- arbitrary/provisional/calibrated metric semantics and scale uncertainty from Phase 8.6B.

Optional per-point evidence:

- normalized multi-view depth consistency residual;
- supporting-view angular span;
- visibility ratio.

Missing optional evidence stays explicitly unavailable and is treated conservatively. It is never imputed as perfect.

## Product-aware policy

- `OBJECT_INSPECTION` uses stricter support/confidence/consistency thresholds for small-part geometry.
- `SPATIAL_DIGITAL_TWIN` tolerates sparser observations while still requiring explicit quality labels.

Per observed point the output carries:

- confidence score in `[0, 1]`;
- HIGH / MEDIUM / LOW quality code;
- accepted/rejected mask;
- positive local uncertainty radius in reconstruction units.

`UNOBSERVED` is reserved for a future expected-domain/mesh visibility analysis. Phase 8.6D does **not** pretend that a point cloud alone reveals all missing spatial regions. The current `unobserved_input_sample_ratio` refers only to sampled input-depth coverage.

## Uncertainty semantics

- Arbitrary reconstruction: uncertainty is reported only in reconstruction units.
- Metric reconstruction: local uncertainty may also be converted to metres through the active scale profile.
- Relative metric scale uncertainty remains a separate term. It is not silently combined into a fake absolute point-coordinate error.
- `certified_metrology=False` always.

## Safety invariants

- `DENSE_FUSION_SCHEMA_VERSION` remains unchanged.
- `DensePointCloud` and `DenseFusionResult` field contracts remain unchanged.
- `T_reconstruction_from_camera` remains the pose convention.
- Estimated camera geometry remains an explicit quality limiter.
- Arbitrary scale never gains `unit_to_metre`.
- Provisional metric geometry remains provisional.
- PyCOLMAP and stella_vslam remain externally isolated.
- Reconstruction UI remains disabled.

## Focused validation

```bash
./scripts/run_phase8_6d_tests.sh quick
./scripts/run_phase8_6d_tests.sh integration
./scripts/run_phase8_6d_tests.sh stress
python scripts/run_geometry_quality_smoke.py
```

Expected package-baseline counts:

- quick: 25 passed
- integration: 1 passed
- stress: 1 passed

The smoke must end with:

```text
Phase 8.6D geometry quality/uncertainty smoke: PASS
```

## Backward compatibility

Run Phase 8.6C and all earlier established quick gates, then:

```bash
./scripts/run_phase8_6d_tests.sh full
python main.py
```

The same two optional real-perception integration skips are acceptable. Reconstruction mode must remain disabled.
