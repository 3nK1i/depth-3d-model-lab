# Phase 8.6C Validation — Dense Multi-View Reconstruction

## Scope

Phase 8.6C adds a bounded dense multi-view fusion kernel that combines aligned depth keyframes with `T_reconstruction_from_camera` poses and Phase 8.6A camera geometry. The output is a deterministic voxel-fused point cloud in reconstruction coordinates.

This step does **not** add mesh generation, neural rendering, semantic reconstruction, world alignment, certified metrology, or Reconstruction UI enablement.

## Architecture

Three concerns remain explicit and independent:

1. **Camera geometry** — estimated or calibrated intrinsics from Phase 8.6A.
2. **Reconstruction gauge** — arbitrary/provisional/calibrated metric scale from Phase 8.6B.
3. **Depth-to-pose-gauge alignment** — depth must be mapped into reconstruction units before fusion.

Supported depth alignment paths:

- depth already expressed in reconstruction units;
- metric depth converted through an explicit metric `ScaleReferenceProfile`;
- explicit depth scale factor with provenance and uncertainty;
- robust sparse-landmark/depth correspondence scale estimation.

Metric depth is never silently fused into arbitrary-scale SLAM coordinates.

## Product profiles

- `OBJECT_INSPECTION`: high-local-detail reconstruction intent.
- `SPATIAL_DIGITAL_TWIN`: larger persistent scene reconstruction intent.

The product profile is separate from GPU quality/performance profiles. Voxel size remains explicit in reconstruction units; Phase 8.6C does not invent a physical voxel size when scale is arbitrary.

## Fusion behavior

- Back-project depth with the active camera profile.
- Apply lens undistortion for supported OpenCV/Brown and fisheye camera models.
- Transform points with `T_reconstruction_from_camera`.
- Fuse per-frame voxel observations using confidence-weighted position/color accumulation.
- Count **distinct frame observations per voxel**, not repeated pixels from one frame.
- Apply a hard `max_voxels` bound and report dropped new voxels instead of growing memory without limit.
- Preserve source pose-layer provenance (`tracking`, `optimized_keyframe`, `optimized_propagated`).

## Safety invariants

- Arbitrary reconstruction remains `unit_to_metre=None`.
- Estimated camera + metric scale remains `provisional_metric`.
- Metric point export is rejected for arbitrary scale.
- `certified_metrology=False` always in Phase 8.6C.
- Pose convention remains `T_reconstruction_from_camera`.
- World alignment remains whatever Phase 8.6B declared; dense fusion does not create a world frame.
- PyCOLMAP and stella_vslam remain externally isolated.
- Reconstruction UI remains disabled.

## Focused validation

```bash
./scripts/run_phase8_6c_tests.sh quick
./scripts/run_phase8_6c_tests.sh integration
./scripts/run_phase8_6c_tests.sh stress
python scripts/run_dense_multiview_fusion_smoke.py
```

Expected focused counts on the package baseline:

- quick: 19 passed
- integration: 1 passed
- stress: 1 passed

The smoke must end with:

```text
Phase 8.6C dense multi-view fusion smoke: PASS
```

## Backward compatibility

Run Phase 8.6B and all earlier established quick gates, then run:

```bash
./scripts/run_phase8_6c_tests.sh full
python main.py
```

The same two optional real-perception integration skips are acceptable. Reconstruction mode must remain disabled.
