# Phase C — Single-Frame Canonical Dense RGB Projection

Phase C validates the first dense colored keyframe in the global offline-SfM reconstruction frame.

## Scope

- Reuse accepted B1/B2 sparse-correspondence depth alignment.
- Convert supported SfM camera models into the existing `ReconstructionCamera` contract without claiming calibrated intrinsics.
- Back-project aligned dense depth with RGB through the SfM intrinsics.
- Transform camera-space XYZRGB with `T_reconstruction_from_camera`.
- Produce a validation NPZ/PLY in the arbitrary SfM reconstruction gauge.
- Preview with the existing Live-3D `PointCloudCanvas`.

## Explicit non-goals

Phase C does **not** perform multi-view accumulation/fusion, persistence, Reconstruction-page integration, physical metric-scale qualification, measurement authority, mesh reconstruction, Gaussian/NeRF work, or certified metrology.

## Validation

```bash
bash scripts/run_c_tests.sh quick
bash scripts/run_c_tests.sh affected

rm -rf /tmp/eia_c_single_frame_projection
python scripts/run_c_single_frame_canonical_projection.py \
  --b2-dir /tmp/eia_b2_real_alignment \
  --output-dir /tmp/eia_c_single_frame_projection

python scripts/run_c_single_frame_canonical_preview.py \
  --cloud /tmp/eia_c_single_frame_projection/c_canonical_single_frame_cloud.npz \
  --output /tmp/eia_c_single_frame_projection/c_canonical_preview.png \
  --overlay-sparse \
  --show
```

The real acceptance gate is a coherent single-frame RGB cloud with the correct camera orientation/position in the SfM reconstruction gauge. The scale remains explicitly arbitrary and the output remains validation-only until later persistence/authority milestones.
