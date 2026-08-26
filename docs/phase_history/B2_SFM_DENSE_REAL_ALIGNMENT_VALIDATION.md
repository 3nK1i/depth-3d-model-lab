# B2 Validation — Real Single-Keyframe SfM ↔ Dense-Depth Alignment

B2 validates the B1 scale-only alignment bridge with one real offline PyCOLMAP keyframe and one real Depth Anything 3 dense-depth result.

## Scope

- Run real offline PyCOLMAP SfM through the isolated `eia-pycolmap-probe` interpreter.
- Select the registered keyframe with the strongest sparse landmark support, unless a specific registered keyframe is requested.
- Run real DA3 dense-depth inference for that keyframe through the isolated `eia-da3` interpreter.
- If DA3 returns its bounded inference grid instead of the original keyframe grid, allow only a near-uniform validation-side resize back to the exact SfM image dimensions. Large width/height scale disagreement is rejected rather than assuming crop/padding.
- Execute the existing B1 `align_sfm_keyframe_dense_depth()` bridge without introducing a second scale estimator.
- Persist numerical evidence plus an RGB/depth diagnostic image with sparse observation coverage.

## Authority boundary

B2 remains arbitrary-scale. The fitted factor maps DA3 depth source units into the PyCOLMAP reconstruction gauge only. It does not establish metres, certified calibration, canonical geometry authority, measurement authority, QC authority, or certified metrology.

The DA3-to-SfM pixel-grid resize is explicitly a B2 validation adapter. It does not establish any geometry or scale authority and is not a substitute for a later production pixel-space/intrinsics bridge.

## Pass gate

B2 passes only when the real B1 evidence is `accepted`. The summary records candidate observations, valid correspondences, robust inliers, fitted scale, relative uncertainty, median relative residual, P90 relative residual, DA3/SfM image sizes, and the exact pixel-space adaptation mode.

A rejected alignment is still a useful B2 result: the script writes the evidence and diagnostic before returning a non-zero status so the failure can be inspected instead of hidden.

## Deferred

B2 does not project RGB-D points into the canonical world, accumulate multiple views, fuse voxels, persist dense canonical geometry, change the Reconstruction UI, build a surface mesh, or touch Live Scan. Those remain later roadmap steps.
