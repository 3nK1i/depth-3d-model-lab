# I4.2E.1 — PyCOLMAP Worker Dependency-Isolation Correction

## Trigger

The I4.2E focused/stress/full test matrix passed in the main `depth` environment,
but the real support-replay worker failed under the isolated PyCOLMAP interpreter
with `ModuleNotFoundError: No module named 'cv2'`.

The failure was not an MVS or TempleRing failure. The worker imported
`core.reconstruction.mvs_visual_ab` and
`core.reconstruction.temple_ring_detail_qualification` only for a PLY reader and
cloud digest. Importing `core.reconstruction` initialized the application
reconstruction package and transitively imported `dense_fusion.py`, which depends
on OpenCV. OpenCV is intentionally not installed in the isolated PyCOLMAP env.

## Correction

The support worker now owns the two tiny dependency-light helpers it needs:

- COLMAP fused PLY XYZ/RGB reader;
- order-independent float32 XYZ + uint8 RGB cloud digest.

Their semantics match the existing I4.2E/application helpers, but the worker no
longer imports any `core.reconstruction` module and does not require OpenCV.
PyCOLMAP remains imported lazily inside execution, preserving the isolated runtime
boundary.

## Locked behavior preserved

This correction does not rerun PatchMatch, SfM or BA, does not refine poses, does
not consume held-out ground truth, does not change or apply production policy to
the authoritative scene, and does not authorize I4.3, metric scale, measurement,
metrology or mesh authority.

## Regression guard

I4.2E tests now include unit/integration/stress coverage for the worker-local PLY
and digest helpers plus an import guard. The shell runner also probes the worker
with the isolated PyCOLMAP interpreter (when present) while explicitly rejecting
any top-level import of `core.*` or `cv2`.

The failed `attempt01` remains immutable evidence. Real I4.2E execution must use a
fresh output directory such as `i4_2e_detail_qualification_attempt02`.

## Superseded provenance assumption

I4.2E.1 correctly fixed the dependency leak, but it retained the original
assumption that the `Reconstruction` returned by Stereo Fusion exposed dense
support through sparse-style `Point3D.track` elements. The real TempleRing run
proved that assumption false. I4.2E.2 supersedes only that provenance mechanism
by consuming the native `fused.ply.vis` sidecar written by I4.2D. The dependency
isolation introduced here remains in force.
