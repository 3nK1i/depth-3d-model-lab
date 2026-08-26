# I4.2E.2 — Native Dense-Visibility Provenance Correction

## Trigger

The I4.2E.1 worker-isolation correction removed the accidental OpenCV import,
but the real TempleRing support replay then failed after a successful 156,168
point Stereo-Fusion replay because the worker required every point in the
returned `pycolmap.Reconstruction` to have a non-empty sparse-style
`Point3D.track`.

That assumption was incorrect for PyCOLMAP/COLMAP dense fusion. In COLMAP 4.1.1,
`RunStereoFuserImpl` imports the fused PLY points into a `Reconstruction`, which
does not preserve the dense support vectors as sparse SfM tracks. For PLY output,
COLMAP writes the native dense support separately as `fused.ply.vis` from
`StereoFusion::GetFusedPointsVisibility()` in the same fused-point ordering.

## Correction

I4.2E now consumes the already-existing I4.2D pair directly:

- `dense/fused.ply`
- `dense/fused.ply.vis`

No Stereo-Fusion replay is required. The dependency-light support worker parses
the native COLMAP visibility binary format:

1. little-endian `uint64` fused-point count;
2. for each fused point, little-endian `uint32` support count;
3. that many little-endian `uint32` MVS image indices.

The image values are native zero-based MVS image **indices**, not COLMAP image
IDs. I4.2E uses them only as stable per-view identities for support aggregation.
No pose semantics are inferred from the index values.

## Fail-closed validation

The correction requires:

- I4.2D PASS and its unchanged raw `fused.ply`;
- the sibling `fused.ply.vis` sidecar to exist;
- I4.2D producer PyCOLMAP version exactly `4.1.1`;
- visibility point count exactly equal to the frozen I4.2D PLY vertex count;
- non-empty support for every fused point;
- no duplicate image index within a fused point;
- every image index in `[0, 46]`;
- every observed MVS image index within the campaign range `[0, 46]`;
- no truncated or trailing visibility bytes.

Full global coverage of all 47 MVS indices is **not** a validity requirement:
`fused.ply.vis` records support of surviving fused points, so a reference image
may legitimately contribute no surviving support observation. The observed
index set is persisted as a diagnostic only. Any actual structural/range
violation aborts I4.2E before the d12 counterfactual is evaluated.

## Why this is stronger than the replay design

The native sidecar was produced by the same I4.2D Stereo-Fusion invocation that
produced the frozen raw cloud. Consuming it avoids a second heavy MVS operation,
removes replay determinism as a confounder, and uses COLMAP's actual dense
provenance representation rather than reconstructing support from an unrelated
sparse-track abstraction.

## Dependency boundary

The support worker now requires only Python standard library + NumPy. It imports
neither:

- `core.reconstruction`;
- OpenCV / `cv2`;
- PyCOLMAP.

It may still be launched with the isolated PyCOLMAP interpreter so the existing
process boundary is preserved without broadening the environment.

## Authority locks

I4.2E.2 does not rerun or alter:

- SfM;
- bundle adjustment;
- undistortion;
- PatchMatch;
- Stereo Fusion;
- poses or intrinsics;
- production `d12_sparse` policy.

The visibility evidence is diagnostic only. It does not establish metric scale,
authoritative measurement, certified metrology, mesh authority, provider
promotion, or I4.3 authorization.

## Next

After automated validation, run I4.2E again in a fresh immutable output directory
(`attempt03`). A successful execution must reach `READY_FOR_MANUAL_REVIEW`; only
then generate the matched pre/post-d12 preview and complete the I4.2E visual
verdict.
