# I4.2B — Authoritative TempleRing SfM

## Scope

I4.2B creates one PyCOLMAP-authoritative sparse reconstruction for the canonical
47-view Middlebury TempleRing benchmark.  It consumes only the I4.2A
reconstruction-input manifest: all 47 images plus the shared benchmark K.
Per-view benchmark R/t, camera centres, gantry angles, laser ground truth and
visual-hull evidence remain held out.

## Locked reconstruction policy

- Canonical images: `templeR0001.png` ... `templeR0047.png`.
- Native image size: 640x480; no downscale.
- Shared camera model: `PINHOLE` with `fx, fy, cx, cy` from I4.2A.
- Intrinsics stay fixed during absolute-pose refinement and bundle adjustment.
- COLMAP SIFT feature extraction, exhaustive matching, incremental mapping + BA.
- One mapping model; no permanent global view cap.
- PyCOLMAP is the sole authoritative frame; scale remains arbitrary.
- No MVS, DA3, production-manifold filtering, mesh generation, measurement
  authority, metrology claim, or I4.3 provider promotion in this step.

## Qualification

A technical I4.2B PASS requires:

1. the exact frozen I4.2A reconstruction manifest and all 47 image hashes;
2. a qualified PyCOLMAP 4.1.1 runtime;
3. one shared PINHOLE camera whose parameters remain equal to the provided K;
4. all 47 canonical views registered into one authoritative sparse frame;
5. the existing generic SfM quality policy to classify the sparse result as
   `accepted`;
6. a complete native COLMAP binary sparse model.

A lower registration count or degraded sparse quality is reported as DEGRADED,
not silently promoted and not automatically retuned.

## Next

Only an I4.2B PASS authorizes I4.2C: held-out trajectory validation against the
benchmark R/t after reconstruction.  Ground-truth poses are not reconstruction
inputs and I4.3 remains unauthorized.
