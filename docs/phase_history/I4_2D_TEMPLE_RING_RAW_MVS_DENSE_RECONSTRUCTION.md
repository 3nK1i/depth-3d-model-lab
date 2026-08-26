# I4.2D — TempleRing Raw MVS Dense Reconstruction

## Scope

I4.2D runs the first dense COLMAP MVS reconstruction on Middlebury TempleRing
from the unchanged I4.2B authoritative PyCOLMAP sparse model, and only after
I4.2C held-out trajectory validation has passed.

This stage exists to create the **raw dense evidence** needed for I4.2E thin-
structure/detail-preservation qualification.  It does not yet test or modify
the production `d12_sparse` manifold cleanup.

## Inputs and evidence locks

I4.2D requires:

- I4.2B `PASS` with 47/47 registered views;
- the fixed shared TempleRing `PINHOLE` intrinsics preserved;
- I4.2C `PASS` with accepted held-out trajectory quality;
- exact checksum agreement between I4.2B and I4.2C;
- unchanged authoritative `cameras.bin`, `images.bin`, `points3D.bin`,
  `frames.bin`, and `rigs.bin`;
- unchanged 47 staged canonical images `templeR0001.png` ...
  `templeR0047.png`.

The I4.2D MVS worker receives no Middlebury ground-truth R/t, camera centres,
or angle file.

## Dense execution

The bounded default configuration is intentionally analogous to the already-
qualified House A2 MVS positive control while preserving all 47 TempleRing
reference views:

- 47/47 reference views available;
- automatic geometry-driven source selection;
- at most 10 source neighbours per reference;
- native image size (`max_image_size=-1`);
- geometric consistency ON;
- PatchMatch cache: 2 GiB;
- Stereo Fusion cache: 2 GiB;
- GPU 0 by default;
- no permanent global view cap;
- no quality-threshold relaxation.

The worker performs only:

1. COLMAP image undistortion/workspace preparation;
2. PatchMatch Stereo;
3. geometric Stereo Fusion to `fused.ply`.

A PASS requires all 47 geometric depth maps, all 47 geometric normal maps, a
non-empty fused PLY, preserved dense-workspace sparse poses/points, and
unchanged authoritative input hashes.

## Authority locks

I4.2D never:

- reruns SfM or bundle adjustment;
- refines or rewrites authoritative camera poses;
- consumes held-out benchmark poses;
- applies the production manifold filter;
- changes production filtering policy;
- starts DA3;
- promotes a dense provider or authorizes I4.3;
- claims metric scale or measurement authority;
- generates a mesh or certified-metrology surface.

The similarity scale observed in I4.2C remains validation evidence only and is
not injected into MVS.

## Next

A passing I4.2D authorizes **I4.2E — Thin-Structure & Detail-Preservation
Qualification**.  I4.2E will inspect the raw TempleRing MVS evidence and make a
controlled comparison against the existing production `d12_sparse` manifold
cleanup before any production-policy decision is allowed.
