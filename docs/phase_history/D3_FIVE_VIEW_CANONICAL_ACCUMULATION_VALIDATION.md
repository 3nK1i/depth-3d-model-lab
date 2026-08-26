# D3 — Controlled Five-View Canonical Accumulation

## Purpose

D3 expands the validated three-view canonical dense-RGB foundation to exactly five registered views while deliberately keeping the geometry raw and unfused.

The step reuses the existing PyCOLMAP reconstruction world, isolated DA3 runtime, B1 SfM↔dense-depth alignment, and Phase C canonical XYZRGB projection. It does not create a new reconstruction backend.

## Selection policy

Automatic D3 selection is overlap-aware. It evaluates contiguous five-view windows in registered workspace order using:

- shared sparse SfM landmark support between every adjacent pair;
- shared-landmark support normalized by the weaker keyframe's sparse observation count;
- adjacent camera baseline relative to a robust arbitrary-gauge scene scale;
- total sparse observation support.

A previously qualified three-view chain may be supplied with `--qualified-keyframe-ids`; only five-view windows containing that exact consecutive chain are then eligible. This is the preferred continuation from D2.2.

## Geometry behavior

For every selected view D3 performs:

1. real isolated DA3 dense depth;
2. safe depth-grid adaptation to the SfM pixel grid;
3. robust scale-only alignment to the arbitrary SfM gauge;
4. canonical XYZRGB projection using the PyCOLMAP camera model and optimized pose;
5. raw concatenation of all five canonical clouds.

No spatial merge, voxel fusion, multi-view filtering, or persistence occurs in D3.

## Authority boundary

D3 remains validation-only:

- SfM scale stays arbitrary;
- metric scale is not claimed;
- canonical geometry authority is not promoted;
- measurement authority is not promoted;
- certified metrology is not claimed;
- Gaussian/NeRF authority is unchanged.

## Acceptance

D3 is technically green only when all five real views align and project successfully and the source-colored preview shows one plausible shared world rather than detached view clouds. Quantitative five-view dense overlap qualification remains separate from fusion and must be reviewed before advancing to wider accumulation.
