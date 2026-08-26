# I4.2A — TempleRing Dataset / Calibration Quarantine Preflight

## Purpose

Start I4.2 without changing reconstruction or filtering behavior. This step validates and freezes the exact **Middlebury TempleRing** benchmark inputs before any new SfM/MVS execution.

I4.2 is the thin-structure/detail-preservation milestone. It must not assume that the room-scale `d12_sparse` manifold cleanup is safe for object-scale details such as thin vertical members, gaps/openings, edges, slots, or small protrusions.

## Dataset contract

I4.2A requires:

- exactly 47 `templeR####.png` images;
- 640×480 PNG inputs;
- exactly 47 rows in `templeR_par.txt`;
- one shared finite/invertible intrinsic matrix `K`;
- proper rigid per-view `R` and finite `t`;
- no cross-dataset `Temple` / `TempleSparseRing` PNGs mixed into the directory.

`templeR_ang.txt` is optional validation-only evidence.

## Calibration quarantine

I4.2A creates **two separate manifests**:

1. `i4_2a_reconstruction_input_manifest.json`
   - 47 image paths + SHA256;
   - shared benchmark-provided `K`;
   - **no** per-view R/t, camera centres, or gantry angles.

2. `i4_2a_ground_truth_validation_manifest.json`
   - per-view ground-truth R/t and `C = -R^T t`;
   - optional angle-file fingerprint;
   - validation-only; forbidden as SfM pose input or BA prior.

The later bounded I4.2 SfM experiment may use the shared `K` as known intrinsics, but ground-truth extrinsics remain held out until post-SfM validation.

## Detail-preservation contract

Later I4.2 stages must explicitly review:

- thin vertical member separation;
- small gaps/openings;
- edge definition;
- small protrusions;
- surface thickness / false bridging;
- local completeness vs noise;
- any legitimate-detail loss caused by the production `d12_sparse` manifold policy.

No threshold change is authorized by this preflight.

## Authority boundaries

I4.2A performs no SfM, BA, DA3, MVS, fusion, manifold filtering, metric-scale inference, measurement qualification, mesh generation, or provider promotion. I4.3 remains unauthorized.

## Next

`I4.2B` — bounded TempleRing SfM using all 47 images and the provided shared K, with per-view ground-truth R/t held out for independent trajectory/camera validation.
