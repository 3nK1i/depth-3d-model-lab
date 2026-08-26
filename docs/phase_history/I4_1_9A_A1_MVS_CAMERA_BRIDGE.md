# I4.1.9A A1 — Authoritative House MVS Camera Bridge

## Scope

A1 freezes and validates the already-qualified I4.1.7 House camera solution for a later classical MVS positive control.

It does **not** run SfM, bundle adjustment, pose refinement, image undistortion, PatchMatch Stereo, Stereo Fusion, mesh generation, or metric-scale promotion.

## Authority lock

- 40 views remain the original authoritative PyCOLMAP SfM anchors.
- 28 views remain accepted localized extra observations in the same SfM frame.
- All 68 House views are registered in the derived MVS sparse model.
- The 40-view historical cap is not reused as an MVS view-selection rule.
- Existing PyCOLMAP-estimated camera parameters are reused.
- `House_K.txt` is not consumed by A1.
- SfM gauge stays arbitrary.
- `metric_scale_claimed=False`.
- `measurement_authoritative=False`.
- `certified_metrology=False`.
- `surface_mesh_generated=False`.

## Bridge method

1. Require the original I4.1.7 workspace, summary, orbit diagnostics, authoritative `real_sfm_report.json`, native selected sparse model, 28 localization evidence files, and 28 materialized candidate frames.
2. Cross-check the summary and NPZ selected-view ordering/provenance and every camera pose.
3. Use the isolated PyCOLMAP runtime only to load the native sparse model and export it to COLMAP text.
4. Preserve native `cameras.txt`, all 40 native image records, all native 3D point attributes, and all native track observations.
5. Append the 28 already-localized image poses and only their accepted existing-landmark observations. No new 3D points are created.
6. Stage byte-identical image copies in the derived MVS workspace.
7. Reload both native and derived sparse models in the isolated PyCOLMAP runtime and verify camera intrinsics, native poses, point positions, track preservation, extra poses/observations, and exact 40+28=68 registration counts.

## Expected output

`/tmp/eia_i4_1_9a_house_mvs_positive_control/a1_camera_bridge`

contains:

- `images/` — all 68 MVS input images,
- `sparse/` — derived 68-view COLMAP sparse text model,
- `native_sparse_text/` — read-only text export of the native 40-view model,
- `evidence/` — frozen authority evidence and fingerprints,
- `worker/` — external PyCOLMAP requests/results,
- `i4_1_9a_a1_summary.json`.

A1 PASS authorizes only the next bounded MVS execution step. It does not promote the MVS result to production authority.
