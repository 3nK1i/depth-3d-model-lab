# I4.1.9A A3 — Controlled DA3 vs COLMAP MVS Visual A/B

## Scope

A3 is a diagnostic visual-comparison stage only. It reuses the existing I4.1.9 DA3 Dense RGB geometry and the successful A2 COLMAP Stereo Fusion PLY. It does not execute SfM, BA, DA3, undistortion, PatchMatch, Stereo Fusion, calibration, metric scaling, measurement qualification, provider promotion, or mesh generation.

## Scientific control

The DA3 I4.1.9 summary and the frozen A1 authority manifest must resolve to the same I4.1.7 House source directory. The current I4.1.7 summary SHA256 must still match the copy frozen by A1. A2 must be PASS, must prove its A1 sparse authority was unchanged, must contain complete 68-view geometric depth/normal maps, and must preserve all authority boundaries.

The primary A/B rendering uses the already-recorded DA3 I2 Core anchor envelope as a shared framing reference only. DA3 and MVS receive identical viewer camera center, distance, FOV, azimuth, and elevation. Independent `fit_scene()` is forbidden because DA3 halo/fan artifacts would otherwise force a different camera distance and visually bias the comparison.

Fixed views: oblique, front, side, top.

## Outputs

A3 writes full-resolution per-side PNGs and a side-by-side A/B PNG for every requested view plus `i4_1_9a_a3_summary.json`. The summary freezes SHA256 for the compared DA3 diagnostics, MVS PLY, A2 summary, A1 authority manifest, and shared I4.1.7 summary.

Descriptive spatial statistics are reported for context only. They are not automated quality gates because DA3 voxel fusion and COLMAP MVS have different density/support semantics.

## Manual decision matrix

- MVS visually clean while DA3 is smeared/fanned: isolate the primary failure to the learned-depth/alignment/fusion branch and treat MVS as a strong static-orbit provider candidate.
- MVS shows similar fan/smearing: investigate camera/intrinsics/projection/frame-transform assumptions before further dense-threshold tuning.
- MVS is much better but imperfect: characterize residual MVS errors and promote only according to evidence.

A3 never selects a winner automatically.
