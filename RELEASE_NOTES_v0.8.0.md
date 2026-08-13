# EIA TECH Depth & 3D Model Lab — v0.8.0 Release Notes

> **Release-preparation status:** these notes describe the target Phase 8 Reconstruction release candidate. During 8.9H, application/package metadata remains `0.7.5` / `Phase 7.5 Stable` and the `v0.8.0` tag must remain absent. The coordinated version transition and annotated tag belong only to the final 8.9I release gate.

## Highlights

- Dedicated Reconstruction workspace integrated with the existing application modes.
- Live SLAM for continuous Video, USB camera and RTSP sources through an isolated native stella worker.
- Offline SfM preparation and isolated PyCOLMAP reconstruction with quality inspection and resume-oriented workspace contracts.
- Camera geometry, scale/reference abstractions, dense multi-view fusion, geometry quality/uncertainty and surface-mesh reconstruction.
- Persistent canonical scene geometry with semantic-object context, measurement presentation, validity state, inspection/QC and reference comparison.
- Isolated Depth Anything 3 support for reconstruction depth/geometry workflows.
- Gaussian splatting and NeRF/Nerfacto appearance representations with resource-aware switching and fallback behavior.
- Reconstruction session save/reload, checkpoint/resume, persistence/reproducibility and multi-artifact scene handling.
- Authority-aware product export bundles and qualified QC report export.
- GPU production profiling, endurance/resource/recovery qualification and actual application end-to-end acceptance.
- RTSP live-SLAM FPS/shutdown hardening and final responsive/fullscreen UI polish before release preparation.

## Reconstruction authority

The release preserves explicit authority boundaries:

- **Canonical geometry** is the authoritative scene geometry.
- **Semantic geometry** supplies semantic/object context; it does not replace canonical geometry.
- **Measurement evidence** retains explicit uncertainty, validity/rejection state and the existing engineering-estimate posture.
- **Inspection/QC** remains the inspection authority.
- **Gaussian and NeRF/Nerfacto representations are appearance only.** They are not canonical geometry, metric authority, measurement authority, QC authority or reference authority.
- `certified_metrology=false` remains the correct release posture absent a separately qualified calibration/reference process.
- Monocular SLAM/reconstruction may remain in arbitrary scale without a valid metric reference; camera intrinsics alone do not establish metric scale.

## Backend and environment isolation

Heavy/native/neural reconstruction runtimes remain outside the production `depth` environment behind explicit process/provider boundaries. This includes the native live-SLAM worker, PyCOLMAP/SfM tooling, Depth Anything 3, Gaussian backends and Nerfstudio/Nerfacto tooling. Backend availability is therefore capability- and environment-dependent rather than implied by the base PyQt installation.

## Qualification evidence

Phase 8 release preparation is backed by the repository's Phase 8.9 qualification chain:

- real external-backend preflight
- real SfM trajectory evidence plus isolated Depth Anything 3 supporting depth/geometry and canonical-geometry qualification
- real Gaussian appearance qualification
- real Nerfacto appearance qualification
- persistent real multi-artifact scene qualification
- actual application end-to-end acceptance
- complete current-tree regression and final backward-compatibility gate
- RTSP post-fix affected qualification
- combined UI-2/UI-3/UI-4 qualification
- final post-fix whole-application requalification

The final pre-H whole-app requalification passed with no blockers and did not rerun the expensive real SfM/Gaussian/NeRF campaigns because the subsequent RTSP/UI changes did not alter those backend authority contracts.

## Compatibility

The Phase 8 work preserves the established Live 3D, Benchmark, Snapshot, Stable Video and Measure & Inspect workflows. Final backward-compatibility qualification includes the historical Phase 5, Phase 6, Phase 7 and Phase 7.5 release gates plus the Phase 8.8 backward chain and MainWindow startup/switch/shutdown sanity.

## Known limitations and deliberate exclusions

- Measurements remain engineering estimates unless separately calibrated/qualified; the known-size metric-validation campaign is post-`v0.8.0` work.
- Monocular live SLAM and monocular reconstruction do not inherently provide certified metric scale.
- Gaussian and NeRF/Nerfacto artifacts are appearance representations only.
- The release does not claim a fully interactive neural free-view renderer/workspace; the qualified product path uses persisted appearance representations/previews and representation switching.
- Broader camera-calibration/metric qualification, custom Auto Segment/SAM workflows and a unified digital-twin workspace are future work.
- External reconstruction backends have their own hardware/software/runtime requirements and remain isolated from the base application environment.
- The shared Live 3D `Mesh (Poisson)` render-card action remains disabled; Phase 8 mesh processing belongs to the dedicated Reconstruction workflow.

## Release procedure

8.9H prepares and qualifies the documentation/release candidate without creating the release tag. After H3 passes from a clean, synchronized branch, 8.9I performs the coordinated metadata transition to `0.8.0`, runs the final release checks, commits and pushes the release state, creates the annotated `v0.8.0` tag, and pushes that tag.
