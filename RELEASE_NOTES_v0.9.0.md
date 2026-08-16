# EIA TECH Depth & 3D Model Lab — v0.9.0 Release Notes

> **Release status:** the finalized source identity is `0.9.0` / `Phase 9 Digital Twin Stable`. The annotated `v0.9.0` tag is created only after the metadata transition is green, committed, pushed and synchronized.

## Highlights

- Unified Digital Twin workspace built on the qualified Phase 8 Reconstruction foundation.
- Automatic calibration and metric-qualification evidence paths with explicit authority boundaries; intrinsics alone never establish physical scale.
- Semantic workspace/project persistence, Auto Segment and prompt-refinement workflows, deterministic review/merge behavior and semantic measurement context.
- Canonical Digital Twin visualization with cameras, semantic objects, inspection overlays and canonical picking.
- Estimated metric ruler results with explicit `±` uncertainty and preserved provenance when only estimated scale evidence is available.
- Recorded-video full-scene Depth Anything 3 supporting geometry with bounded overlapping-window consolidation and persistent canonicalization.
- Persistent Gaussian splatting and Nerfacto/NeRF appearance workflows, including static preview and interactive presentation.
- Unified navigation/interaction state, progressive-disclosure Digital Twin UX, save/restart/restore and state export.
- Application-wide heavy-runtime ownership and reciprocal handoff between Reconstruction, DA3, Gaussian and NeRF workflows.
- Final real-scene Phase 9.6H qualification plus complete unit/UI/integration/stress regression before the release transition.

## Authority and measurement posture

The release preserves explicit authority boundaries:

- **Canonical geometry** remains spatial, picking and measurement geometry authority.
- **Semantic geometry** supplies contextual/object information and never replaces canonical geometry.
- **Inspection/QC** remains separate inspection authority.
- **Depth Anything 3** may supply supporting relative geometry/cameras; it does not invent physical scale.
- **Gaussian and NeRF/Nerfacto are appearance only.** They are not canonical, metric, measurement, QC or reference authority.
- An estimated ruler/reference may expose a best available metric estimate with explicit uncertainty while preserving its original provenance.
- `certified_metrology=false` remains the correct release posture unless a separately qualified calibration/reference path establishes otherwise.
- Camera intrinsics alone do not establish metric scale.

## Persistence and resource isolation

Digital Twin save/restart/restore preserves canonical scene identity, camera/appearance metadata, estimated-scale provenance and UI state without auto-resuming heavyweight work. State export is intentionally reference-based rather than a portable payload bundle when heavy artifacts remain external.

Heavy runtimes remain isolated from the production `depth` interpreter. The qualified architecture keeps DA3, gsplat and Nerfstudio/Nerfacto behind their external runtime boundaries, with one application-wide ownership model preventing conflicting heavy execution.

## Qualification evidence

Phase 9.6H closed the feature work under a final release gate using a real 2–3 minute recorded-video scene. The campaign exercised fresh full-scene DA3 geometry, a persistent canonical scene/camera trajectory, fresh NeRF appearance, fresh Gaussian appearance, ruler uncertainty presentation, save/restart/restore, state export, interactive/static appearance behavior and resource cleanup.

The final H release-gate summary reported `pass=true` with no blockers. The complete automated current-tree regression then passed:

- 2279 unit tests
- 566 UI tests
- 241 integration tests, with 2 explicitly expected optional target-hardware skips
- 87 stress tests

The metadata transition does not rerun the expensive real DA3/Gaussian/NeRF campaigns because it changes release identity, release-facing documentation and current release assertions only.

## Deliberate limitations

- The release does not certify photorealistic convergence from bounded Gaussian/NeRF qualification profiles.
- The release does not certify global loop closure for the Phase 9.6H DA3 scene.
- Metric scale is not authoritative unless qualified reference/calibration evidence is actually present for the scene.
- The release is not certified metrology.
- Neural appearance never supplies QC, measurement or canonical-geometry authority.
- External heavy backends retain their own hardware/software/runtime requirements.

## Release procedure

Phase 9.6H closes functional qualification without changing version metadata or creating a tag. The final release transition changes the central identity to `0.9.0` / `Phase 9 Digital Twin Stable`, qualifies the affected/current test surface, commits and pushes that exact state, and only then creates and pushes the annotated `v0.9.0` tag.
