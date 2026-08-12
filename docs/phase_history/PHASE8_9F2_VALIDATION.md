# Phase 8.9F2 — Real SfM + DA3 + Canonical Geometry Qualification

## Purpose

Release-qualify the geometry side of the real-scene Phase 8.9F campaign without redesigning the reconstruction engine.

F2 executes the already-qualified components against the same South Building scene:

1. real external PyCOLMAP sparse SfM;
2. real isolated Depth Anything 3 (`depth-anything/DA3-SMALL`) inference;
3. the existing DA3 → canonical geometry integration contract.

## Truthful chain boundary

PyCOLMAP and DA3 are separate real evidence paths. The current DA3 provider does **not** accept SfM poses as an input; DA3 predicts its own supporting camera extrinsics. F2 therefore never claims literal SfM-pose → DA3 injection.

The F2 canonical surface is generated from real DA3 evidence through the existing canonical geometry pipeline. SfM is independently qualified against the same scene and is not silently fused into that DA3-derived canonical surface.

## Authority

- SfM: supporting sparse geometry / pose evidence, arbitrary monocular gauge.
- DA3: supporting relative geometry until canonicalized.
- Canonical geometry: authoritative geometry.
- Gaussian: not executed in F2.
- NeRF/Nerfacto: not executed in F2.
- `certified_metrology`: remains `false`.
- No metric scale is invented; `unit_to_metre` must remain absent for this campaign.

## Hardware/runtime envelope

Development measurement target: RTX 2060 6 GB only.

Default real campaign:

- South Building: 128 source images available;
- SfM: efficient profile, sequential matching, max 80 keyframes;
- DA3: 3 contiguous mid-sequence images (overlap window), process resolution 392, DA3-SMALL;
- canonical integration: pixel stride 4.

The DA3 and PyCOLMAP runtimes remain external/isolated. Heavy reconstruction modules must remain non-importable from the production `depth` environment.

## Release qualification thresholds

F2 requires, by default:

- SfM real execution;
- at least 75% keyframe registration;
- at least 1,000 sparse landmarks;
- arbitrary SfM gauge with no metre conversion;
- real DA3 result with expected source revision/model and relative arbitrary scale;
- DA3 selected images belonging to the same South Building dataset;
- successful existing canonical integration;
- non-empty canonical surface mesh;
- canonical geometry authoritative;
- DA3 supporting evidence non-authoritative;
- `certified_metrology=false`;
- no heavy-runtime leak into `depth`.

Historical South Building qualification was substantially stronger (80 registered keyframes and tens of thousands of sparse landmarks); the F2 thresholds are release gates, not quality targets.

## Repository scope

Qualification-only. No production files are changed by F2.

Expected F2 repository additions:

- `docs/phase_history/PHASE8_9F2_VALIDATION.md`
- `scripts/run_phase8_9f2_real_geometry_campaign.py`
- `scripts/run_phase8_9f2_tests.sh`
- `tests/integration/test_phase8_9f2_real_geometry_contract.py`
- `tests/unit/test_phase8_9f2_real_geometry_campaign.py`

Real outputs remain under `/tmp/phase8_9f2_real_geometry` and must never be committed.

## Failure handling

A failed heavy stage is evidence, not permission to change production code. Preserve the output/logs and diagnose first. The initial F2 release harness used the legacy DA3 helper directly against all 128 images, which caused that helper to choose first/middle/last frames via `linspace`; this produced insufficient multi-view overlap for the existing quality-aware mesher on the South Building sequence. F2 therefore materializes a validation-only contiguous mid-sequence overlap window before invoking the unchanged isolated DA3 helper. No geometry-quality, confidence, meshing, authority, or metrology threshold is weakened. `--reuse-completed` may be used after removing only the failed DA3/canonical stage outputs to retain the already-qualified SfM evidence.

Do not clean/reset the deliberate DA3 compatibility patch or runtime lock.
