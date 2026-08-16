# Phase 9.6H — Final Real-Scene Qualification / Release Gate

Baseline: `bcffe314c7e6fe2cb68de55a3ab7c9168c4c90b7`

## Scope

Phase 9.6H is a feature freeze and release qualification step. It adds no Digital Twin runtime, authority source, renderer, reconstruction provider, metric-scale source, measurement algorithm, persistence schema or resource owner.

H binds already-qualified Phase 9.6 evidence into one final functional release gate:

- a readable fixed-focal recorded-video scene between 120 and 180 seconds;
- Phase 9.6E2 full-scene DA3 geometry with 24–96 selected keyframes and the existing overlapping-window contract;
- a verified persistent canonical scene with one persisted camera trajectory entry per selected E2 view;
- Phase 9.6E3 NeRF appearance trained within the qualified 64/128/256-step bounds and persisted as an immutable appearance-only revision;
- Phase 9.6F state export with referenced payload integrity and explicit non-portable/non-authoritative semantics;
- explicit real-UI confirmation of canonical geometry/cameras, ruler estimated metric value with ± uncertainty, restart/restore, NeRF interactive/preview behavior, resource handoff/cleanup and authority wording;
- full current-tree unit, UI, integration and stress regression.

The gate intentionally fails closed if lineage, integrity, authority, duration, camera/view count or explicit manual evidence is missing.

## Release truthfulness

A Phase 9.6H PASS is a functional Digital Twin release gate. It does **not** assert:

- certified photorealistic convergence from the bounded NeRF qualification profile;
- globally optimized/loop-closed DA3 geometry;
- calibrated or authoritative physical metric scale from DA3/intrinsics alone;
- certified metrology;
- QC authority from neural appearance.

Canonical geometry remains spatial/picking/measurement authority. DA3 remains supporting relative geometry. Gaussian and NeRF remain appearance-only. An E0 manual ruler reference may provide an estimated metric value with explicit uncertainty, while remaining non-certified and non-authoritative.

## Real qualification campaign

Use one 2–3 minute fixed-focal scene. A new real phone capture is preferred for release evidence. The previously discussed TUM `freiburg2_large_with_loop` sequence (~173 s) is also suitable as a deterministic functional fallback if converted to a fixed-focal recorded video.

1. Run E2 **Build Full Scene** from the selected recorded video and retain `phase9_6e2_full_scene_summary.json`.
2. From the resulting persistent canonical scene run E3 **Build & Train NeRF** with a qualified profile (Efficient 64 steps is sufficient for the functional gate), retaining `phase9_6e3_nerf_build_summary.json`.
3. Confirm canonical surface + camera trajectory remain the authority. Exercise the ruler and set an approximate physical reference so the UI shows an estimated metric result with explicit ± uncertainty; do not claim calibration/certification.
4. Save the Digital Twin to a non-temporary directory, close the app, restart it and load that saved scene. Confirm canonical geometry/cameras, estimated-scale provenance and appearance metadata restore without auto-starting a heavyweight renderer/job.
5. Run **Export State** and retain `digital_twin_state_export.json`. Referenced heavy payloads must remain declared external (`artifact_payloads_embedded=false`, `portable_payload_bundle=false`, `restore_requires_referenced_files=true`).
6. Exercise Interactive NeRF and then Preview/Hide. Confirm appearance remains non-authoritative, static Preview is non-owning, resource ownership returns idle and no orphan CUDA process remains after application exit.
7. Run `scripts/run_phase9_6h_real_scene_gate.py` with the real evidence plus all manual confirmation flags. The generated summary must report `pass=true`.
8. Run `scripts/run_phase9_6h_tests.sh full`. All test suites must pass; only explicitly expected optional skips are acceptable.

No version bump or Git tag is performed by the H harness. `v0.9.0` requires explicit post-H approval after both the real-scene gate and full regression are reviewed.

## Full-regression compatibility correction

The first H full-regression pass exposed one legacy Phase 8.8G2 text-contract failure: the Phase 9.6F reload tooltip had broadened the wording from `No checkpoint resume is executed` to `No checkpoint or heavyweight build resume is executed`. Runtime behavior was already correct; the older contract checks the exact compatibility wording. H restores the original phrase while retaining the Phase 9.6F heavyweight-build clarification: `No checkpoint resume is executed, and no heavyweight build resume is executed.` No persistence, checkpoint, resume, build, authority or resource behavior changes.

## Full-UI regression contract correction

The second H full-regression pass exposed two stale source-inspection assertions from Phase 9.6C/9.6D. Phase 9.6E4 intentionally moved the live `Navigation` and `Interaction` status rows from the always-visible `workspace_layout` into `workspace_details_layout` behind the qualified `Show details` disclosure. Runtime behavior and the live value widgets are unchanged. H updates only those legacy UI assertions to validate the current E4-compatible location; no application code, navigation, interaction, picking, authority or resource behavior changes.
