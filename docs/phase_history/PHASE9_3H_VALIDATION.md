# Phase 9.3H Validation — DA3 Dense Geometry → Canonical Scene Integration

Phase 9.3H integrates the already-qualified Depth Anything 3 external provider into the existing canonical dense/surface pipeline without changing DA3's authority.

## Implemented

- Explicit **Integrate DA3** action for selected Reconstruction image-sequence or recorded-video sources.
- Non-blocking depth-environment `QProcess` orchestration with bounded cancellation/shutdown.
- Real DA3 inference remains in the isolated `eia-da3` environment/checkout; the production `depth` environment is not contaminated.
- The real worker verifies that the `eia-da3` interpreter imports `DepthAnything3` from the exact qualified external checkout, not merely another checkout at the same Git revision; this preserves the deliberate RTX-2060 compatibility patch/runtime-lock qualification.
- Reuse of the qualified Phase 8 DA3 pose/depth contracts and existing `canonical-dense-fusion → quality-aware-surface-mesh` chain.
- Persisted DA3-derived canonical OBJ, explicit camera matrices/trajectory and DA3 provider provenance.
- Calibration remains optional. Without separate scale qualification the scene stays arbitrary-scale, `unit_to_metre=None`, and `certified_metrology=false`.
- New DA3 scenes do not fabricate Phase 9.2 semantics or inspection/QC evidence. Those contexts are persisted and presented as unavailable.
- Provider selection remains inspection-only; only the explicit integration action may launch the isolated DA3 workflow.
- Existing application resource interlocks now include the DA3 canonical job.
- Main Runtime Status mirrors the DA3 job lifecycle and stage without changing the existing live/offline runtime ownership.

## Authority boundary

DA3 output remains **supporting relative geometry / pose evidence**. Canonical geometry authority belongs only to the downstream canonical dense/surface providers. DA3 never establishes metric scale, measurement authority, QC authority or certified metrology.

## Dependency-light validation

The dedicated smoke reuses deterministic DA3-shaped evidence and exercises the complete depth-side canonicalization/persistence/reload path without probing or starting the real DA3 runtime. This validates contracts and persistence only; it is not a real DA3 performance/quality claim.

## Real validation status

- Real Qt/VisPy post-integration presentation: validate in the production `depth` environment with `scripts/run_phase9_3h_da3_canonical_preview.py --show`.
- Real heavy DA3 execution from the new UI action is intentionally **deferred to Phase 9.3J real-scene end-to-end qualification**, where the already-qualified RTX-2060 path and actual scene evidence can be exercised truthfully.
- Real camera calibration is not required for 9.3H. If unavailable, the canonical scene remains usable at arbitrary scale.

## Regression boundary

Focused backward sanity covers the existing DA3 contracts, Phase 8 F3 canonical geometry bridge, Phase 9.3G provider inspector and Phase 9.3F scene layers. Full application regression remains a phase-closure gate.
