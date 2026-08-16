# Phase 9.6F — Digital Twin Persistence / Restore / Export Consolidation

Baseline: `ec09a519d5f8399c6a4dfd4c0a1aa30abca90cd5`

## Scope

Phase 9.6F keeps the existing `PersistentReconstructionScene` as the authoritative scene manifest and adds one checksum-protected, scene-bound workspace sidecar. The sidecar persists only unified-workspace state that was intentionally session-side before F:

- Phase 9.6E0 estimated metric-scale hint and its explicit uncertainty/provenance;
- the active canonical ruler endpoints/result, including the original estimated metric result so later qualified scale can supersede the current readout without erasing provenance;
- selected persisted Gaussian/NeRF appearance artifact metadata (selection only; no renderer auto-launch);
- last terminal E1 build-job summary for history only. Active jobs are never persisted as resumable work.

Save Scene writes the existing scene/authority contracts first and then writes the workspace sidecar. Load/Reload verifies the canonical scene first, loads the authority sidecar through the existing controller, then applies the optional 9.6F workspace sidecar. Legacy scenes without the sidecar remain valid.

`Export State` is deliberately separate from the existing Phase 8.9 `Export Bundle`. It packages integrity manifests and workspace state for audit/same-environment restore, explicitly declares that heavy referenced payloads are not embedded, and never claims portability, geometry authority, metric authority, QC authority or certified metrology.

## Authority invariants

- canonical geometry remains the only spatial/picking/measurement geometry authority;
- estimated scale remains uncalibrated and non-authoritative;
- Gaussian/NeRF remain appearance-only;
- terminal build history is non-resumable and grants no authority;
- Product export remains the materialized geometry/QC export route;
- workspace/state export is metadata/integrity context only;
- `certified_metrology` remains false.

## Manual acceptance

1. Load a scene and create an estimated ruler reference.
2. Save Scene to a new folder; confirm `digital_twin_workspace_state.json` and its SHA-256 sidecar are created beside the persistent scene manifest.
3. Clear the estimate or restart the app, then Load Scene from the saved folder. Confirm the estimated-scale hint and ruler provenance return, while no Gaussian/NeRF interactive backend starts automatically.
4. If a terminal DA3/NeRF build exists before save, confirm BUILD & PROCESSING shows it under `Last build` after restore, but the current job remains Idle.
5. Use `Export State`; verify the state-export folder is created and the UI explicitly states referenced payloads are not embedded.
6. Confirm the existing `Export Bundle`, checkpoint, product-QC, scene layers and authority presentation remain available.
