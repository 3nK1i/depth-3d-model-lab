# Phase 8.9F6 — Actual Application End-to-End Acceptance

## Purpose

F6 is the final Phase 8.9F acceptance gate. It loads the self-contained real
8.9F5 `PersistentReconstructionScene` through the actual PyQt
`ReconstructionController` and `MainWindow` and validates the existing
application presentation/lifecycle contracts. F6 adds no reconstruction or
neural-rendering algorithm.

## Source baseline

- Branch: `phase8-reconstruction`
- Required HEAD/origin: `eb5062bebeb146ad98d658c19fab7ec4750f2834`
- Required F5 source: `/tmp/phase8_9f5_persistent_scene`

## Real acceptance chain

1. Load the F5 persistent manifest through `ReconstructionController.load_persistent_scene()` with file-reference verification.
2. Confirm scene identity, canonical-geometry authority, and `certified_metrology=false`.
3. Confirm the initial application view is the canonical/scene-ready state.
4. Switch to the persisted backend-authored Gaussian static preview.
5. Switch to the persisted backend-authored NeRF static preview.
6. Return to canonical state and reload the persistent scene; identity/fingerprint must remain stable.
7. Exercise missing-NeRF UI fallback using an in-memory presentation clone only. The real F5 manifest/artifacts are never modified. Fallback must retain canonical geometry and must not silently substitute Gaussian.
8. Rebind the exact real scene and prove NeRF presentation recovers.
9. Validate 1366×768 layout, no horizontal settings-sidebar scroll, authority text, isolation, and clean shutdown.

## Truthful scope

- Canonical geometry remains authoritative.
- Gaussian and NeRF remain `appearance_only`.
- F6 does not directly render the persisted canonical OBJ with new code. It validates the existing application canonical/scene-ready state and the already-qualified persisted appearance switching contract.
- Gaussian/NeRF previews are already-authored static images; F6 starts no neural backend.
- No PyCOLMAP, DA3, gsplat training/rendering, or Nerfstudio/Nerfacto training/rendering is started.
- No new UI architecture is introduced.
- No certified-metrology claim is made.
- F5 already qualified missing/tampered persistent-file rejection. The F6 missing-preview probe is deliberately an in-memory UI presentation clone so source evidence remains immutable.

## Outputs

Default output root: `/tmp/phase8_9f6_actual_application`

- `01_canonical_scene_ready.png`
- `02_gaussian_appearance.png`
- `03_nerf_appearance.png`
- `04_missing_nerf_fallback.png`
- `f6_application_acceptance_contact_sheet.png`
- `summary.json`

## Pass criteria

The campaign passes only when all of the following are true:

- real `MainWindow` and Reconstruction mode are exercised;
- F5 scene loads through the actual controller and all file references verify;
- canonical scene-ready state is present;
- Gaussian and NeRF persisted-preview switching both work;
- hide/reload returns to canonical state;
- scene ID and fingerprint survive reload unchanged;
- missing selected NeRF preview falls back to canonical with no neural cross-fallback;
- real scene presentation recovers after the fallback probe;
- canonical authority remains true and `certified_metrology=false`;
- 1366×768 UI has no horizontal settings-sidebar scroll;
- no external heavy reconstruction runtime is imported into `depth` or started by F6;
- application shutdown is clean;
- repository scope contains only the five F6 qualification files.
