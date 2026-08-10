# Phase 8.8B5 — Reconstruction Navigation Enablement

## Scope

Phase 8.8B5 is the final Phase 8.8B gate. It enables the already-built Reconstruction page in normal primary navigation after the Phase 8.8A controller hardening and B1–B4 UI/runtime/status gates.

Production changes are intentionally small:

- `ModeTabs` enables `AppMode.RECONSTRUCTION` through the explicit `_RECONSTRUCTION_NAVIGATION_ENABLED` B5 gate while preserving the legacy five-mode `_ENABLED_MODES` set.
- `MainWindow._on_mode_changed()` routes Reconstruction to the existing sixth page.
- Reconstruction activity locks primary navigation while a live/offline reconstruction owns the long-running job slot, then restores all primary tabs when the job is terminal.
- Merely entering Reconstruction does not select a source, create a session, or start Stella/PyCOLMAP.

No reconstruction algorithm, worker protocol, persistence schema, heavy dependency, release metadata, or `core/reconstruction` implementation changes in B5.

## Historical regression migration

Earlier Phase 7/8 UI tests deliberately asserted that Reconstruction was disabled until Phase 8.8. B5 intentionally supersedes that temporary UI lock. The stable five-mode `_ENABLED_MODES` contract is retained, while the separate B5 gate activates Reconstruction; only assertions that describe effective navigation/routing are migrated.

## Required validation

```bash
./scripts/run_phase8_8b5_tests.sh quick
./scripts/run_phase8_8b5_tests.sh history
./scripts/run_phase8_8b5_tests.sh backward
python scripts/run_phase8_8b5_navigation_smoke.py
```

Manual sanity in the normal application:

1. Start `python main.py`.
2. Verify Reconstruction is enabled in the top navigation.
3. Enter Reconstruction and verify no backend starts automatically.
4. Switch repeatedly between Reconstruction and every existing mode.
5. Return to Reconstruction and verify its page/state remains intact.
6. Close the application from Reconstruction and verify clean shutdown.

## B5 acceptance contract

- all six primary modes are enabled while idle;
- Reconstruction routes to the existing `ReconstructionPage`;
- navigation-only entry never starts Stella or PyCOLMAP;
- active Reconstruction temporarily locks other primary tabs;
- terminal Reconstruction restores idle navigation;
- existing operational modes remain intact;
- no `core/reconstruction` changes;
- no heavy backend imports added to the production `depth` environment.
