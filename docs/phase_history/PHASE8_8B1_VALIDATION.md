# Phase 8.8B1 Validation — Reconstruction Page Skeleton

## Scope

Phase 8.8B1 introduces the dormant Reconstruction page shell after the Phase 8.8A controller boundary reached its readiness gate.

This step is intentionally UI-foundation only:

- adds `ReconstructionPage` to the existing `QStackedWidget`;
- keeps `AppMode.RECONSTRUCTION` outside `_ENABLED_MODES`;
- adds no `_on_mode_changed()` Reconstruction branch;
- passively mirrors qualified live/offline/scene/checkpoint controller state;
- adds disabled placeholders for source/workflow and runtime actions;
- adds a future reconstruction viewport surface and settings/persistence column;
- starts no Stella, PyCOLMAP, DA3, Gaussian, or NeRF backend;
- changes no `core/reconstruction` contract or release metadata.

## Page hierarchy

- Reconstruction header/readiness badge
- Source & Workflow card
- Runtime Status card
- Reconstruction viewport placeholder
- Reconstruction Settings sidebar
  - Workflow
  - Session & Persistence
  - Authority & Safety

## Safety gates

The page exists in the stack but cannot be reached through primary navigation. Programmatic `ModeTabs.set_active_mode(AppMode.RECONSTRUCTION)` must still reject the request.

All B1 action buttons are disabled. Controller binding is passive only.

## Validation

```bash
./scripts/run_phase8_8b1_tests.sh quick
./scripts/run_phase8_8b1_tests.sh backward
```

Manual sanity:

```bash
python main.py
```

Confirm existing modes remain unchanged and Reconstruction remains visible-but-disabled in the mode tabs. No backend should start during application launch.

Because production navigation is intentionally still locked, inspect the B1 page with the UI-only preview helper:

```bash
python scripts/run_phase8_8b1_reconstruction_page_preview.py
```

The preview bypasses navigation only inside that process and starts no reconstruction backend.
