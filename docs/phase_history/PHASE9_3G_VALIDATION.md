# Phase 9.3G — Advanced Reconstruction / Dense Geometry Provider UI

Baseline: clean pushed `phase9-digital-twin` at `f6a647ac29070bc131f7b3d7a63cbff7d55d4c1c` (Phase 9.3F closed).

## Scope

9.3G adds a professional, read-only **ADVANCED RECONSTRUCTION** provider inspector to the existing Reconstruction/Digital Twin workspace. It reuses persistent-scene artifact/provider provenance and the Phase 9.3A canonical-scene scale presentation.

The UI exposes:

- current dense-geometry provider;
- current canonical surface provider;
- current supporting depth-evidence provider;
- execution mode and whether each provider is bound to the active scene;
- canonical pipeline identity;
- current canonical scale qualification text;
- a visible Depth Anything 3 provider contract with its isolated/external and supporting-only authority clearly stated;
- explicit notice that DA3 canonical integration is deferred to Phase 9.3H.

The provider combo is an **inspection selector only**. It does not alter pipeline planning, provider priority, scene evidence, or runtime execution.

## Authority / runtime boundaries

- Canonical dense/surface artifacts keep their existing canonical-geometry authority.
- Depth providers remain supporting geometry evidence.
- DA3 never creates metric scale and does not receive canonical authority merely by appearing in the UI.
- `certified_metrology` remains false unless separately qualified by the existing authority system; 9.3G never changes it.
- 9.3G does not probe, import, launch, install, reset, clean, or modify the isolated DA3 runtime/checkout.
- Gaussian and NeRF behavior is unchanged and remains appearance-only.
- No persistence schema or controller redesign is introduced.

## Deferred to 9.3H

- real DA3 execution from the Digital Twin workflow;
- DA3 result materialization for the active scene;
- registration/canonicalization of DA3 supporting geometry;
- provider activation/execution controls;
- any scale qualification derived from legitimate external evidence.

## Validation gates

```bash
./scripts/run_phase9_3g_tests.sh quick
./scripts/run_phase9_3g_tests.sh integration
./scripts/run_phase9_3g_tests.sh backward
python scripts/run_phase9_3g_dense_provider_smoke.py
python scripts/run_phase9_3g_dense_provider_preview.py --show
```

Manual validation is limited to the new provider inspector, its 1366x768 responsiveness, correct scene-bound/provider state, and preservation of all existing viewport/layer/appearance behavior.
