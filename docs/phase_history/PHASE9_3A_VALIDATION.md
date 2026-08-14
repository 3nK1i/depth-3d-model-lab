# Phase 9.3A — Canonical Scene Presentation Foundation

## Scope

Phase 9.3A adds a passive application presentation model over the existing persistent reconstruction scene and the Phase 9.1H authority sidecar. It does not change reconstruction persistence schemas, canonical geometry authority, backend orchestration, or viewer geometry rendering.

Implemented presentation state includes:

- persistent scene ready/degraded identity;
- canonical artifact/provider identity;
- canonical geometry authority;
- file-backed canonical `surface_obj_path` materialization state for the later 9.3B viewer handoff;
- persisted calibration presence;
- fail-closed metric scale state (`not_qualified`, `provisional`, `qualified`);
- explicit relative scale uncertainty when present;
- `certified_metrology=false` throughout;
- truthful viewport scale wording driven only by scene-bound Phase 9.1H metric authority.

## Authority boundaries

- Canonical geometry remains the authoritative spatial scene geometry.
- A materialized OBJ is only a viewer resource handoff; 9.3A does not parse or render it.
- Calibration alone never establishes metric scale.
- Metric wording is derived only from the scene-bound metric qualification assessment.
- Semantic geometry remains semantic context only.
- Measurement/QC authority is unchanged.
- Gaussian and NeRF remain appearance-only.
- DA3 remains an isolated supporting dense/relative geometry provider until later canonicalization/integration.
- Certified metrology remains false.

## Explicitly deferred

- Persistent canonical surface parsing/rendering: Phase 9.3B.
- World frame/camera frustums/trajectory promotion: Phase 9.3C.
- Semantic objects in the canonical scene: Phase 9.3D.
- 3D measurement overlays: Phase 9.3E.
- Scene layers/visibility controls: Phase 9.3F.
- Dense-provider UI and DA3 canonical integration: Phase 9.3G/H.
- Performance/resource hardening and real-scene qualification: Phase 9.3I/J.

## Focused validation

```bash
./scripts/run_phase9_3a_tests.sh quick
./scripts/run_phase9_3a_tests.sh integration
./scripts/run_phase9_3a_tests.sh backward
```

Responsive Reconstruction geometry assertions activate the Reconstruction mode before
checking scroll-area dimensions. Hidden stacked pages may retain stale geometry and are
not a valid 1366×768 workspace measurement.

Dependency-light smoke:

```bash
python scripts/run_phase9_3a_canonical_scene_smoke.py
```

UI-only preview at 1366×768 (no reconstruction/DA3/Gaussian/NeRF runtime is launched):

```bash
QT_QPA_PLATFORM=offscreen python scripts/run_phase9_3a_canonical_scene_preview.py
xdg-open /tmp/phase9_3a_canonical_scene_preview.png
```

For a real application sanity check, load an existing persistent Reconstruction scene and verify the new **CANONICAL SCENE** group. A legacy/no-sidecar scene must remain **Not qualified**. A scene with correctly bound metric authority may show **Provisional metric** or **Qualified metric** with explicit scale uncertainty. The canonical surface must not render in the viewport yet; that is Phase 9.3B.

## Compatibility invariant

`PersistentReconstructionScene` schema version 1 and its historical `reconstruction_ui_enabled=false` fingerprint field remain untouched. Phase 9.3A adds no new top-level mode and no heavy runtime dependency.
