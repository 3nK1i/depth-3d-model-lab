# Phase 9.3B — Persistent Canonical Geometry → Interactive 3D Viewer

## Scope

Phase 9.3B renders the already-persisted canonical `surface_obj_path` inside the existing Reconstruction `ReconstructionCanvas`. The viewer consumes the integrity-bound persistent OBJ as a presentation resource; it does not rebuild canonical geometry, alter scene authority, or create scale.

Implemented behavior includes:

- lightweight parsing of the project's persisted triangular OBJ surface format;
- independent file existence, size, and SHA-256 verification before parsing;
- reconstruction-frame → viewer Z-up coordinate conversion through the existing presentation transform;
- one reusable VisPy canonical mesh visual in the existing Reconstruction viewport;
- persisted vertex colors when present, with a neutral presentation fallback when absent;
- canonical surface bounds included in **Fit scene** without retaining duplicate full-size fit geometry;
- automatic load on verified persistent-scene load/reload and deterministic clear on scene removal/shutdown;
- VisPy-compatible mesh clearing that never submits a zero-vertex mesh to `MeshVisual.set_data()`; loaded visuals are hidden, detached from the scenegraph, and replaced with a fresh empty hidden mesh node before reuse;
- fail-closed viewer behavior if the materialized surface is missing, tampered, malformed, or inconsistent with persisted vertex/face counts;
- canonical authority retained when only the optional viewer resource fails.

## Authority boundaries

- `PersistentReconstructionScene` remains canonical geometry authority.
- The parsed OBJ arrays are presentation resources only.
- Phase 9.3B never fabricates a `SurfaceMeshResult` from persisted metadata.
- Viewer Z-up conversion never changes reconstruction coordinates or metric authority.
- Calibration alone never establishes metric scale.
- `certified_metrology` remains false.
- Semantic geometry, measurements, inspection/QC, Gaussian, NeRF, and DA3 authority boundaries are unchanged.
- No reconstruction, DA3, Gaussian, or NeRF backend is launched by canonical-surface loading.

## Explicitly deferred

- World frame/camera frustums/trajectory presentation work: Phase 9.3C.
- Semantic objects in canonical 3D: Phase 9.3D.
- Measurement overlays: Phase 9.3E.
- Professional scene layer/visibility controls: Phase 9.3F.
- Advanced provider UI and DA3 canonical integration: Phase 9.3G/H.
- Performance/resource hardening and real-scene qualification: Phase 9.3I/J.

## Focused validation

```bash
./scripts/run_phase9_3b_tests.sh quick
./scripts/run_phase9_3b_tests.sh integration
./scripts/run_phase9_3b_tests.sh backward
```

Dependency-light canonical-resource smoke:

```bash
python scripts/run_phase9_3b_canonical_mesh_smoke.py
```

Real VisPy/PyQt preview at 1366×768:

```bash
python scripts/run_phase9_3b_canonical_mesh_preview.py --show
```

The preview must show the persisted canonical surface interactively in the existing Reconstruction viewport, remain horizontally clean at 1366×768, and report canonical geometry authority without certified-metrology claims. Orbit/pan/zoom, reset, and fit should remain responsive. Closing the preview must release the canonical viewer resource without starting any heavy reconstruction backend.

The real-VisPy preview is also the required regression gate for the qualified VisPy runtime: initial scene binding, scene clear, reload, and viewport shutdown must complete without a zero-size bounds reduction error.

## Compatibility invariant

`PersistentReconstructionScene` schema version 1, Phase 9.1H authority persistence, the existing Reconstruction controller/persistence flow, representation switching, semantic/measurement/QC authority, and heavy-runtime isolation remain unchanged. Phase 9.3B adds no top-level mode and does not pull forward Phase 9.3C–9.3J behavior.
