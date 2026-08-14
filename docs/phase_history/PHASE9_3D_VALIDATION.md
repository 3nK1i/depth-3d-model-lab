# Phase 9.3D — Semantic Objects in Canonical 3D

## Scope

Phase 9.3D extends the existing Reconstruction/Digital Twin viewport with persistent semantic-object presentation bound to the already-authoritative canonical surface.

Implemented behavior:

- preserves the exact `vertex_indices` and `face_indices` already produced by the existing `Semantic3DObject` reconstruction contract when semantic geometry is serialized;
- consumes detailed persisted `semantic_geometry` snapshots only; no perception or semantic reconstruction is run while loading a persistent scene;
- validates persisted semantic membership against the integrity-verified Phase 9.3B canonical mesh before a surface highlight is allowed;
- preserves legacy detailed semantic snapshots that did not store exact membership, while explicitly withholding canonical-surface highlighting for them;
- reuses the existing Phase 8.8D5 semantic selector, support boxes, centroids, Prev/Next/Focus/Clear flow and visibility control;
- adds bounded 3D semantic labels and a selected-object surface overlay without altering the canonical mesh;
- clears semantic labels/highlights deterministically on scene replacement and viewport shutdown;
- keeps persistent-scene schema version unchanged by using the existing generic semantic `payload_snapshot` contract.

No Phase 9.2 reviewed mask is promoted directly to 3D. No YOLO/SAM2 inference is rerun to populate a persistent scene. No reconstruction controller, measurement/QC, DA3, Gaussian or NeRF architecture is added or promoted by this step.

## Authority and safety

- The persistent canonical surface remains the authoritative scene geometry.
- Semantic objects remain **semantic membership/context only** and never canonical geometry authority.
- Exact vertex/face membership is accepted only when it is already persisted and validates against the active canonical surface.
- Legacy objects without persisted membership retain labels/support bounds but receive no fabricated surface highlight.
- 2D reviewed masks remain 2D semantic evidence until a valid reconstruction path explicitly creates supported 3D semantic geometry.
- Semantic/model confidence and mesh quality remain context signals, not measurement uncertainty or inspection acceptance.
- Measurement and QC authority remain separate and unchanged.
- Metric scale is never inferred from semantics, camera poses or intrinsics.
- `certified_metrology` remains `false`.
- A bounded maximum of 256 semantic objects and 64 simultaneous 3D labels is enforced for presentation responsiveness.
- No YOLO, SAM2, DA3, Gaussian or NeRF heavy runtime is imported or launched by the 9.3D persistent presentation loader.

## Focused automated validation

```bash
./scripts/run_phase9_3d_tests.sh quick
./scripts/run_phase9_3d_tests.sh integration
./scripts/run_phase9_3d_tests.sh backward
python scripts/run_phase9_3d_semantic_scene_smoke.py
```

The focused suites cover additive membership serialization, canonical-mesh binding, malformed/out-of-range membership rejection, certified-metrology rejection, legacy compatibility, persistent save/reload determinism, UI selection/highlighting, existing semantic visibility controls, scene switching/clearing and prior 9.3B/9.3C/D5 sanity.

## Real VisPy manual validation

```bash
python scripts/run_phase9_3d_semantic_scene_preview.py --show
```

Verify on the production `depth` environment:

- the integrity-verified canonical surface remains visible and unchanged;
- semantic labels align with their existing 3D objects;
- selecting an object highlights only its validated persisted canonical-surface support;
- the existing support box/centroid remains visible as semantic context, not measurement geometry;
- Prev, Next, Focus and Clear remain responsive;
- the existing semantic visibility button hides/shows boxes, centroids, labels and selected semantic surface presentation;
- Orbit, Pan, Zoom, Fit and Reset remain responsive;
- the viewport continues to report arbitrary/provisional/validated scale truthfully and never claims certified metrology;
- no horizontal sidebar overflow appears at 1366x768;
- switching scenes clears stale semantic labels/highlights;
- viewport shutdown releases both canonical mesh and semantic presentation resources;
- scene loading does not launch YOLO/SAM2, reconstruction, DA3, Gaussian or NeRF heavy backends.

Measurement/inspection overlays remain deferred to Phase 9.3E. Professional per-layer visibility/styling remains deferred to Phase 9.3F.
