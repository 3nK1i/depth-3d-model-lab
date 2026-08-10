# Phase 8.7B Validation — Semantic Object / Connector Geometry

## Baseline

- Branch: `phase8-reconstruction`
- Required pre-apply HEAD: `81bdf64e0d03f55d5dae1130b5105c4db0befc8e`
- Phase 8.7A surface/mesh representation remains unchanged.
- Reconstruction UI remains disabled.

## Scope

Phase 8.7B adds a dependency-light semantic geometry layer that projects authoritative Phase 7.5 perception masks onto the trusted Phase 8.7A mesh and associates repeated observations into persistent 3D semantic objects.

The step adds:

- explicit `SemanticFrameObservation` camera-pose input,
- distortion-aware mesh-to-image projection,
- authoritative-mask-only 2D→3D assignment,
- per-observation mesh vertex/face assignments,
- cross-view association using explicit tracker IDs and mesh-overlap evidence,
- persistent `Semantic3DObject` artifacts,
- explicit connector roles (`CONNECTOR_BODY`, `CONNECTOR_PIN`, `CONNECTOR_PIN_ROW`, `CONNECTOR_SLOT`, `CONNECTOR_HOUSING`, `REFERENCE_SURFACE`),
- generic digital-twin entity semantics,
- quality/uncertainty, camera, pose and scale provenance propagation,
- deterministic grouped OBJ export for non-UI inspection,
- hard observation/object capacity bounds.

## Safety semantics

Phase 8.7B deliberately does **not**:

- infer connector/pin roles from class names without an explicit role mapping,
- promote one 2D mask directly to a persistent object without association,
- merge different explicit tracker IDs merely because masks overlap,
- merge same-frame observations as cross-view evidence,
- deform or rewrite Phase 8.7A mesh vertices,
- promote arbitrary geometry to metric geometry,
- claim certified metrology,
- perform CAD/golden registration,
- make pass/warn/reject inspection decisions,
- add Gaussian Splatting or NeRF,
- enable Reconstruction UI.

`T_reconstruction_from_camera` remains the pose convention. Arbitrary geometry keeps `unit_to_metre=None`; provisional metric semantics are inherited unchanged from Phase 8.6B/8.7A.

## Package files

Exactly 9 files are delivered:

1. `PHASE8_7B_VALIDATION.md`
2. `core/reconstruction/__init__.py`
3. `core/reconstruction/semantic_geometry.py`
4. `scripts/run_phase8_7b_tests.sh`
5. `scripts/run_semantic_geometry_smoke.py`
6. `tests/integration/test_reconstruction_semantic_geometry_pipeline.py`
7. `tests/stress/test_reconstruction_semantic_geometry_stress.py`
8. `tests/unit/test_phase8_7b_backward_compatibility.py`
9. `tests/unit/test_reconstruction_semantic_geometry.py`

## Focused validation

```bash
./scripts/run_phase8_7b_tests.sh quick
./scripts/run_phase8_7b_tests.sh integration
./scripts/run_phase8_7b_tests.sh stress
```

Expected focused counts for this package:

- quick: `34 passed`
- integration: `1 passed`
- stress: `1 passed`

## Deterministic semantic smoke

```bash
rm -rf /tmp/phase8_7b_semantic_geometry
python scripts/run_semantic_geometry_smoke.py
cat /tmp/phase8_7b_semantic_geometry/summary.json
```

Acceptance requires:

- final line begins with `Phase 8.7B semantic object/connector geometry smoke: PASS`,
- 9 observations accepted,
- 3 persistent objects,
- connector body associated across 3 views,
- two pin objects remain distinct across 3 views each,
- `ambiguous_vertices=0` for the synthetic smoke,
- association reports `track_id_plus_cross_view_mesh_overlap`,
- `pose_convention=T_reconstruction_from_camera`,
- `metric_geometry_status=arbitrary`,
- `unit_to_metre=null`,
- `certified_metrology=false`,
- `inspection_qc_enabled=false`,
- `reconstruction_ui_enabled=false`.

A grouped mesh artifact is written to:

`/tmp/phase8_7b_semantic_geometry/semantic_groups.obj`

## Backward gates

Run the established quick gates from 8.7A back through 8.1 and keep their established target counts unchanged. Verify direct PyCOLMAP and stella imports remain absent from `core/reconstruction`.

## Full target regression

```bash
./scripts/run_phase8_7b_tests.sh full
python main.py
```

All test groups must be green, except the same two established optional target-hardware perception skips if still applicable. Existing application modes must remain unaffected and Reconstruction must remain disabled.

## Commit gate

Before commit:

```bash
git diff --check
git status --short
git branch --show-current
git rev-parse HEAD
```

HEAD must still be `81bdf64e0d03f55d5dae1130b5105c4db0befc8e` before staging the Phase 8.7B changes.
