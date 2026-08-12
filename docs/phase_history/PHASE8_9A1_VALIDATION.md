# Phase 8.9A1 Validation — Reconstruction Export Core / Contract

## Scope

Phase 8.9A1 productizes reconstruction export at the dependency-light core boundary only. It does not modify the Reconstruction UI, controllers, persistence/checkpoint/recovery lifecycles, backend execution, or external environments.

The export contract:

- copies already-qualified offline sparse SfM PLY + SfM JSON without importing PyCOLMAP;
- preserves sparse SfM as `supporting_geometry` with arbitrary scale and no metric conversion;
- writes a canonical OBJ only when a `SURFACE_MESH` pipeline artifact is the current canonical geometry artifact;
- writes inspection/QC JSON only from an `INSPECTION_REPORT` artifact with `inspection_decision` authority;
- reuses the existing `write_surface_mesh_obj()` and `write_inspection_report_json()` writers;
- records SHA-256 hashes, file authority, provenance metadata, and `certified_metrology=false` in a bundle manifest;
- protects the manifest with its own SHA-256 sidecar;
- verifies manifest and payload integrity fail-closed;
- refuses existing output directories rather than silently overwriting an export.

Gaussian/NeRF authority is not changed and remains `appearance_only`.

## Files

Production:

- `core/reconstruction/export.py`
- `core/reconstruction/__init__.py`

Validation only:

- `tests/unit/test_phase8_9a1_reconstruction_export.py`
- `tests/integration/test_phase8_9a1_reconstruction_export.py`
- `scripts/run_phase8_9a1_tests.sh`
- `docs/phase_history/PHASE8_9A1_VALIDATION.md`

## Validation

Focused A1:

```bash
./scripts/run_phase8_9a1_tests.sh quick
```

Expected at delivery: 8 passed.

Affected/backward compatibility:

```bash
./scripts/run_phase8_9a1_tests.sh affected
```

Expected at delivery: 79 passed.

The affected gate includes the existing surface mesh, inspection/QC, persistence, Phase 8.7A/8.7C backward-compatibility, and related integration tests.

## Safety conclusion

Phase 8.9A1 adds one bounded export contract and public symbols. Existing reconstruction authority, persistence, checkpoints, recovery, external backend isolation, UI behavior, and release metadata remain unchanged. No Phase 8 release tag is created by this step.
