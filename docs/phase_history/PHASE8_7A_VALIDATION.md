# Phase 8.7A Validation — Surface / Mesh Representation

## Scope

Phase 8.7A adds the first quality-aware triangle-mesh representation on top of the locked Phase 8.6 dense-geometry contracts.

Required baseline:

- branch: `phase8-reconstruction`
- HEAD before applying: `85ccbdd5a87a4f0b9713a094a0275784172cbed3`

This step is intentionally bounded. It adds surface/mesh representation only. It does **not** add semantic objects, connector QC, CAD/golden registration, Gaussian Splatting, NeRF, or Reconstruction UI enablement.

## Architecture

`core/reconstruction/surface_mesh.py` introduces:

- `SurfaceMeshingPolicy`
- `SurfaceMesh`
- `SurfaceMeshStatistics`
- `SurfaceMeshResult`
- deterministic quality-aware local tangent-plane triangulation
- product-aware defaults for `object_inspection` and `spatial_digital_twin`
- bounded `max_vertices` / `max_faces`
- small-component cleanup
- quality / uncertainty propagation from Phase 8.6D
- scale / world-alignment / pose provenance propagation
- validation-only OBJ writer for non-UI inspection

The mesher uses SciPy `cKDTree`, already in the project dependency set. No Open3D/VTK dependency is introduced.

## Safety invariants

- Dense-fusion and geometry-quality schema versions remain unchanged.
- Dense point positions are not smoothed or silently moved.
- Rejected/low-quality points are not silently promoted into the mesh.
- Large local gaps are not bridged beyond the meshing policy.
- Arbitrary geometry stays arbitrary with `unit_to_metre=None`.
- Metric geometry remains provisional/calibrated according to the existing 8.6 semantics.
- Local mesh uncertainty and relative metric-scale uncertainty remain separate.
- `certified_metrology=False` is enforced.
- Pose convention remains `T_reconstruction_from_camera`.
- PyCOLMAP and stella_vslam remain externally isolated.
- Reconstruction UI remains disabled.

## Focused validation

```bash
./scripts/run_phase8_7a_tests.sh quick
./scripts/run_phase8_7a_tests.sh integration
./scripts/run_phase8_7a_tests.sh stress
python scripts/run_surface_mesh_smoke.py
```

Expected focused counts for this package:

- quick: 28 passed
- integration: 1 passed
- stress: 1 passed

The smoke must end with:

```text
Phase 8.7A quality-aware surface mesh smoke: PASS
```

It writes:

- `/tmp/phase8_7a_surface_mesh/arbitrary_surface.obj`
- `/tmp/phase8_7a_surface_mesh/metric_surface.obj`
- `/tmp/phase8_7a_surface_mesh/summary.json`

The OBJ files are validation artifacts only. Productized export remains a later release/productization concern.

## Backward gates

Run prior quick gates from 8.6D through 8.1 and keep the established target counts. Also verify:

```bash
grep -RInE \
'^[[:space:]]*(from[[:space:]]+pycolmap|import[[:space:]]+pycolmap)' \
core/reconstruction \
&& echo "ERROR: direct PyCOLMAP import found" \
|| echo "PASS: core/reconstruction remains PyCOLMAP-free"

grep -RInE \
'^[[:space:]]*(from[[:space:]]+stella_vslam|import[[:space:]]+stella_vslam)|ctypes\.(CDLL|PyDLL)' \
core/reconstruction \
&& echo "ERROR: native stella linkage found in Python" \
|| echo "PASS: stella_vslam remains external-process isolated"
```

## Final target regression

```bash
./scripts/run_phase8_7a_tests.sh full
python main.py
```

Acceptance:

- unit/UI/integration/stress groups green;
- only the same established optional real-perception skips are acceptable;
- application starts and exits cleanly;
- existing modes remain unaffected;
- Reconstruction remains disabled;
- `git diff --check` is empty.

Do not commit until the target-machine smoke and full regression have passed.

## Known Phase 8.7A boundary

The local tangent-fan mesher is a deterministic, bounded surface representation for trusted dense geometry. It is not yet a watertight CAD mesher, topology repair system, semantic surface model, or certified metrology backend. Those concerns belong to later 8.7/8.9 work.
