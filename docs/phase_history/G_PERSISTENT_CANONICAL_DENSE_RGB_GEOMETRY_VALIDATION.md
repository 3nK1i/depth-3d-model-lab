# Phase G — Persistent Canonical Dense RGB Geometry

## Scope

Phase G consumes the passing Phase-F filtered dense scene and promotes **Dense RGB Points** into a first-class persistent canonical geometry representation. It does not rerun PyCOLMAP or DA3 and does not introduce UI or surface-mesh work.

The persistence contract lives in `core/reconstruction/dense_scene_persistence.py`. It intentionally does not force the Phase-G point cloud through the older integrated `PersistentReconstructionScene` manifest because that legacy contract requires surface-mesh, semantic and inspection stages. G must not fabricate those later-stage artifacts merely to persist Dense RGB Points.

## Persistent bundle

The G scene is self-contained and portable. All references stored in the manifest are relative to the scene directory and integrity-bound with SHA-256:

- `geometry/dense_rgb_extended.npz` — primary canonical Dense RGB Points.
- `geometry/dense_rgb_core.npz` — stricter high-confidence canonical Dense RGB Points.
- `exports/dense_rgb_extended.ply` — portable primary point-cloud export.
- `exports/dense_rgb_core.ply` — portable high-confidence export.
- `diagnostics/rejected_supported.npz` — F rejected supported geometry.
- `diagnostics/attached_single_view_fringe.npz` — attached single-view coverage diagnostic.
- `diagnostics/detached_single_view_artifacts.npz` — detached single-view artifact diagnostic.
- `diagnostics/coverage_candidate.npz` — F coverage candidate diagnostic.
- `provenance/f_summary.json` — immutable F qualification provenance.
- `provenance/e3_summary.json` — immutable E3 expansion provenance.
- `persistent_dense_rgb_scene.json` + SHA-256 sidecar — scene identity, authority, relative references and quality snapshots.
- `g_summary.json` — G validation/next-milestone decision.

## Authority

A passing G run promotes:

- `canonical_geometry_authoritative=True` for Dense RGB Points.
- `canonical_persistence_promoted=True`.

It deliberately preserves:

- arbitrary SfM gauge / world alignment;
- `metric_scale_claimed=False`;
- `measurement_authoritative=False`;
- `certified_metrology=False`;
- `surface_mesh_generated=False`;
- `reconstruction_ui_enabled=False`.

Thus G establishes durable geometry/picking/spatial authority without inventing physical scale. H owns UI integration; J owns mesh reconstruction.

## Required gates

- F is `PASS` and explicitly authorizes G.
- F filtered dense geometry is qualified and no later authority was prematurely promoted.
- Core/Extended NPZ counts, voxel size and selected-view provenance match F.
- Core voxel keys are a subset of Extended voxel keys.
- Core/Extended PLY export vertex counts match their NPZ products.
- The output directory is atomically published and never overwrites non-empty existing data.
- Every persistent reference is relative, inside the bundle and SHA-256 verified.
- Both Extended and Core restore from the G bundle alone.
- Canonical persistence/geometry authority is promoted while metric/measurement/UI/mesh authority remains unpromoted.

A passing G run authorizes only **H — Dense RGB Reconstruction UI Integration**.

## Validation commands

```bash
python -m py_compile \
  core/reconstruction/dense_scene_persistence.py \
  scripts/run_g_persistent_dense_rgb_geometry.py \
  scripts/run_g_persistent_dense_rgb_restore.py \
  tests/unit/test_g_persistent_dense_rgb_geometry.py

bash scripts/run_g_tests.sh quick
bash scripts/run_g_tests.sh affected
```

Real persistence:

```bash
rm -rf /tmp/eia_g_persistent_dense_rgb_geometry
python scripts/run_g_persistent_dense_rgb_geometry.py \
  --f-dir /tmp/eia_f_confidence_artifact_filtering \
  --output-dir /tmp/eia_g_persistent_dense_rgb_geometry
```

Bundle-only reload:

```bash
python scripts/run_g_persistent_dense_rgb_restore.py \
  --scene-dir /tmp/eia_g_persistent_dense_rgb_geometry
```

Optional relocation check:

```bash
rm -rf /tmp/eia_g_persistent_dense_rgb_geometry_moved
cp -a /tmp/eia_g_persistent_dense_rgb_geometry \
  /tmp/eia_g_persistent_dense_rgb_geometry_moved

python scripts/run_g_persistent_dense_rgb_restore.py \
  --scene-dir /tmp/eia_g_persistent_dense_rgb_geometry_moved
```
