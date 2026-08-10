# Phase 8.8D1 — Offline Reconstruction Visualization Foundation

## Scope

D1 gives the existing production PyCOLMAP sparse-SfM path a first-class view in the Reconstruction viewport.

The qualified offline worker still owns the complete `sfm_result.json` and sparse PLY. D1 adds a bounded companion `offline_visualization.json` containing only registered camera poses plus a deterministic landmark/color sample for UI rendering.

D1 does **not** promote sparse SfM to canonical geometry, metric scale, or certified metrology.

## Presentation contract

- pose convention remains `T_reconstruction_from_camera`;
- raw monocular SfM remains arbitrary gauge;
- completed A3-style results remain `Scene = Partial` unless a later integrated pipeline produces canonical geometry;
- UI parsing is bounded to a 16 MiB artifact;
- worker landmark sample cap is 8,192;
- current live trajectory/current-camera state is cleared before an offline result is shown;
- registered offline cameras reuse bounded historical-frustum presentation;
- sparse landmark and camera visibility controls remain presentation-only;
- a visualization-artifact failure must not rewrite a completed authoritative offline result to failed.

## Required gates

```bash
./scripts/run_phase8_8d1_tests.sh quick
./scripts/run_phase8_8d1_tests.sh protocol
./scripts/run_phase8_8d1_tests.sh backward
```

Interactive preview without a backend:

```bash
python scripts/run_phase8_8d1_offline_visualization_preview.py
```

Real South Building / isolated PyCOLMAP gate:

```bash
rm -rf /tmp/phase8_8d1_offline_visualization
python scripts/run_phase8_8d1_offline_visualization_smoke.py \
  --images /home/eiatech/Desktop/phase8_3c_south_building/south-building/images \
  --sfm-python ~/anaconda3/envs/eia-pycolmap-probe/bin/python
```

Expected real result:

- offline state `completed`;
- `Scene = Partial`;
- bounded visualization artifact exists;
- registered cameras visible;
- sparse landmarks visible;
- no live trajectory/current-camera pose remains;
- scale remains `arbitrary`;
- `canonical_geometry_authoritative = false`;
- `certified_metrology = false`;
- PyCOLMAP remains an external isolated process;
- production application process does not import `pycolmap`.
