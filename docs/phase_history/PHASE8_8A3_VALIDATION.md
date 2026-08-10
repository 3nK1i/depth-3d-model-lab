# Phase 8.8A3 — Offline Reconstruction Execution Bridge

## Scope

A3 adds the first production application-facing offline execution path while Reconstruction remains disabled in the UI.

The bridge supports:

- image-sequence directories;
- explicit ordered image paths;
- recorded video;
- non-blocking `QProcess` execution from `ReconstructionController`;
- production ingestion/keyframe selection in the `depth` environment;
- isolated external PyCOLMAP sparse reconstruction;
- application-level progress, log, result, cancellation and completion signals;
- bounded process-group cleanup fallback.

A3 intentionally does **not** claim a persistent scene, certified metrology, canonical dense geometry, or UI enablement. Persistent scene/checkpoint/resume integration remains A4.

## Architecture

```text
ReconstructionController
    -> OfflineReconstructionController
        -> QProcess (depth Python)
        -> scripts/run_reconstruction_offline_job.py
            -> ReconstructionIngestionPipeline
            -> ExternalPycolmapSfmBackend
                -> eia-pycolmap-probe Python / PyCOLMAP worker
```

`ApplicationController` remains the live capture owner. Offline image/video files are read independently by the reconstruction source readers; A3 does not create a second live camera/RTSP capture subsystem.

## Locked boundaries

- no `tests` imports in the production offline worker;
- no direct `pycolmap` import in `app/` or the production worker;
- no Gaussian, NeRF, DA3 or Stella dependency leakage into `depth`;
- sparse monocular SfM remains arbitrary-scale (`unit_to_metre=None`);
- Reconstruction navigation/page remains disabled;
- no `core/reconstruction` algorithm changes in A3.

## Validation

Focused automated gates:

```bash
./scripts/run_phase8_8a3_tests.sh quick
./scripts/run_phase8_8a3_tests.sh integration
./scripts/run_phase8_8a3_tests.sh backward
```

Real controller-level PyCOLMAP gate, using a real overlapping image set:

```bash
rm -rf /tmp/phase8_8a3_offline_controller_pycolmap
python scripts/run_phase8_8a3_offline_controller_pycolmap_smoke.py \
  --images /ABSOLUTE/PATH/TO/OVERLAPPING_IMAGES \
  --sfm-python ~/anaconda3/envs/eia-pycolmap-probe/bin/python \
  --max-keyframes 50
```

or a recorded video with sufficient camera motion and overlap:

```bash
rm -rf /tmp/phase8_8a3_offline_controller_pycolmap
python scripts/run_phase8_8a3_offline_controller_pycolmap_smoke.py \
  --video /ABSOLUTE/PATH/TO/VIDEO.mp4 \
  --sfm-python ~/anaconda3/envs/eia-pycolmap-probe/bin/python \
  --max-keyframes 50 \
  --video-target-fps 2.0
```

Required real result characteristics:

- controller smoke PASS;
- `production_input_assembly=true`;
- `tests_fixture_used=false`;
- `pycolmap_external_process=true`;
- `production_depth_imports_pycolmap=false`;
- at least 3 registered keyframes and non-zero landmarks;
- gauge remains arbitrary and `unit_to_metre=null`;
- `canonical_geometry_authoritative=false` at this A3 sparse boundary;
- `persistent_scene_ready=false`;
- `reconstruction_ui_enabled=false`.

Do not commit A3 until the focused gates and one real controller-level PyCOLMAP run pass.
