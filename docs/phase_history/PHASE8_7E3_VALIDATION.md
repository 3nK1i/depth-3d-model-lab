# Phase 8.7E3 — NeRF Real-Scene Validation

## Scope

E3 validates the E2-qualified Nerfstudio/Nerfacto backend on a real image capture with a deterministic held-out split. It measures appearance quality only.

E3 owns:

- real-image dataset preparation into the canonical `NeuralSceneDataset` contract;
- OpenGL camera-to-world -> EIA `T_reconstruction_from_camera` conversion;
- distortion removal/downscaling for a pinhole validation dataset;
- deterministic trajectory-spread train/held-out selection;
- exact Nerfstudio filename train/val/test split materialization;
- real Nerfacto CUDA training and checkpoint reload/rendering;
- held-out PSNR / SSIM / LPIPS using the already-qualified backend-neutral D3 metrics worker;
- visual reference-vs-render contact sheet;
- conservative non-degenerate quality sanity gates.

E3 does **not** own VRAM profiling, sustained training time, render FPS, crash/cancel/recovery, artifact hardening, RTX 5090 measurement, or whole-engine behavior. Those remain in 8.7E4 / 8.7G / 8.9 as already assigned.

## Authority boundaries

- NeRF scene role: `appearance_only`.
- `canonical_geometry_authoritative=False`.
- `certified_metrology=False`.
- Metric geometry remains whatever the canonical input dataset says; the official poster validation path is deliberately `arbitrary` scale.
- Reconstruction UI remains disabled.

## Default real-scene qualification profile

- Official Nerfstudio `poster` real-world capture.
- 24 trajectory-spread selected source views.
- 20 training views + 4 held-out views.
- Images are OpenCV-undistorted and downscaled to maximum long edge 720 px before entering the EIA neural-scene contract.
- Nerfacto profile: `efficient`.
- Training steps: 1000.
- Camera optimizer: off.
- Nerfstudio auto orientation/centering/scaling: off.
- Metrics: existing Phase 8.7D3 isolated `eia-gaussian-metrics` worker, LPIPS AlexNet package 0.1.4.

These settings are a real-scene validation workload, not a production-quality or performance ranking profile.

## Conservative quality sanity policy

Aggregate held-out evidence must be finite and satisfy:

- PSNR >= 8 dB
- SSIM >= 0.05
- LPIPS <= 0.95
- at least 3 held-out views

These are only non-degeneracy gates. They are not product acceptance thresholds and must not be represented as certified image-quality targets.

## Focused validation

```bash
./scripts/run_phase8_7e3_tests.sh quick
./scripts/run_phase8_7e3_tests.sh integration
./scripts/run_phase8_7e3_tests.sh stress
python scripts/run_nerf_e3_contracts_smoke.py
```

## Real-scene preparation

After downloading the official Nerfstudio poster capture to a local directory containing `transforms.json` and `images/`:

```bash
rm -rf /tmp/phase8_7e3_poster_prepared

python scripts/prepare_nerf_e3_real_scene.py \
  --source-dir /ABSOLUTE/PATH/TO/poster \
  --output-dir /tmp/phase8_7e3_poster_prepared \
  --max-frames 24 \
  --heldout-count 4 \
  --max-long-edge 720
```

Expected outputs:

- `/tmp/phase8_7e3_poster_prepared/dataset.json`
- `/tmp/phase8_7e3_poster_prepared/split.json`
- `/tmp/phase8_7e3_poster_prepared/preparation_summary.json`
- prepared undistorted/downscaled images.

## Real RTX 2060 held-out validation

```bash
rm -rf /tmp/phase8_7e3_nerf_real_scene

python scripts/run_nerf_real_scene_validation.py \
  --dataset-manifest /tmp/phase8_7e3_poster_prepared/dataset.json \
  --split-manifest /tmp/phase8_7e3_poster_prepared/split.json \
  --source-label nerfstudio-official-poster \
  --profile efficient \
  --training-steps 1000
```

Expected successful closure includes:

```text
Phase 8.7E3 real-scene Nerfacto validation: PASS
quality_evidence_complete=True quality_sanity_pass=True
scene_role=appearance_only canonical_geometry_authoritative=False certified_metrology=False
performance_hardening_deferred_to=8.7E4 reconstruction_ui_enabled=False
```

Inspect:

- `/tmp/phase8_7e3_nerf_real_scene/summary.json`
- `/tmp/phase8_7e3_nerf_real_scene/quality_evaluation.json`
- `/tmp/phase8_7e3_nerf_real_scene/heldout_contact_sheet.png`
- real Nerfstudio config/checkpoint/render artifacts under `worker-output/`.

Do not commit E3 until the real scene run and the small E2/E1/D3 backward gates are green.
