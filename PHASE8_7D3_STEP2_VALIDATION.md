# Phase 8.7D3 Step 2 Validation — Held-Out Rendering + PSNR / SSIM / LPIPS

## Baseline

Apply only on:

- branch: `phase8-reconstruction`
- HEAD: `14a697cb22db636847184762e57caf641acd33e4`
- clean working tree synchronized with `origin/phase8-reconstruction`
- Phase 8.7D3 Step 1 benchmark contracts committed and pushed

## Step 2 scope

Step 2 adds real held-out-view execution for the two D2-qualified external Gaussian backends and one common backend-neutral image-quality evaluator.

It covers:

- loading a real `NeuralSceneDataset` manifest back into the canonical contract;
- training on the Step 1 training split only;
- rendering every held-out validation view exactly once;
- PSNR on RGB values in `[0,1]` using MSE with a `1e-12` numerical floor;
- SSIM using the same 11-pixel Gaussian-window family used by the reference 3DGS training code (`sigma=1.5`, RGB, unit range);
- LPIPS AlexNet metric version `0.1`, package `lpips==0.1.4`;
- per-view metrics plus an arithmetic held-out-view mean;
- strict benchmark/dataset/split fingerprint propagation.

Step 2 deliberately does **not** complete the performance benchmark. It records training duration as partial evidence, but peak VRAM and render FPS remain unmeasured until Step 3. Therefore:

- `quality_comparison_ready=true` is possible in Step 2;
- final `comparison_ready` must remain `false` until Step 3.

## Isolation / safety

- no `lpips`, torch image metric, Graphdeco, or gsplat import is added to production `core/reconstruction`;
- `lpips` is installed only in isolated `eia-gaussian-metrics`;
- the production `depth` environment is not modified;
- D2 Graphdeco and gsplat workers are not modified;
- Step 2 uses new external benchmark-worker scripts;
- the pinned Graphdeco revision reads `test.txt` when its COLMAP reader receives `llffhold=0`; the audited `explicit-test-txt-heldout-split` compatibility patch forces that mode for `--eval` without changing the pinned upstream revision;
- Graphdeco benchmark execution fails closed unless both upstream `test.txt` support and the audited held-out split patch are present;
- gsplat optimizes only `training_frame_ids` and renders only `validation_frame_ids`;
- held-out render manifests must cover every validation view exactly once;
- metric image dimensions must match the canonical frame metadata;
- Gaussian output remains appearance-only and can never become canonical geometry or certified metrology;
- Reconstruction UI remains disabled.

## Metrics environment

Create one separate CPU image-quality environment:

```bash
./scripts/setup_gaussian_metrics.sh
```

Pinned runtime:

- Python 3.10
- PyTorch 2.7.1 CPU
- torchvision 0.22.1 CPU
- lpips 0.1.4
- LPIPS AlexNet metric version 0.1

The setup instantiates LPIPS once so its standard pretrained AlexNet weights are resolved/cached before a benchmark run.

Probe it with:

```bash
python scripts/check_gaussian_metrics_environment.py \
  --python /home/eiatech/anaconda3/envs/eia-gaussian-metrics/bin/python
```

## Files

This Step 2 overlay contains exactly 17 files:

1. `PHASE8_7D3_STEP2_VALIDATION.md`
2. `core/reconstruction/__init__.py`
3. `core/reconstruction/neural_scene.py`
4. `core/reconstruction/gaussian_benchmarking.py`
5. `scripts/check_gaussian_metrics_environment.py`
6. `scripts/setup_gaussian_metrics.sh`
7. `scripts/gaussian_benchmark_workers/common.py`
8. `scripts/gaussian_benchmark_workers/gsplat_worker.py`
9. `scripts/gaussian_benchmark_workers/graphdeco_worker.py`
10. `scripts/gaussian_benchmark_workers/metrics_worker.py`
11. `scripts/run_gaussian_real_scene_benchmark.py`
12. `scripts/run_gaussian_heldout_quality_smoke.py`
13. `scripts/run_phase8_7d3_tests.sh`
14. `tests/unit/test_reconstruction_gaussian_benchmark_quality.py`
15. `tests/unit/test_phase8_7d3_step2_backward_compatibility.py`
16. `tests/integration/test_reconstruction_gaussian_heldout_quality.py`
17. `tests/stress/test_reconstruction_gaussian_heldout_quality_stress.py`

## Validation order

```bash
git diff --check
./scripts/run_phase8_7d3_tests.sh quick
./scripts/run_phase8_7d3_tests.sh integration
./scripts/run_phase8_7d3_tests.sh stress
./scripts/run_phase8_7d2_tests.sh quick
```

Then set up/probe the isolated metrics environment and run the real CUDA held-out smoke:

```bash
./scripts/setup_gaussian_metrics.sh

python scripts/check_gaussian_metrics_environment.py \
  --python /home/eiatech/anaconda3/envs/eia-gaussian-metrics/bin/python

python scripts/run_gaussian_heldout_quality_smoke.py \
  --backend both \
  --training-steps 64
```

The smoke must report:

- both real Gaussian workers complete;
- one held-out validation view per backend;
- finite PSNR / SSIM / LPIPS for both backends;
- `quality_comparison_ready=True`;
- `comparison_ready=False` because Step 3 performance evidence is intentionally incomplete;
- canonical geometry authority remains false for Gaussian output;
- certified metrology remains false.

## Real-scene command

Once a real neural-scene manifest and matching Gaussian seed are available, run:

```bash
python scripts/run_gaussian_real_scene_benchmark.py \
  --dataset-manifest /ABSOLUTE/PATH/dataset.json \
  --seed /ABSOLUTE/PATH/gaussian_seed.npz \
  --validation-frame-id <HELD_OUT_FRAME_ID> \
  --backend both \
  --profile efficient \
  --hardware-profile rtx_2060_6gb
```

Use multiple `--validation-frame-id` arguments for multiple held-out views.

Do not commit Step 2 until the focused tests, D2 backward quick gate, isolated metrics probe, and real both-backend held-out smoke are all green.

## Pinned Graphdeco held-out split compatibility

Pinned revision `54c035f7834b564019656c3e3fcc3646292f727d` contains explicit `test.txt` support in `scene/dataset_readers.py` when `llffhold=0`, but does not expose `--llffhold` through its CLI model arguments. Step 2 therefore applies the audited repository patch `scripts/gaussian_patches/graphdeco_explicit_test_split.patch`, which makes COLMAP evaluation call the existing reader with `llffhold=0`. The D3 benchmark worker passes only the supported `--eval` flag and verifies the patch before training.

### Graphdeco explicit-test-list path correction

The pinned Graphdeco reader opens the explicit evaluation split from `sparse/0/test.txt` when `llffhold=0`. The Step 2 benchmark exporter therefore materializes the held-out image-name list at exactly `benchmark-dataset/sparse/0/test.txt`. A focused unit regression test materializes a synthetic COLMAP benchmark dataset and verifies both the exact split path and validation-image content before any real Graphdeco training is attempted.
