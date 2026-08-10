# Phase 8.7D3 Step 3 Validation — Gaussian Performance Benchmarking

## Baseline

Apply only on:

- branch: `phase8-reconstruction`
- HEAD: `0ef21bd85331c4b5786da01b70d0ed1ef01b9b44`
- clean working tree synchronized with `origin/phase8-reconstruction`
- Phase 8.7D3 Step 2 held-out quality benchmarking committed and pushed

## Step 3 scope

Step 3 extends the existing D3 held-out benchmark with Gaussian-backend performance evidence only:

- peak training VRAM;
- peak render VRAM;
- aggregate peak VRAM;
- training / optimization wall-clock duration;
- backend-native warm render duration and FPS;
- GPU identity / UUID;
- GPU total VRAM;
- NVIDIA driver version;
- backend CUDA runtime version;
- CUDA compute capability;
- real RTX 2060 6 GB hardware-profile validation;
- RTX 5090 32 GB readiness/profile contract without fabricated measurements.

It does **not** add engine-level recovery, checkpoint/resume, full-app GPU profiling,
UI activation, NeRF, Depth Anything 3, or new reconstruction algorithms.

## Measurement methodology

### VRAM

Step 3 measures the actual isolated backend process tree with `nvidia-smi`.

- gsplat: benchmark worker process;
- Graphdeco: benchmark worker plus spawned `train.py` child during training;
- Graphdeco native renderer is loaded inside the isolated benchmark worker for render throughput;
- unrelated GPU processes are excluded by PID/process-tree filtering;
- training and render phases are measured independently.

This deliberately avoids using production-process VRAM as a proxy.

### Render FPS

Quality images are still generated through the already validated Step 2 path.
Performance timing is separate from quality-image save I/O.

For both backends Step 3 measures a warmed native render loop over the held-out cameras.
The `render_repeats` count is recorded in the benchmark evidence so the FPS value is auditable.

### Hardware profiles

`rtx_2060_6gb` is a real measurement gate on the current machine.

`rtx_5090_32gb` is defined as a readiness/identity contract only. Step 3 refuses to claim an RTX 5090 measurement unless the observed GPU actually matches that profile. Real RTX 5090 measurements are collected later on RTX 5090 hardware; full-application 2060/4090/5090 profiling remains Phase 8.9D.

## Backward compatibility

Running `scripts/run_gaussian_real_scene_benchmark.py` **without** `--measure-performance`
preserves the committed Step 2 behavior:

- held-out quality evaluation only;
- `step2_summary.json`;
- partial training-duration evidence;
- final `comparison_ready=false`.

Step 3 is explicit via `--measure-performance` and writes `step3_summary.json`.

## Isolation / authority rules

- production `depth` environment remains unchanged;
- D2 backend workers remain unchanged;
- Graphdeco and gsplat remain isolated external backends;
- `nvidia-smi` performance sampling lives only in D3 benchmark workers;
- production core receives dependency-light result contracts only;
- Gaussian representation remains appearance-only;
- `canonical_geometry_authoritative=false`;
- `certified_metrology=false`;
- Reconstruction UI remains disabled.

## Automated validation

Run:

```bash
git diff --check
./scripts/run_phase8_7d3_tests.sh quick
./scripts/run_phase8_7d3_tests.sh integration
./scripts/run_phase8_7d3_tests.sh stress
./scripts/run_phase8_7d2_tests.sh quick
```

Expected focused counts for this delivery:

- D3 quick: **34 passed**
- D3 integration: **3 passed**
- D3 stress: **3 passed**
- D2 backward quick: **31 passed**

## Real RTX 2060 performance gate

The isolated Graphdeco, gsplat, and metrics environments from D2/Step 2 must already be ready.

Run:

```bash
rm -rf /tmp/phase8_7d3_gaussian_performance_smoke

python scripts/run_gaussian_performance_smoke.py \
  --backend both \
  --training-steps 64 \
  --render-repeats 3 \
  --hardware-profile rtx_2060_6gb
```

The smoke must report finite positive performance values for both backends and finish with:

```text
Phase 8.7D3 Step 3 Gaussian performance benchmark: PASS
quality_comparison_ready=True performance_metrics_complete=True comparison_ready=True
canonical_geometry_authoritative=False certified_metrology=False
Phase 8.7D3 Gaussian performance smoke: PASS
```

Inspect:

```bash
cat /tmp/phase8_7d3_gaussian_performance_smoke/benchmark/step3_summary.json
```

For each backend verify:

- quality metrics remain measured;
- `peak_training_vram_mib > 0`;
- `peak_render_vram_mib > 0`;
- `training_duration_s > 0`;
- `render_duration_s > 0`;
- `render_fps > 0`;
- `measurement_scope=backend_process_tree_nvidia_smi`;
- `render_measurement_mode=backend_native_warm`;
- hardware evidence is complete;
- `profile_compatible=true`;
- current measured profile is `rtx_2060_6gb`;
- RTX 5090 appears only as a readiness profile, not as measured performance.

## Real-scene Step 3 command

For a real neural-scene manifest and matching Gaussian seed:

```bash
python scripts/run_gaussian_real_scene_benchmark.py \
  --dataset-manifest /ABSOLUTE/PATH/dataset.json \
  --seed /ABSOLUTE/PATH/gaussian_seed.npz \
  --validation-frame-id <HELD_OUT_FRAME_ID> \
  --backend both \
  --profile efficient \
  --measure-performance \
  --render-repeats 3 \
  --hardware-profile rtx_2060_6gb
```

Use multiple `--validation-frame-id` arguments when multiple held-out cameras are available.

Do not commit Step 3 until the focused automated gates and real both-backend RTX 2060 performance smoke are green.
