# Phase 5 Release Test Strategy

Release: **0.5.0 — Phase 5 Stable**

## Release contract

The Phase 5 baseline is accepted only when all required checks below pass:

- Python 3.11 or 3.12
- required project files and dependencies present
- runtime, `pyproject.toml`, config and visible UI version synchronized
- physical CUDA execution probe passes on the target GPU
- automatic RTX 2060 profile resolves to Efficient / ViT-S / FP16
- simulated RTX 5090 resolves to Ultra / ViT-L / FP16
- simulated RTX 5090 memory pressure resolves safely to Balanced / ViT-B
- unit, UI, spawned-worker integration and stress tests pass
- repeated model lifecycle shows bounded CPU/GPU memory
- workers, queues, pipes, camera probes and Qt threads exit cleanly
- application and worker diagnostics are written to separate rotating logs

## Automated gate

Quick development gate:

```bash
./scripts/run_phase5_release.sh quick
```

Complete release gate:

```bash
./scripts/run_phase5_release.sh full
```

Complete gate plus physical MoGe memory cycling:

```bash
./scripts/run_phase5_release.sh full --moge-stress-cycles 5
```

The atomic machine-readable result is written to:

```text
outputs/release/phase5_release_report.json
```

## Direct test modes

```bash
./scripts/run_phase5_tests.sh smoke
./scripts/run_phase5_tests.sh unit
./scripts/run_phase5_tests.sh ui
./scripts/run_phase5_tests.sh integration
./scripts/run_phase5_tests.sh stress
./scripts/run_phase5_tests.sh all
```

Qt tests run with the non-OpenGL test double. Real VisPy rendering remains part
of desktop manual validation because the Qt offscreen plugin cannot reliably
provide a native OpenGL framebuffer on every Mesa/driver combination.

## Physical RTX 2060 release checks

```bash
python scripts/check_hardware.py --assert-ready --require-cuda
```

Expected automatic profile:

```text
Efficient
MoGe-2 ViT-S Normal
FP16
level 5 / 1200 tokens
250,000 static / 100,000 live points
1 x 256 px warm-up
TF32 disabled
native SDPA enabled when available
```

Memory gate:

```bash
python scripts/run_memory_stress.py \
  --model moge2 \
  --profile auto \
  --cycles 12 \
  --settle-ms 1000 \
  --strict-device-vram \
  --output outputs/release/moge2_release_stress.json
```

Required booleans:

```text
same_worker_process
no_crash_signal
no_command_timeout
worker_rss_bounded
unloaded_allocator_reserved_bounded
device_vram_bounded
worker_exited_after_shutdown
```

## RTX 5090 readiness without physical hardware

```bash
python scripts/check_hardware.py --simulate rtx5090
python scripts/check_hardware.py --simulate rtx5090-pressure
```

Expected normal path:

```text
Ultra / ViT-L / FP16
level 9 / 2500 tokens
1,000,000 static / 500,000 live points
2 x 512 px warm-up
TF32 and SDPA enabled when supported
```

Expected pressure path at 10 GB startup free VRAM:

```text
Balanced / ViT-B / FP16
500,000 static / 250,000 live points
```

On the physical RTX 5090, run the same release gate. No source or configuration
file changes are permitted. The only external prerequisite is a compatible
NVIDIA driver and CUDA-enabled PyTorch build; the active CUDA probe must pass
before loading ViT-L.

## Manual end-to-end matrix

Validate each source independently:

```text
Image
Video
USB camera
RTSP
```

For every available source:

1. Connect and verify preview/status.
2. Start MoGe-2 and verify depth plus 3D geometry.
3. Pause/resume where applicable.
4. Stop without losing the source.
5. Switch MoGe-2 to Development Mock and back.
6. Change source type and confirm stale RGB/depth/geometry is cleared.
7. Exercise Source, Depth and 3D focus modes and exit with Esc.
8. Close the application while idle, loading, running and switching models.

After every close:

```bash
ps -eo pid,ppid,args | \
  grep -E "depth_3d_model_lab/main.py|multiprocessing.spawn" | \
  grep -v grep
```

Expected: no remaining application or inference worker.

## Failure-recovery matrix

- offline uncached checkpoint preserves source and allows Mock fallback
- missing/corrupted checkpoint returns a classified retryable failure
- CUDA OOM restarts only the worker and does not auto-loop inference
- killed worker is reaped, replaced once and reloads the requested model
- command/result timeout never leaves controls permanently locked
- late results and acknowledgements are ignored after shutdown/restart

## Diagnostic evidence

Review after manual validation:

```text
logs/app.log
logs/worker.log
outputs/release/phase5_release_report.json
outputs/release/moge2_release_stress.json (when requested)
```

Do not commit generated logs or reports.
