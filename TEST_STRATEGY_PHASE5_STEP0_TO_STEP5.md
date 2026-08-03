# Phase 5 Test Strategy — Step 0 through Step 5

Phase 5 now covers the complete MoGe-2 runtime foundation: process isolation,
hardware detection, GPU profiles, live backpressure, metric geometry, telemetry,
model switching, failure recovery, stress hardening and automatic cross-GPU
tuning.

## Step 5 acceptance criteria

### RTX 2060 physical path

- Auto resolves to Efficient / ViT-S / FP16.
- Static and live point budgets resolve to 250K and 100K.
- Warm-up performs one 256 px pass.
- TF32 remains disabled; native SDPA remains enabled when available.
- Existing lifecycle, OOM, crash, source and memory tests remain green.

### RTX 5090 simulated path

- A 32 GB, compute-capability 12.0 device resolves to Ultra / ViT-L / FP16.
- Static and live point budgets resolve to 1M and 500K.
- Warm-up performs two 512 px passes.
- TF32 and native SDPA are enabled.
- No source-code or application-config edits are required.
- A CUDA build that cannot execute on the device falls back to the CPU-safe
  runtime and exposes a compatibility warning.

### Memory-pressure safety

- A 32 GB device with only 10 GB free resolves Auto to Balanced.
- A forced Ultra request is downgraded and reports memory pressure.
- Point-budget changes remain render/per-frame settings and do not reload model
  weights.

## Commands

```bash
python -m compileall -q app core tests scripts main.py

pytest -q \
  tests/unit/test_gpu_info.py \
  tests/unit/test_gpu_profiles.py \
  tests/unit/test_hardware_tuning_matrix.py \
  tests/unit/test_moge2_adapter.py \
  tests/unit/test_performance_formatting.py

QT_QPA_PLATFORM=offscreen pytest -q \
  tests/ui/test_hardware_profile_ui.py \
  tests/ui/test_main_window_shell.py

python scripts/check_hardware.py --simulate rtx2060
python scripts/check_hardware.py --simulate rtx5090
python scripts/check_hardware.py --simulate rtx5090-pressure

./scripts/run_phase5_tests.sh all
```

## Physical RTX 5090 commissioning later

The first run on the real RTX 5090 should only require the same repository and
launch command. Before loading MoGe-2, run `scripts/check_hardware.py
--assert-ready` to confirm that the installed PyTorch CUDA wheel can execute on
the detected compute capability. Then validate one image, one live source and a
five-cycle MoGe memory stress report. No profile or code edits should be made.
