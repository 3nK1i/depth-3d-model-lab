# Depth & 3D Model Lab — Test Strategy

Scope: Phase 5 Step 0 through Step 4.3.

The pack covers the PyQt6 UI, spawned inference process, Development Mock,
MoGe-2 adapter contracts, latest-frame backpressure, source preservation,
structured failure recovery, repeated worker replacement, clean shutdown and
memory-stability diagnostics.

## Principles

1. Normal automated tests never download model weights.
2. Normal automated tests never require a physical camera, RTSP endpoint or GPU.
3. Real-process integration and stress tests use Development Mock on CPU.
4. QSettings is redirected to a temporary directory.
5. Offscreen UI tests replace only the VisPy OpenGL canvas with a Qt test double.
6. Real MoGe-2/CUDA/VRAM and visual geometry checks remain explicit manual runs.
7. Longer repeated loops are separate from the normal `full` development suite.

## Test commands

```bash
./scripts/run_phase5_tests.sh smoke
./scripts/run_phase5_tests.sh unit
./scripts/run_phase5_tests.sh ui
./scripts/run_phase5_tests.sh integration
./scripts/run_phase5_tests.sh full
```

Longer repeated lifecycle loops:

```bash
./scripts/run_phase5_tests.sh stress
```

Everything, including stress:

```bash
./scripts/run_phase5_tests.sh all
```

Coverage:

```bash
./scripts/run_phase5_tests.sh coverage
```

## Coverage map

### Step 0 — Runtime stabilization

- `tests/ui/test_state_machine.py`
- `tests/unit/test_session_tracker.py`
- `tests/ui/test_command_watchdog.py`
- `tests/ui/test_process_manager_unit.py`
- `tests/unit/test_worker_helpers.py`
- `tests/integration/test_inference_lifecycle.py`
- `tests/ui/test_main_window_shell.py`

Validates explicit application transitions, centralized runtime buttons,
session invalidation, latest-only queues, command acknowledgements, spawned
worker startup/restart/shutdown and stale-result rejection.

### Step 1 — Hardware detection and profiles

- `tests/unit/test_gpu_info.py`
- `tests/unit/test_gpu_profiles.py`
- `tests/ui/test_hardware_profile_ui.py`
- `tests/ui/test_settings_persistence.py`

Validates CPU fallback, NVIDIA parsing, RTX 2060 Efficient, RTX 5090 Ultra,
unsafe-profile downgrade, precision capability and persistent selections.

### Step 1.1–1.2 — Truthful metrics and live backpressure

- `tests/unit/test_performance_formatting.py`
- `tests/ui/test_performance_bar.py`
- `tests/unit/test_live_sampling.py`
- `tests/unit/test_geometry_sampling.py`
- `tests/unit/test_runtime_telemetry.py`
- `tests/ui/test_main_window_shell.py`

Validates active-model labels, point count, timing semantics, latest-frame
replacement, dropped-frame counters and hard point budgets.

### Step 2–2.1 — MoGe-2 and static/live source lifecycle

- `tests/unit/test_moge2_adapter.py`
- `tests/unit/test_geometry_projection.py`
- `tests/unit/test_geometry_filtering.py`
- `tests/unit/test_geometry_colors.py`
- `tests/unit/test_geometry_validity.py`
- `tests/unit/test_image_source_lifecycle.py`
- `tests/unit/test_capture_sources.py`
- `tests/ui/test_view_content_states.py`

MoGe tests inject a fake model and never access Hugging Face. Coverage includes
metric depth, direct points, masks, normals, intrinsics, CPU fallback, warm-up,
unload, static-image resend and continuous source behavior.

### Step 3–3.2 — Telemetry, viewport and media controls

- `tests/unit/test_runtime_telemetry.py`
- `tests/unit/test_temporal_smoothing.py`
- `tests/ui/test_viewport_controls.py`
- `tests/ui/test_viewport_layout.py`
- `tests/ui/test_settings_sidebar.py`
- `tests/ui/test_media_toolbars.py`
- `tests/ui/test_view_content_states.py`

Validates end-to-end/model/geometry timing, VRAM wording, viewport interactions,
cached rerendering, temporal smoothing, Source toolbar actions and synchronized
Depth controls.

### Step 4.1 — Explicit model lifecycle and safe switching

- `tests/unit/test_model_lifecycle.py`
- `tests/unit/test_runtime_messages.py`
- `tests/unit/test_mock_adapter.py`
- `tests/unit/test_moge2_adapter.py`
- `tests/ui/test_process_manager_unit.py`
- `tests/ui/test_main_window_shell.py`
- `tests/integration/test_inference_lifecycle.py`

Validates UNLOADED/LOADING/WARMING UP/READY/UNLOADING/FAILED, acknowledged
cleanup, same-process unload/reload, MoGe↔Mock switching, source preservation and
no reload for point-budget-only changes.

### Step 4.2 — Failure classification and bounded recovery

- `tests/unit/test_failure_classification.py`
- `tests/ui/test_command_watchdog.py`
- `tests/ui/test_process_manager_unit.py`
- `tests/ui/test_main_window_shell.py`
- `tests/integration/test_inference_lifecycle.py`

Validates structured OOM/network/auth/checkpoint/configuration/warm-up failures,
command/result timeouts, one-shot crash signals, worker replacement, retryable
source state and no automatic inference restart loops.

### Step 4.3 — Repeated lifecycle, memory and shutdown hardening

Fast diagnostics:

- `tests/unit/test_runtime_stress.py`
- `tests/ui/test_process_manager_stress.py`
- `tests/ui/test_shutdown_lifecycle.py`

Explicit repeated loops:

- `tests/stress/test_model_lifecycle_stress.py`

Coverage includes:

- repeated Mock load → infer → stop → unload in one worker;
- repeated fresh-worker generations and old-child reaping;
- terminate then kill escalation for a stubborn worker;
- closed pipe/queue handling without duplicate crash signals;
- shutdown from every application state;
- no extra Stop command during application close;
- idempotent close and ignored late Ack/result/stage/timeout/crash callbacks;
- JSON-serialisable RSS/VRAM stress reports.

## Command-line stress tools

### Development Mock lifecycle

```bash
python scripts/run_lifecycle_stress.py \
  --cycles 10 \
  --frames-per-cycle 2
```

Optional periodic process replacement:

```bash
python scripts/run_lifecycle_stress.py \
  --cycles 12 \
  --restart-every 3
```

### Forced worker crash/recovery

```bash
python scripts/run_worker_recovery_stress.py --cycles 5
```

This kills only the spawned inference worker, verifies one crash signal, starts a
fresh worker generation and confirms the old PID is gone.

### Model memory stability

Development Mock:

```bash
python scripts/run_memory_stress.py --model mock --cycles 10
```

MoGe-2 on the detected safe profile:

```bash
python scripts/run_memory_stress.py \
  --model moge2 \
  --profile auto \
  --cycles 5
```

Add `--strict-device-vram` only when other GPU applications are stable. The
stronger model-specific indicators are allocator allocated/reserved values after
unload and repeated tail drift. Device-wide VRAM includes Xorg, GNOME, browser,
VisPy and unrelated CUDA processes.

Reports are written to `outputs/stress/*.json` and include settings, per-cycle
samples, PID/generation history, RSS, cleanup VRAM and pass/fail checks.

## Manual validation matrix

Run normal tests first, then stress tests, then desktop checks without the Qt
offscreen environment.

### Static image / MoGe-2

- IMAGE → Browse → Connect → Start.
- Final state is RESULT READY; depth and point cloud are coherent.
- Start again reuses the resident model.
- Cached visual controls do not rerun inference.

### Video / camera / RTSP

- Preview and 3D stay responsive under latest-frame backpressure.
- Pause/Resume/Stop remain valid after repeated cycles.
- Source-type changes clear stale output but keep a healthy resident model.
- Camera and RTSP resources are released on disconnect/application close.

### Model switching and recovery

- Repeat MoGe-2 → Mock → MoGe-2 at least five times.
- Force one worker crash and confirm source preservation plus deliberate restart.
- Confirm a second failure does not create an infinite automatic recovery loop.
- Close the app during loading, running and model switching; no worker remains.

### RTX 2060 memory run

```bash
python scripts/run_memory_stress.py \
  --model moge2 \
  --profile auto \
  --cycles 5
```

Expected: Efficient/ViT-S/FP16, no OOM, bounded allocator memory after unload,
and no continuously rising tail trend.

### RTX 5090 memory run

Use Auto first. Ultra/ViT-L is allowed only when the installed PyTorch build is
compatible with the GPU architecture. FP16/BF16 comparison belongs to the next
hardware-tuning step.

## Clean checkpoint rule

Commit only after the normal and explicit stress suites pass:

```bash
git status
git diff --check
./scripts/run_phase5_tests.sh full
./scripts/run_phase5_tests.sh stress
git add -A
git commit -m "test: harden repeated model lifecycle and shutdown"
```
