# Phase 6 Step 4 — Metric3D v2 Validation

Do not commit until compile validation, the quick gate, the full gate, and the applicable manual scenarios pass.

## 1. Installation validation

```bash
conda activate depth
cd /home/eiatech/Desktop/depth_3d_model_lab

python scripts/install_metric3d_v2.py
python scripts/download_metric3d_v2_models.py --variant auto
```

Expected on RTX 2060:

```text
Automatic GPU selection: small
checkpoints/metric3d-v2/metric_depth_vit_small_800k.pth
```

Expected on RTX 5090:

```text
Automatic GPU selection: large
checkpoints/metric3d-v2/metric_depth_vit_large_800k.pth
```

Verify:

```bash
cat third_party/Metric3D/.eia_metric3d_v2_ready.json
cat checkpoints/metric3d-v2/.ready
find checkpoints/metric3d-v2 -maxdepth 1 -type f -ls
```

Pass criteria:

- The source marker exists and records the pinned official commit.
- The selected checkpoint exists and is not empty.
- The existing PyTorch/CUDA build remains unchanged.
- `python -c "import torch, mmengine, mmcv, timm; print(torch.__version__, torch.version.cuda)"` succeeds.

## 2. Compile and automated validation

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase6_step4_tests.sh quick
./scripts/run_phase6_step4_tests.sh full
```

Pass criteria:

- No compilation errors.
- All focused adapter, registry, downloader, output-contract, and stress tests pass.
- All Phase 5 and previous Phase 6 regression tests pass.
- No tests are skipped to hide a failure.
- No segmentation fault occurs; Qt suites remain process-isolated through the stable test runner.

## 3. Startup and registry scenarios

Start:

```bash
python main.py
```

Validate:

1. The application starts with MoGe-2 still selected by default.
2. Model order is: MoGe-2, Depth Pro, UniDepthV2, Metric3D v2, Development Mock.
3. Selecting Metric3D v2 does not import or load it until inference starts.
4. Metric3D v2 displays `METRIC + NORMALS` when active.
5. RTX 2060 displays `Metric3D v2 ViT-S`, `1064×616 · Metric + normals`, and `FP16`.
6. RTX 5090 displays `Metric3D v2 ViT-L`, `1064×616 · Metric + normals`, and `FP16`.
7. The selected model persists after closing and reopening the application.
8. Phase 5 version text and all existing controls remain unchanged.

## 4. Image scenarios

Run Metric3D v2 with:

- Landscape office image.
- Portrait image.
- Square image.
- Small image below 640 px.
- Full-HD image.
- Very wide image.
- Very tall image.
- Image with dark regions.
- Image with bright windows.
- Image containing reflective or textureless surfaces.

For every input verify:

- RGB orientation and colors are correct.
- Depth fills the complete original frame after padding removal.
- No black padding remains in the depth result.
- Depth contains finite positive values in valid regions.
- Surface normals render without channel or orientation errors.
- Point cloud is not mirrored or upside down.
- Point count respects the selected budget.
- Header model badge, precision, and VRAM update.
- Footer latency, model, precision, points, resolution, and model VRAM update.
- Results are identified as metric but based on estimated intrinsics.
- No calibrated measurement claim appears.

## 5. Video scenarios

Validate with a short and a long video:

1. Connect and start inference.
2. Pause and confirm no new results are produced.
3. Resume and confirm inference continues.
4. Stop and restart without reconnecting the source.
5. Seek or replay where supported.
6. Let a short video reach the end naturally.
7. Run a video for at least 10 minutes.

Pass criteria:

- UI remains responsive.
- No stale result appears after pause, stop, model change, or replay.
- Point clouds update with the current frame.
- Memory does not grow continuously.
- Worker remains alive unless deliberately recovered.

## 6. USB camera scenarios

1. Connect a valid USB camera.
2. Start, pause, resume, stop, and restart inference.
3. Disconnect while stopped.
4. Disconnect the physical camera while running, then reconnect.
5. Change from Camera to Image and back.

Pass criteria:

- Camera lifecycle remains stable.
- Source errors are reported cleanly.
- Metric3D v2 can be unloaded or switched after camera failure.
- No worker or capture process remains after closing.

## 7. RTSP scenarios

1. Connect a valid RTSP stream.
2. Start, pause, resume, stop, and restart inference.
3. Interrupt the network temporarily.
4. Restore the network and reconnect.
5. Switch models while the RTSP source remains connected.
6. Run for at least 10 minutes.

Pass criteria:

- The UI never blocks on network interruption.
- Stale frames from the old session are rejected.
- Recovery does not create duplicate workers.
- VRAM and RAM remain bounded.

## 8. Model lifecycle and switching matrix

With one image connected, test each transition:

- MoGe-2 → Metric3D v2.
- Depth Pro → Metric3D v2.
- UniDepthV2 → Metric3D v2.
- Metric3D v2 → Development Mock.
- Metric3D v2 → MoGe-2.
- Metric3D v2 → Depth Pro.
- Metric3D v2 → UniDepthV2.

Then repeat this full cycle ten times:

```text
MoGe-2 → Depth Pro → UniDepthV2 → Metric3D v2 → Mock → MoGe-2
```

Pass criteria:

- The source stays connected.
- Each model badge, variant, precision, latency, and point count changes correctly.
- Metric3D v2 returns to approximately the same VRAM level each cycle.
- Switching to Mock releases Metric3D v2 CUDA allocations near the normal baseline.
- VRAM does not increase cumulatively.
- No OOM, crash loop, timeout loop, or stale result occurs.

## 9. Geometry and semantic validation

For Metric3D v2 verify:

- `depth_scale` is `metric`.
- `intrinsics_source` is `estimated`.
- `confidence_kind` is `confidence`.
- Intrinsics contain positive `fx` and `fy`, and centered `cx` and `cy`.
- Confidence has the same H×W shape as depth.
- Validity mask has the same H×W shape and boolean type.
- Surface normals are finite and approximately unit length in valid areas.
- RGB point colors match the source image.
- Point stride and point budget remain consistent.
- Changing point budget changes density without rerunning model inference where the existing cached-render path applies.
- RGB Points, Depth Colors, and Surface Normals render modes work.

Because calibration is postponed, any future dimensions derived from this result must be labeled:

```text
Estimated — Not Calibrated
```

## 10. RTX 2060 validation

Expected automatic behavior:

- Efficient profile.
- ViT-Small + RAFT-4iter.
- FP16 mixed precision through CUDA autocast.
- Metric3D model weights and source input remain FP32 to avoid input/bias dtype mismatches.
- 250K static point budget.
- 100K live point budget.

Validate:

- Load and warm-up complete without OOM.
- No `Input type (float) and bias type (c10::Half)` error appears.
- One image inference succeeds.
- Video or camera inference remains stable.
- VRAM stays below the physical 6 GB limit with recovery headroom.
- Closing other GPU applications improves available headroom as expected.
- Unsafe Ultra selection falls back to Efficient.

## 11. RTX 5090 validation

Expected automatic behavior:

- Ultra profile.
- ViT-Large + RAFT-8iter.
- FP16 mixed precision through CUDA autocast.
- Metric3D model weights and source input remain FP32 for compatibility.
- TF32 enabled when supported.
- 1M static point budget.
- 500K live point budget.

Validate:

- ViT-L checkpoint is selected automatically.
- Load, warm-up, and inference succeed.
- High point budgets render without UI blocking.
- Performance telemetry reports the selected model and real VRAM values.
- Startup memory pressure can still trigger the existing safe fallback.

## 12. Error and recovery scenarios

Test separately:

### Missing dependencies

Temporarily use an environment without `mmengine`/`mmcv` or simulate availability tests.

Expected: clear `NOT INSTALLED`/missing-dependency state; MoGe-2, Depth Pro, UniDepthV2, and Mock remain usable.

### Missing source checkout

Temporarily rename `third_party/Metric3D`.

Expected: clear instruction to run `python scripts/install_metric3d_v2.py`.

### Missing checkpoint

Temporarily rename the selected `.pth` file.

Expected: clear instruction to run `python scripts/download_metric3d_v2_models.py --variant auto`.

### Corrupted checkpoint

Use a deliberately invalid temporary checkpoint.

Expected: model load fails cleanly, source remains connected, and another model can be selected.

### CUDA OOM

Create memory pressure or use the existing controlled OOM test path.

Expected: OOM is classified, CUDA cache is cleaned, no restart loop occurs, and Efficient can be retried.

### Worker crash and timeout

Use the existing recovery tests/manual hooks.

Expected: one controlled worker recreation, no duplicate process, no stale result, and preserved source connection.

## 13. Render and UI regression

Verify all Phase 5 controls:

- Original-source toolbar.
- Depth colormap and range controls.
- 3D orbit, pan, zoom, fit, reset, focus, and Escape restore.
- Grid and axes toggles.
- Background selection.
- Point size, density, and smoothing.
- Save/export foundations.
- Performance header and footer refresh after load, unload, inference, and model switching.

Known future refinement, not part of Step 4:

```text
Rename footer “VRAM Usage” to “Model VRAM” or “Process VRAM”.
```

## 14. Shutdown validation

Close the application while:

- Idle.
- Model loaded but not running.
- Running image inference.
- Running video/camera/RTSP inference.
- Paused.
- After a load failure.
- After a recoverable worker error.

Then run:

```bash
pgrep -af "depth_3d_model_lab|core.inference.worker"
```

Pass criteria:

- Application exits with code 0 in normal scenarios.
- No inference worker remains.
- No camera/RTSP capture process remains.
- No traceback or Qt segmentation fault appears.

## 15. Commit

Only after all applicable checks pass:

```bash
git status --short
git diff --stat
git add -A
git commit -m "feat(models): integrate Metric3D v2 metric geometry adapter"
```
