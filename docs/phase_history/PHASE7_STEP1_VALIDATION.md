# Phase 7 Step 1 — Interactive Benchmark Mode Validation

## Scope

This checklist validates only the interactive Benchmark Mode added in Phase 7 Step 1, plus a minimal Live 3D compatibility check. The complete application regression remains covered by the automated test suite.

## Preconditions

```bash
conda activate depth
cd /home/eiatech/Desktop/depth_3d_model_lab
```

All model dependencies and checkpoints needed for the selected real models must already be installed.

## Automated validation

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase7_step1_tests.sh quick
./scripts/run_phase7_step1_tests.sh full
```

## Focused manual validation

### 1. Open Benchmark Mode

```bash
python main.py
```

- Confirm **BENCHMARK** is enabled in the primary navigation.
- Confirm Stable Video, Snapshot, and Reconstruction remain disabled.
- Open Benchmark and confirm the page shows model selection, run options, progress, results, and log panels.
- Return once to Live 3D and confirm the existing page is unchanged.

### 2. Live-runtime safety gate

- In Live 3D, connect a source.
- Open Benchmark Mode.
- Confirm **Run Benchmark** is disabled and the page asks you to stop and disconnect Live 3D first.
- Return to Live 3D, stop inference if needed, and disconnect.
- Return to Benchmark and confirm the Run button is enabled.

### 3. Development Mock smoke run

- Select only **Development Mock**.
- Enable **Smoke test**.
- Run the benchmark.
- Confirm the UI stays responsive.
- Confirm progress changes from Starting to Running to Completed.
- Confirm the results table contains Development Mock with `PASSED`.
- Confirm JSON, CSV, Markdown summary, capability matrix, and hardware report files exist in a timestamped run directory.

### 4. One real-model run

- Select one lightweight installed production model, preferably MoGe-2 or Depth Anything V2.
- Disable Smoke test.
- Run with Auto profile and a 100,000 point budget.
- Confirm load time, warm-up, latency, FPS, Model VRAM, peak VRAM, output resolution, point count, scale, and intrinsics source are populated.
- Confirm metric and relative labels match the selected model's registry contract.

### 5. Two-model comparison

- Select two installed production models.
- Run one comparison.
- Confirm each model runs in sequence and receives its own result row.
- Confirm a failed or unavailable model does not prevent the next selected model from running unless **Require all** is enabled.

### 6. Input and option controls

- Run Development Mock with **Landscape only**, then **Portrait only**.
- Confirm the reported output changes between `640×384` and `384×640`.
- Change point budget and iterations and confirm the generated report records the selected values.
- For Video Depth Anything, confirm the sequence-frame value is recorded and multiple temporal frames are processed.

### 7. Cancellation

- Start a longer real-model benchmark.
- Press **Cancel** while a model is running.
- Confirm the status changes to Cancelling and then Cancelled.
- Confirm the application remains responsive.
- Confirm the active isolated model process terminates and a partial report is written when possible.
- Confirm a new benchmark can be started afterward.

### 8. Report access

- After a completed run, click **Open Results**.
- Confirm it opens the timestamped run directory rather than the shared output root.
- Confirm previous run directories are not overwritten by later runs.

### 9. Minimal compatibility check

- Return to Live 3D.
- Connect one image and run one inference with MoGe-2 or Development Mock.
- Confirm Live 3D still renders normally.
- Close the application and verify no benchmark process remains:

```bash
pgrep -af "run_phase6_benchmark.py|--worker-model"
```

No active benchmark parent or model worker should remain.
