# Phase 6 Step 6 — Marigold Validation

This checklist covers only the Marigold integration and the generic slow-model timeout behavior introduced in this step.

## Installation

```bash
conda activate depth
cd /home/eiatech/Desktop/depth_3d_model_lab

python scripts/install_marigold.py
python scripts/download_marigold_model.py
```

Verify the pinned local snapshot:

```bash
cat checkpoints/marigold/depth-v1-1/.ready
ls -lh checkpoints/marigold/depth-v1-1/model_index.json
```

## Compile and automated tests

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase6_step6_tests.sh quick
./scripts/run_phase6_step6_tests.sh full
```

## Focused manual validation

### 1. Registry and profile

Start the application and select **Marigold**.

RTX 2060 expected values:

- Variant: `Marigold Depth v1.1 · Fast`
- Detail: `512 px · 1 step(s)`
- Precision: `FP16`
- CPU offload enabled internally
- Output marked **Relative**, never metric or calibrated

RTX 5090 expected values:

- Variant: `Marigold Depth v1.1 · Precision`
- Detail includes `768 px`, `10 step(s)`, and `uncertainty`
- Precision: `FP16`
- CPU offload disabled

### 2. Image inference

Load one landscape image and one portrait image.

Confirm:

- Model loads and completes warm-up without an OOM.
- Depth map matches the input resolution.
- Point cloud is visible and correctly oriented.
- RGB point colors align with the source image.
- Depth is labelled relative/affine-invariant.
- Intrinsics are labelled estimated.
- No metric measurement claim is shown.

### 3. Deterministic result

Run the same image twice without changing the Marigold profile.

Confirm the depth result is visually stable because the adapter uses a fixed seed.

### 4. Slow-model Stop behavior

Start Marigold inference, then press **Stop** while it is still processing.

Confirm:

- The UI remains responsive.
- The application does not trigger a false 5-second worker-timeout recovery.
- The current inference is allowed to finish within the registry-defined bounded timeout.
- Stop is acknowledged afterward and no new frame is submitted.

This is bounded waiting, not mid-diffusion-step pre-emption.

### 5. Model switching and VRAM cleanup

Run:

```text
Marigold → Development Mock → Marigold
```

Confirm:

- Marigold unloads without a worker crash.
- Header VRAM falls near the non-model baseline after switching to Mock.
- Reloading Marigold does not increase the stable peak on each cycle.
- Repeat the cycle five times.

### 6. Optional uncertainty validation

On RTX 5090 Ultra/Precision profile:

- Confirm the uncertainty output is present.
- Confirm it has the same resolution as the depth map.
- Confirm it is described as uncertainty, not probability or calibrated confidence.

### 7. Error handling

Temporarily rename the snapshot marker:

```bash
mv checkpoints/marigold/depth-v1-1/.ready \
   checkpoints/marigold/depth-v1-1/.ready.disabled
```

Try loading Marigold. Confirm the application reports a missing/incomplete Marigold snapshot while preserving the connected source.

Restore it:

```bash
mv checkpoints/marigold/depth-v1-1/.ready.disabled \
   checkpoints/marigold/depth-v1-1/.ready
```

### 8. Clean shutdown

Close the application after one successful Marigold inference.

```bash
pgrep -af "depth_3d_model_lab|core.inference.worker"
```

No inference worker should remain.

## Commit gate

Commit only after the quick gate, full gate, and applicable focused manual scenarios pass.

```bash
git add -A
git commit -m "feat(models): integrate Marigold diffusion depth adapter"
```
