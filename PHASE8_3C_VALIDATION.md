# Phase 8.3C Validation — Real External PyCOLMAP Sparse SfM

Phase 8.3C adds the first real offline sparse-reconstruction backend. PyCOLMAP
4.1.1 remains outside the main `depth` environment and executes only in a
standalone external worker process. Reconstruction mode remains disabled.

## Safety invariants

- Work only on `phase8-reconstruction`.
- Do not install `pycolmap-cuda12` in `depth`.
- `depth` remains NumPy 1.26.4 / PyTorch 2.13 + CUDA 13.
- The external reconstruction interpreter owns PyCOLMAP 4.1.1 / CUDA 12.x.
- Native PyCOLMAP objects never cross into the application process.
- Raw monocular SfM stays `arbitrary` scale with `unit_to_metre = null`.
- COLMAP world-to-camera poses are inverted before becoming
  `T_reconstruction_from_camera`.
- Reconstruction UI remains disabled.

## 1. Compile

```bash
python -m compileall -q \
  core/reconstruction \
  scripts/reconstruction_pycolmap_worker.py \
  scripts/run_reconstruction_pycolmap_smoke.py
```

## 2. Phase 8.3C automated gates

```bash
./scripts/run_phase8_3c_tests.sh quick
./scripts/run_phase8_3c_tests.sh integration
```

Then run backward-compatibility gates:

```bash
./scripts/run_phase8_3b_tests.sh quick
./scripts/run_phase8_2_tests.sh quick
./scripts/run_phase8_1_tests.sh quick
```

Before commit, run the complete application suite:

```bash
./scripts/run_phase8_3c_tests.sh full
```

The two existing optional target-hardware perception skips remain acceptable.
No other failures are expected.

## 3. Verify dependency isolation

From `depth`:

```bash
python - <<'PY'
import importlib.util
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("pycolmap:", "installed" if importlib.util.find_spec("pycolmap") else "not installed")
PY
```

Expected: PyCOLMAP is **not installed**.

Verify the external environment:

```bash
/home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python - <<'PY'
import pycolmap
print("PyCOLMAP:", pycolmap.__version__)
print("CUDA devices:", pycolmap.get_num_cuda_devices())
PY
```

Expected: PyCOLMAP `4.1.1` and at least one CUDA device on the RTX machine.

Also verify that the application package contains no direct PyCOLMAP import:

```bash
grep -RInE '^[[:space:]]*(from[[:space:]]+pycolmap|import[[:space:]]+pycolmap)' \
  core/reconstruction \
  && echo "ERROR: direct PyCOLMAP import found" \
  || echo "PASS: main reconstruction package is PyCOLMAP-free"
```

## 4. Real image-sequence smoke

Use a real ordered image sequence with substantial overlap. For a first smoke,
20–80 sharp images of one static scene are sufficient. Avoid duplicate frames,
large jumps, moving objects dominating the view, and severe blur.

On the RTX 2060 start with Efficient + sequential matching:

```bash
rm -rf /tmp/phase8_3c_real_sfm

python scripts/run_reconstruction_pycolmap_smoke.py \
  --images ~/Desktop/datasets/YOUR_SEQUENCE \
  --sfm-python /home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python \
  --profile efficient \
  --matching sequential \
  --max-keyframes 60 \
  --output-dir /tmp/phase8_3c_real_sfm
```

For a small **unordered** photo collection, use `--matching exhaustive` instead.

A successful run ends with:

```text
Phase 8.3C real PyCOLMAP smoke: PASS
real_sfm_executed=True
gauge=arbitrary unit_to_metre=None
```

The exact camera/landmark counts depend on the real scene and are not fixed.

Inspect the report:

```bash
python -m json.tool \
  /tmp/phase8_3c_real_sfm/real_sfm_report.json \
  | less
```

Important fields:

- `real_sfm_executed: true`
- external `pycolmap_version: 4.1.1`
- `gauge.scale_status: arbitrary`
- `gauge.unit_to_metre: null`
- at least 3 registered cameras
- at least 1 sparse landmark
- one or more preserved PyCOLMAP camera-model estimates

Sparse point cloud:

```text
/tmp/phase8_3c_real_sfm/real_sparse_arbitrary_scale.ply
```

Open it with a PLY-capable viewer such as CloudCompare or MeshLab if available.
The coordinates are deliberately **not metres**.

## 5. Optional recorded-video smoke

```bash
rm -rf /tmp/phase8_3c_real_video_sfm

python scripts/run_reconstruction_pycolmap_smoke.py \
  --video ~/Desktop/datasets/YOUR_VIDEO.mp4 \
  --sfm-python /home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python \
  --profile efficient \
  --matching sequential \
  --video-target-fps 2 \
  --max-keyframes 60 \
  --output-dir /tmp/phase8_3c_real_video_sfm
```

This validates Phase 8.2 video re-decoding -> durable workspace -> external real
SfM without enabling the application UI.

## 6. Small manual backward-compatibility sanity check

Start the application normally and confirm Reconstruction is still disabled.
Perform one existing image -> depth -> 3D sanity check. Full regression belongs
in the automated suite above; do not manually re-test the whole application.

## 7. Pre-commit inspection

```bash
git diff --check
git status --short
git diff --stat
```

After all validation succeeds:

```bash
git add -A
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

Generated workspaces, databases, reports, PLY files, caches, and model weights
must not be staged.

Recommended commit after all real-hardware validation passes:

```bash
git commit -m "feat(reconstruction): add isolated real PyCOLMAP sparse backend"
git push origin phase8-reconstruction
```

Do not create a release tag and do not merge to `main` during Phase 8.3C.
