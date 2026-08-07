# Phase 8.3D — Offline SfM hardening and finalization

Baseline: `phase8-reconstruction` at `30cc8532c165fc880b689e879b064c3f30b312d8`.

## Scope

Phase 8.3D keeps Reconstruction disabled and hardens the already validated real
PyCOLMAP sparse path. It adds:

- real sequential-matching validation support,
- recorded-video → keyframes → real SfM validation support,
- deterministic multi-model ranking using registered images, observations,
  landmarks, reprojection error, then model ID,
- conservative typed SfM quality assessment (`accepted`, `degraded`, `rejected`),
- workspace SHA-256 integrity validation before worker launch,
- deterministic request/workspace fingerprinting,
- stage checkpoint reuse for completed feature extraction / matching,
- safe rerun of mapping after cancellation, timeout, crash, or previous completion,
- atomic worker checkpoint/result writes,
- expanded provenance in result metadata,
- compact `real_sfm_summary.json` output for real validation,
- tests for checkpoint compatibility, model selection, quality, workspace mutation,
  recorded-video lifecycle, cancellation/failure/timeout, and backward compatibility.

The stage resume is deliberately conservative: it reuses a validated COLMAP
feature/matching database, then reruns sparse mapping into a fresh sparse output
directory. Native PyCOLMAP objects are never persisted across processes.

Raw monocular SfM remains arbitrary scale. `unit_to_metre` remains `None`.

## Automated validation performed during package creation

In the package environment:

- `compileall`: PASS
- Phase 8.3D quick: **76 passed**
- Phase 8.3D integration: **4 passed**
- Phase 8.3C quick regression: **63 passed**
- Phase 8.3B quick regression: **48 passed**
- Phase 8.2 quick regression: **58 passed**
- Phase 8.1 quick regression: **67 passed**
- reconstruction stress subset: **2 passed**
- external-worker dependency isolation inspection: PASS
- direct PyCOLMAP imports in `core/reconstruction`: NONE

The complete application suite must be run in the production `depth`
environment because the package environment does not contain PyQt6/VisPy.
Real PyCOLMAP validation must be run on the target machine using the isolated
`eia-pycolmap-probe` interpreter.

## Required target validation

### 1. Focused automated tests

```bash
./scripts/run_phase8_3d_tests.sh quick
./scripts/run_phase8_3d_tests.sh integration
```

Expected:

```text
76 passed
4 passed
```

### 2. Backward gates

```bash
./scripts/run_phase8_3c_tests.sh quick
./scripts/run_phase8_3b_tests.sh quick
./scripts/run_phase8_2_tests.sh quick
./scripts/run_phase8_1_tests.sh quick
```

Expected:

```text
63 passed
48 passed
58 passed
67 passed
```

### 3. Real ordered-image sequential matching + cancellation/resume

Use the existing COLMAP South Building images:

```bash
IMAGES_DIR="$HOME/Desktop/phase8_3c_south_building/south-building/images"

rm -rf /tmp/phase8_3d_sequential_resume

python scripts/run_reconstruction_pycolmap_smoke.py \
  --images "$IMAGES_DIR" \
  --sfm-python /home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python \
  --profile efficient \
  --matching sequential \
  --sequential-overlap 10 \
  --max-keyframes 60 \
  --exercise-resume \
  --output-dir /tmp/phase8_3d_sequential_resume
```

Required observations:

- the first worker attempt is cancelled after matching,
- the second attempt reports reuse of a matching-stage checkpoint,
- final smoke result is PASS,
- `real_sfm_executed=True`,
- `resume_from_stage=matching`,
- quality is not `rejected`,
- gauge remains `arbitrary`,
- `unit_to_metre=None`.

Compact result:

```text
/tmp/phase8_3d_sequential_resume/real_sfm_summary.json
```

### 4. Recorded-video → real sequential SfM

Create a deterministic test video from the known-good South Building images:

```bash
python - <<'PY'
from pathlib import Path
import cv2
from core.reconstruction import natural_path_key

src = Path.home() / "Desktop/phase8_3c_south_building/south-building/images"
out = Path.home() / "Desktop/phase8_3d_south_building.avi"
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
paths = sorted(
    (p for p in src.iterdir() if p.is_file() and p.suffix.lower() in extensions),
    key=natural_path_key,
)[:60]
if len(paths) < 20:
    raise SystemExit(f"Need at least 20 source images, found {len(paths)}")
first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
if first is None:
    raise SystemExit("Could not decode first source image")
h, w = first.shape[:2]
scale = min(1.0, 1280.0 / w)
out_w = max(2, int(round(w * scale)))
out_h = max(2, int(round(h * scale)))
out_w -= out_w % 2
out_h -= out_h % 2
writer = cv2.VideoWriter(
    str(out), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (out_w, out_h)
)
if not writer.isOpened():
    raise SystemExit("OpenCV MJPG VideoWriter is unavailable")
for path in paths:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not decode {path}")
    if image.shape[1] != out_w or image.shape[0] != out_h:
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    writer.write(image)
writer.release()
print(out)
PY
```

Run real video reconstruction:

```bash
rm -rf /tmp/phase8_3d_video_sfm

python scripts/run_reconstruction_pycolmap_smoke.py \
  --video "$HOME/Desktop/phase8_3d_south_building.avi" \
  --sfm-python /home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python \
  --profile efficient \
  --matching sequential \
  --sequential-overlap 10 \
  --video-target-fps 2.5 \
  --max-keyframes 30 \
  --output-dir /tmp/phase8_3d_video_sfm
```

Required observations:

- `source_kind=recorded_video` in the compact summary,
- real SfM PASS,
- at least three registered keyframes,
- nonzero sparse landmarks,
- quality is not `rejected`,
- arbitrary gauge is preserved.

Compact result:

```text
/tmp/phase8_3d_video_sfm/real_sfm_summary.json
```

### 5. Complete application regression

```bash
./scripts/run_phase8_3d_tests.sh full
```

Based on the Phase 8.3C baseline plus the new tests, the expected counts are:

```text
628 unit passed
185 UI passed
22 integration passed
2 expected optional hardware skips
38 stress passed
```

Zero failures are required.

### 6. Minimal manual sanity

Run:

```bash
python main.py
```

Only validate affected/backward-critical behavior:

- Reconstruction remains visible but disabled.
- Existing image → depth → 3D still works.

Do not repeat the entire manual application regression.
