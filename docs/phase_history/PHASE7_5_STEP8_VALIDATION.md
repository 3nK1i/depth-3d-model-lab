# EIA TECH Depth & 3D Model Lab
## Phase 7.5 Step 8 — Measure & Inspect UI, Manual Correction and Recompute

## Baseline

- Authoritative input archive: `depth_3d_model_lab_phase7_5_step7_committed.zip`
- User-provided commit: `101e4e412f4811959a0dd1318ae2cb15afa6d9c9`
- Input archive checksum: verified before implementation
- Application version: `0.7.0`
- Release name: `Phase 7 Stable`
- Reconstruction: remains disabled

## Architecture delivered

Step 8 adds a dedicated **Measure & Inspect** operational mode. It does not place measurement controls over the existing Live 3D workflow and does not add model-specific behavior to the existing depth or perception workers.

The workflow is:

```text
Current source frame
→ release any resident Live 3D depth model
→ isolated QProcess inspection job
→ perception and depth executed sequentially
→ alignment
→ object-specific point extraction
→ robust fitting
→ measurements and uncertainty
→ temporary validated inspection bundle
→ RGB/mask inspection + object-only 3D view + measurement panel
```

Manual mask edits use cached aligned depth and intrinsics. They rerun Steps 4–6 only; perception and depth models are not rerun. An approved manual mask remains authoritative. **Reset Mask** restores the original automatic aligned mask from the trusted temporary state before recomputing.

Operator box selection is generic and model-independent:

- Auto/preferred
- Axis-aligned
- Camera-aligned
- Oriented

The selected method is passed through an append-only measurement policy field and is recorded in measurement diagnostics.

## UI delivered

The new page includes:

- Registry-driven perception-model selection
- Efficient/Balanced/Ultra profile selection
- Maximum-object count
- Current-frame capture
- Isolated analysis progress and cancellation
- Object list and object selection
- Aligned RGB with masks, projected 3D boxes, labels and uncertainty
- Interactive object-only VisPy point-cloud inspection
- Raw/filtered point selection
- Box-method selection
- Include/exclude mask brush
- Non-destructive editing until explicit recompute
- Restore-automatic-mask workflow
- Per-session approved-result lock
- Width, height, depth, distance, centre, orientation and volume table
- Reliability, rejection reasons and uncertainty
- Depth/perception provenance, intrinsics source, calibration status and point counts
- Persistent `Estimated Uncertainty — Not Calibrated` notice

## GPU/resource behavior

Before starting inspection, the application refuses active/transient Live 3D states and releases any resident depth model without disconnecting the source. The isolated job then runs perception and depth sequentially. This avoids unnecessary model co-residency on the RTX 2060 6 GB while remaining suitable for the RTX 5090.

## Files added

```text
PHASE7_5_STEP8_VALIDATION.md
app/measurement_inspection_controller.py
app/widgets/inspection_3d_canvas.py
app/widgets/inspection_canvas.py
app/widgets/inspection_page.py
core/inspection/__init__.py
core/inspection/bundle.py
core/inspection/contracts.py
scripts/run_measurement_inspection_job.py
scripts/run_measurement_inspection_smoke.py
scripts/run_phase7_5_step8_tests.sh
tests/inspection_helpers.py
tests/integration/test_measurement_inspection_job.py
tests/stress/test_inspection_bundle_stress.py
tests/ui/test_inspection_page.py
tests/ui/test_measurement_inspection_controller.py
tests/unit/test_inspection_bundle.py
tests/unit/test_inspection_contracts.py
tests/unit/test_measurement_box_override.py
tests/unit/test_phase75_step8_backward_compatibility.py
```

## Files modified

```text
app/inference_controller.py
app/main_window.py
app/theme.py
app/widgets/mode_tabs.py
core/measurement/geometry.py
core/measurement/pipeline.py
core/measurement/policy.py
tests/conftest.py
tests/ui/test_main_window_shell.py
tests/ui/test_phase7_mode_switching.py
tests/unit/test_phase75_step1_backward_compatibility.py
tests/unit/test_phase75_step2_backward_compatibility.py
```

The Step 1/2 compatibility tests were updated only because Step 8 intentionally enables the new Measure & Inspect mode. They continue to require Reconstruction to remain disabled and the release identity to remain `0.7.0 — Phase 7 Stable`.

## Files removed

None.

## Dependency changes

None. The feature uses dependencies already present in the project.

## Validation actually executed in the delivery environment

Delivery environment limitations:

- Python 3.13.5
- PyQt6 unavailable
- VisPy unavailable
- No CUDA GPU

Therefore real Qt/VisPy application rendering and GPU inference are **not claimed** here.

Executed successfully:

```text
python -m compileall -q app core scripts tests                       PASS
Step 8 focused gate                                                   10 passed
Step 8 spawned-job integration                                        1 passed
Step 8 bundle stress sanity                                            1 passed
Step 1 focused regression                                             39 passed
Step 2 focused regression                                             34 passed
Step 3 focused regression                                             39 passed
Step 4 focused regression                                             32 passed
Step 5 focused regression                                             21 passed
Step 6 focused regression                                             20 passed
Step 7 focused regression                                             21 passed
Critical depth/perception/alignment/geometry/measurement/tracking     106 passed
```

The spawned integration test validates:

- Initial mock inspection job
- Bundle serialization and loading
- Object-point output
- Approved manual mask revision
- Manual provenance and revision
- Operator-selected box method
- Safe reset to the original automatic mask

Standalone smoke executed successfully:

```text
status=success bundle_status=rejected objects=1
```

The bundle is intentionally `rejected` for metric measurements because Development Mock depth is synthetic/nonmetric. Extraction and fitting remain available; the smoke validates inspection-bundle and rendering behavior without making a false metric claim.

## Installation

From the repository root after extracting the complete-files ZIP:

```bash
chmod +x \
  scripts/run_measurement_inspection_job.py \
  scripts/run_measurement_inspection_smoke.py \
  scripts/run_phase7_5_step8_tests.sh
```

No additional Pip or Conda installation is required.

## Compile validation

```bash
python -m compileall -q app core scripts tests
```

## Automated gates

Focused contracts and compatibility:

```bash
./scripts/run_phase7_5_step8_tests.sh quick
```

Isolated spawned process and manual-mask lifecycle:

```bash
./scripts/run_phase7_5_step8_tests.sh integration
```

Focused Qt page/controller/navigation gate:

```bash
./scripts/run_phase7_5_step8_tests.sh ui
```

Full split-process regression:

```bash
./scripts/run_phase7_5_step8_tests.sh full
```

Full mode runs independently:

```text
tests/unit
tests/ui
tests/integration
tests/stress
```

## Standalone smoke

```bash
python scripts/run_measurement_inspection_smoke.py \
  --output /tmp/phase7_5_step8_inspection.png \
  --bundle-dir /tmp/phase7_5_step8_inspection_bundle
```

Open:

```bash
google-chrome /tmp/phase7_5_step8_inspection.png
```

## Focused desktop/manual validation

Run in the configured Python 3.11/3.12 `depth` environment:

```bash
python main.py
```

1. Load or connect an Image, Video, USB camera or RTSP source in Live 3D.
2. Select a metric-capable depth model such as Depth Pro for measurement testing.
3. Switch to **Measure & Inspect**.
4. Click **Capture Current Frame**.
5. Select Ultralytics YOLO Segmentation, `Efficient`, and an appropriate object count.
6. Click **Analyze Frame**.
7. Confirm the progress stages complete without UI blocking.
8. Select objects from the list and by clicking the RGB image.
9. Toggle masks, 3D boxes, labels and uncertainty independently.
10. Switch between raw and filtered object points.
11. Switch Auto, axis-aligned, camera-aligned and oriented boxes.
12. Use Include/Exclude brush on one selected object.
13. Click **Apply & Recompute** and confirm:
    - the manual mask becomes approved and authoritative;
    - depth/perception models are not rerun;
    - points, boxes and measurements update.
14. Click **Reset Mask**, then **Apply & Recompute**, and confirm the original automatic mask returns.
15. Lock one approved result and confirm editing/recompute is disabled until unlocked.
16. Confirm every displayed available value includes uncertainty and reliability.
17. Confirm the wording `Estimated Uncertainty — Not Calibrated` remains visible.
18. Confirm Reconstruction remains disabled.
19. Return to Live 3D and run Development Mock once.
20. Run one Snapshot export.
21. Close the application and confirm no child worker remains.

Process check:

```bash
ps aux | grep -E \
  'main.py|inference_worker|perception_worker|run_measurement_inspection_job|depth_3d_model_lab' \
  | grep -v grep
```

## Acceptance conditions

- No model-specific branches were added to generic workers or controllers.
- Existing Phase 7 pages remain available and unchanged in purpose.
- Reconstruction remains disabled.
- The inspection job is isolated and cancellable.
- Live 3D resident models are released before the isolated GPU job.
- Manual correction never silently overwrites the automatic mask.
- Approved manual masks are explicit and revisioned.
- Reset restores the original automatic mask.
- Measurement eligibility remains per measurement kind.
- Detection confidence is never presented as measurement accuracy.
- Relative or synthetic depth cannot produce metric dimensions without valid scale recovery.
- Version remains `0.7.0`.

## Commit after all target-environment gates and manual checks pass

```bash
git status --short
git diff --check
git add -A
git diff --cached --check
git diff --cached --name-only
git commit -m "feat(inspection): add Phase 7.5 measurement inspection workflow"
git push
```

No version bump, tag or release is required for Step 8.
