# EIA TECH Depth & 3D Model Lab
## Phase 7.5 Step 1 — Segmentation-First Measurement Architecture and Data Contracts

Date: 2026-08-05  
Baseline: `v0.7.0 — Phase 7 Stable`  
Application version after this step: `0.7.0`  
Release name after this step: `Phase 7 Stable`

## 1. Architecture summary

Phase 7.5 Step 1 adds contract-only, dependency-light packages without changing the existing capture, depth inference, multiprocessing, operational-mode, UI, geometry-rendering, snapshot, stable-video, benchmark or reconstruction code.

The new architecture consists of:

- `core/contracts`: immutable frame references, image sizes, model provenance, processing provenance, pixel coordinate spaces, authoritative 2D transforms, 3D coordinate frames, rigid transforms and camera intrinsics.
- `core/perception`: independent perception metadata, GPU resource declarations, capability-specific adapter protocols, typed requests, detection normalization, extensible mask payloads, mask approval/revision provenance, object observations and an intentionally empty production registry.
- `core/measurement`: object-point-cloud, 3D box, shape, scale-recovery, per-kind measurement, uncertainty, reliability and versioned export-schema contracts.
- `core/tracking`: algorithm-independent tracking states and reset provenance.

Depth and perception registries remain completely separate. `GeometryResult` and `InferenceResult` are unchanged. No perception model, worker, process manager, UI control, alignment implementation, measurement calculation, tracking algorithm or export writer is activated.

## 2. Required revision coverage

1. **Manual mask authority**  
   Approved manual revisions override native, prompted and box-fallback masks. Pending or rejected manual revisions do not override automatic masks. Parent ID, revision, approval status, approver and timestamp are validated.

2. **Split-process test runner**  
   `scripts/run_phase7_5_step1_tests.sh` supports `quick` and `full`. Full mode invokes unit, UI, integration and stress groups in four separate pytest processes and attempts all groups before returning the combined status.

3. **Perception GPU-resource metadata**  
   Perception descriptors include estimated/minimum VRAM, preferred device, CPU fallback, co-residency policy, exclusive-GPU requirement, supported precision and hardware profile metadata. No scheduling or allocation is implemented.

4. **Authoritative ImageTransform2D**  
   One 3×3 homogeneous matrix is authoritative. Crop, resize and padding records follow a fixed crop → resize → pad order and must reproduce the matrix and destination size exactly. Lens distortion is explicitly outside this contract.

5. **Identity RigidTransform3D**  
   Equal source/destination frame IDs are accepted only for an identity transform within tolerance. Finite values, homogeneous row, orthonormal rotation and positive determinant are validated.

6. **Capability-specific perception interfaces**  
   A lifecycle-only base is combined with `FramePerceptionAdapter`, `PromptedSegmentationAdapter` and `SequencePerceptionAdapter` protocols. Typed requests are defined. `UnsupportedPerceptionOperation` is the common unsupported-operation error.

7. **Flexible mask payload representation**  
   Dense boolean, RLE, polygon and shared-memory-reference payload schemas are defined. Dense normalization is complete. Other representations are valid schema contracts but explicitly raise a typed deferred-runtime-normalization error in Step 1.

8. **Honest uncertainty basis**  
   Heuristic perturbation, temporal observation, model-provided, empirically calibrated and unknown bases are represented. Coverage probability/factor are accepted only for empirically calibrated uncertainty. Non-calibrated wording is exposed as `Estimated Uncertainty — Not Calibrated`.

9. **Authoritative uncertainty representation**  
   `value_si` and `expanded_uncertainty_si` are authoritative. Bounds are derived and any supplied/serialized convenience bounds must match. Physically nonnegative quantities reject negative lower bounds unless an explicit clip-to-zero policy is selected.

10. **Per-measurement eligibility**  
    Status and rejection reasons are stored on every `MeasurementEstimate`. Object warnings do not reject unrelated measurement kinds.

11. **Explicit point source-pixel space**  
    `ObjectPointCloud` stores `source_pixel_space_id`, its dimensions and source pixel indices. Every index is validated against that declared space.

12. **Detection-box normalization**  
    Finite raw detector coordinates are clipped to `[0, width] × [0, height]`, original raw coordinates are preserved, `was_clipped` is recorded and only boxes empty after clipping are rejected. Boxes use `[x_min, y_min, x_max, y_max)`.

## 3. Files added

### Shared contracts

- `core/contracts/__init__.py`
- `core/contracts/common.py`
- `core/contracts/spatial.py`
- `core/contracts/validation.py`

### Perception

- `core/perception/__init__.py`
- `core/perception/base.py`
- `core/perception/messages.py`
- `core/perception/metadata.py`
- `core/perception/registry.py`
- `core/perception/result.py`
- `core/perception/validation.py`

### Measurement

- `core/measurement/__init__.py`
- `core/measurement/contracts.py`
- `core/measurement/export_schema.py`
- `core/measurement/validation.py`

### Tracking

- `core/tracking/__init__.py`
- `core/tracking/contracts.py`
- `core/tracking/validation.py`

### Tests

- `tests/unit/test_common_contracts.py`
- `tests/unit/test_spatial_contracts.py`
- `tests/unit/test_perception_metadata.py`
- `tests/unit/test_perception_registry.py`
- `tests/unit/test_perception_contracts.py`
- `tests/unit/test_measurement_contracts.py`
- `tests/unit/test_measurement_export_schema.py`
- `tests/unit/test_tracking_contracts.py`
- `tests/unit/test_phase75_step1_backward_compatibility.py`

### Test runner and validation

- `scripts/run_phase7_5_step1_tests.sh`
- `PHASE7_5_STEP1_VALIDATION.md`

## 4. Files modified

None of the existing v0.7.0 files were modified.

A SHA-256 comparison verified all 101 pre-existing Python files under `app/` and `core/` against the extracted authoritative archive.

## 5. Files removed

None.

## 6. Installation / replacement

From the root of the authoritative v0.7.0 repository:

```bash
unzip -o eia_depth_phase7_5_step1_complete_files.zip -d .
chmod +x scripts/run_phase7_5_step1_tests.sh
```

No dependency was added. Do not install a detection or segmentation package for Step 1.

Verify the delivery archive before extraction:

```bash
sha256sum -c eia_depth_phase7_5_step1_complete_files.zip.sha256
```

## 7. Compile validation

```bash
python -m compileall -q app core scripts tests
```

Executed in the delivery environment: **passed**.

Environment note: the delivery container uses Python 3.13.5, while the project formally supports Python 3.11–3.12. This compile result is a syntax/import-file check, not a supported-runtime certification.

## 8. Focused Step 1 gate

```bash
./scripts/run_phase7_5_step1_tests.sh quick
```

Executed result:

```text
37 passed
```

This gate covers all new contract packages and all 12 required revisions.

## 9. Critical backward-compatibility gate

```bash
python -m pytest -q \
  tests/unit/test_model_output_contract.py \
  tests/unit/test_model_registry.py \
  tests/unit/test_runtime_messages.py \
  tests/unit/test_geometry_projection.py \
  tests/unit/test_snapshot_export.py
```

Executed result:

```text
50 passed
```

Additional assertions in the Step 1 gate confirm:

- Existing depth `MODEL_REGISTRY` order is unchanged:
  - `moge-2`
  - `depth-pro`
  - `unidepth-v2`
  - `metric3d-v2`
  - `depth-anything-v2`
  - `video-depth-anything`
  - `marigold`
  - `development-mock`
- Production `PERCEPTION_MODEL_REGISTRY` is empty.
- Legacy `GeometryResult` positional construction remains valid.
- `InferenceResult` remains pickle-compatible.
- Live 3D, Stable Video, Snapshot and Benchmark remain enabled in the unchanged source.
- Reconstruction remains disabled.
- Version remains `0.7.0` and release name remains `Phase 7 Stable`.
- New core packages contain no PyTorch, CUDA or Qt imports.

## 10. Full split-process regression

```bash
./scripts/run_phase7_5_step1_tests.sh full
```

Full mode runs these as separate processes:

```bash
python -m pytest -q tests/unit
python -m pytest -q tests/ui
python -m pytest -q tests/integration
python -m pytest -q tests/stress
```

Delivery-environment status: **not passed / not claimable**.

The complete legacy unit group was attempted and collection stopped on missing project runtime dependencies:

- `PyQt6` not installed
- `vispy` not installed

The delivery container also uses unsupported Python 3.13.5 instead of the project’s Python 3.11–3.12 range. Therefore no real Qt regression result is claimed.

Run full mode in the configured project environment before committing.

## 11. Applicable registry stress sanity

```bash
python -m pytest -q tests/stress/test_registry_adapter_stress.py
```

Executed result:

```text
1 passed
```

No GPU test was executed and no GPU allocation was introduced or claimed.

## 12. Focused manual validation

Run only after installing the normal v0.7.0 environment on Python 3.11 or 3.12:

1. Launch:

   ```bash
   python main.py
   ```

2. Confirm Live 3D, Stable Video, Snapshot and Benchmark are unchanged.
3. Confirm Reconstruction remains disabled.
4. Confirm no perception or measurement controls appear.
5. Run one Development Mock Live 3D inference.
6. Run one Snapshot export.
7. Close the application and confirm no worker remains.

Manual application validation was **not executed** in the delivery container because PyQt6 and VisPy are unavailable. No real Qt or GPU pass is claimed.

## 13. Commit and push commands

Only after compile, quick, critical, full split-process regression, applicable stress tests and focused manual validation all pass in the configured project environment:

```bash
git status --short
git add -A
git commit -m "feat(measurement): add Phase 7.5 architecture contracts"
git push
```

No version bump.  
No tag.  
No release.  
No production perception model.  
No UI activation.  
No reconstruction activation.
