# Depth & 3D Model Lab

## Phase 7.5 Step 2 — Segmentation Perception Pipeline Validation

## 1. Scope

This delivery implements the approved Phase 7.5 Step 2 perception pipeline on top of the committed Phase 7.5 Step 1 source tree.

The application version remains:

```text
0.7.0 — Phase 7 Stable
```

Step 2 adds a separate, registry-driven perception runtime for:

- Ultralytics YOLO native instance segmentation
- Ultralytics YOLO detection
- Ultralytics SAM 2.1 prompted mask refinement
- Dependency-free development mock validation
- Multiple independent objects and overlapping instance masks
- Native-mask validation
- Prompted refinement
- Detection-box mask fallback
- Approved manual-mask precedence
- Hardware-adaptive primitive runtime profiles
- Conservative GPU co-residency planning
- A standalone spawned perception worker and bounded process manager

The existing depth runtime, source controllers, operational-mode controllers, UI, measurement implementation, geometry renderer and Reconstruction mode are not activated or coupled to this pipeline.

## 2. Architectural summary

### 2.1 Independent registry

`PERCEPTION_MODEL_REGISTRY` remains separate from the existing depth `MODEL_REGISTRY`.

Registered descriptors, in deterministic order:

1. `ultralytics-yolo-seg`
2. `ultralytics-yolo-detect`
3. `ultralytics-sam2-prompted`
4. `development-perception-mock`

The development mock is nonselectable and development-only. No perception model is added to the existing depth-model selector.

### 2.2 Capability-specific adapters

The generic runtime uses declared capabilities and typed protocols:

- `FramePerceptionAdapter`
- `PromptedSegmentationAdapter`
- `SequencePerceptionAdapter`

The generic worker contains no Ultralytics, YOLO, SAM or model-ID branching. Optional imports exist only inside the corresponding adapter `load()` methods.

### 2.3 Authoritative pipeline

```text
RGB FramePerceptionRequest
→ primary frame adapter
→ dependency-light RawPerceptionBatch
→ detection-box clipping and provenance
→ dense native-mask normalization
→ optional prompted refinement
→ explicit box-mask fallback
→ authoritative mask selection
→ ObjectObservation
→ PerceptionFrameResult
```

Mask authority remains:

1. Approved manual correction
2. Valid native instance mask
3. Valid prompted refinement
4. Detection-box fallback

Pending or rejected manual masks remain preserved as provenance but do not override a valid automatic mask.

### 2.4 Separate process

`PerceptionProcessManager` starts a dedicated spawned process with:

- Latest-only bounded input behavior
- Typed configuration and request messages
- Pickle-safe NumPy contracts
- Typed worker responses
- Sequence-reset support when an adapter declares it
- Bounded shutdown with terminate fallback
- No Qt dependency
- No dependency on the existing depth worker

### 2.5 Hardware and resource behavior

Initial runtime profiles are resolved without allocating a model:

- Efficient: YOLO11n, 640 px
- Balanced: YOLO11s, 640 px
- Ultra: YOLO11s, 960 px
- Prompted SAM profiles: SAM 2.1 Tiny, Small or Base

CUDA uses FP16 only when the detected hardware and PyTorch environment support it. CPU fallback uses FP32.

Descriptor VRAM estimates intentionally remain unknown until measured on the target systems. Prompted co-residency requires explicit approval after profiling; the generic worker rejects an unapproved refiner configuration.

## 3. Implementation boundaries preserved

This step does **not** implement or activate:

- Application perception UI
- Mask-editing UI
- Depth/mask/intrinsics alignment
- Object-specific 3D points
- 3D box fitting
- Measurement calculation
- Measurement uncertainty calculation
- Tracking algorithms
- Measurement export writers
- Camera calibration
- SfM or SLAM
- Scene or multi-camera fusion
- Reconstruction
- Digital Twin functionality
- RF-DETR integration

## 4. Files added

```text
core/perception/adapters/__init__.py
core/perception/adapters/mock_adapter.py
core/perception/adapters/ultralytics_sam.py
core/perception/adapters/ultralytics_yolo.py
core/perception/availability.py
core/perception/catalog.py
core/perception/normalization.py
core/perception/pipeline.py
core/perception/process_manager.py
core/perception/raw.py
core/perception/resource_planner.py
core/perception/runtime_config.py
core/perception/telemetry.py
core/perception/worker.py
requirements/models-ultralytics-perception.txt
scripts/check_perception_environment.py
scripts/profile_perception_vram.py
scripts/run_perception_smoke.py
scripts/run_phase7_5_step2_tests.sh
tests/integration/test_perception_process_lifecycle.py
tests/integration/test_perception_worker_mock.py
tests/integration/test_ultralytics_sam2_adapter_optional.py
tests/integration/test_ultralytics_yolo_adapter_optional.py
tests/stress/test_perception_latest_frame_backpressure.py
tests/stress/test_perception_registry_stress.py
tests/stress/test_perception_worker_restart_stress.py
tests/unit/test_perception_availability.py
tests/unit/test_perception_catalog.py
tests/unit/test_perception_normalization.py
tests/unit/test_perception_pipeline.py
tests/unit/test_perception_raw_outputs.py
tests/unit/test_perception_resource_planner.py
tests/unit/test_perception_runtime_config.py
tests/unit/test_perception_worker_messages.py
tests/unit/test_phase75_step2_backward_compatibility.py
tests/unit/test_ultralytics_perception_adapters.py
PHASE7_5_STEP2_VALIDATION.md
```

## 5. Files modified

Complete replacements are supplied for:

```text
core/perception/__init__.py
core/perception/base.py
core/perception/messages.py
core/perception/metadata.py
core/perception/registry.py
core/perception/result.py
pyproject.toml
tests/unit/test_perception_registry.py
tests/unit/test_phase75_step1_backward_compatibility.py
```

The Step 1 `ObjectObservation` positional field order is preserved. The new prompted-refinement history is appended at the end of the contract.

## 6. Files removed

None.

## 7. Existing production areas unchanged

The delivery does not replace files under these existing areas:

```text
app/**
core/inference/**
core/runtime/messages.py
core/models/**
core/geometry/**
core/measurement/**
core/tracking/**
core/version.py
main.py
```

Consequently, Step 2 does not activate a UI control or connect the perception process to the Phase 7 operational modes.

## 8. Installation and replacement

From the download directory, validate the archive first:

```bash
cd ~/Downloads
sha256sum -c eia_depth_phase7_5_step2_complete_files.zip.sha256
```

Expected:

```text
eia_depth_phase7_5_step2_complete_files.zip: OK
```

Apply the complete files from the repository root:

```bash
cd /home/eiatech/Desktop/depth_3d_model_lab

git status --short

unzip -o \
  ~/Downloads/eia_depth_phase7_5_step2_complete_files.zip \
  -d .

chmod +x \
  scripts/run_phase7_5_step2_tests.sh \
  scripts/check_perception_environment.py \
  scripts/profile_perception_vram.py \
  scripts/run_perception_smoke.py
```

Install the optional perception runtime inside the existing validated `depth` environment:

```bash
conda activate depth

python -m pip install \
  --upgrade-strategy only-if-needed \
  -r requirements/models-ultralytics-perception.txt

python -m pip check
```

This dependency is intentionally not added to the base `requirements.txt`. Application import and the depth pipeline remain available when Ultralytics is not installed.

## 9. Compile validation

```bash
python -m compileall -q app core scripts tests
```

Executed in the delivery environment: **passed**.

## 10. Focused Step 2 quick gate

```bash
./scripts/run_phase7_5_step2_tests.sh quick
```

Executed result:

```text
34 passed
```

The gate covers:

- Registry order and descriptor metadata
- Ultralytics licensing metadata
- Optional dependency isolation
- Adapter capability contracts
- Runtime profile resolution
- Conservative resource planning
- Raw output validation and pickle safety
- Read-only arrays after pickle round trips
- Detection-box clipping and provenance
- Dense native-mask normalization
- Multiple and overlapping masks
- Prompted refinement and refinement provenance
- Detection-box fallback
- Approved manual-mask authority
- Pending manual-mask behavior
- Worker configuration safety
- No model-ID logic in the generic worker
- Step 1 positional-constructor compatibility
- Existing depth registry and release compatibility

## 11. Mock process integration gate

```bash
./scripts/run_phase7_5_step2_tests.sh integration
```

Executed result:

```text
2 passed
```

This gate uses only the development mock and executes a real `spawn` process. It does not import Ultralytics, allocate CUDA memory or require Qt.

## 12. Phase 7.5 Step 1 regression gate

```bash
./scripts/run_phase7_5_step1_tests.sh quick
```

Executed result:

```text
37 passed
```

## 13. Critical depth/IPC backward-compatibility gate

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

This confirms the existing depth model ordering, `GeometryResult`, `InferenceResult`, geometry projection and Snapshot contracts remain compatible.

## 14. Optional-adapter and perception stress sanity

```bash
python -m pytest -q \
  tests/integration/test_ultralytics_yolo_adapter_optional.py \
  tests/integration/test_ultralytics_sam2_adapter_optional.py \
  tests/stress/test_perception_registry_stress.py \
  tests/stress/test_perception_worker_restart_stress.py \
  tests/stress/test_perception_latest_frame_backpressure.py
```

Executed result:

```text
5 passed
```

The optional-adapter tests use controlled fake modules and do not claim real Ultralytics or GPU execution.

## 15. Full split-process regression

```bash
./scripts/run_phase7_5_step2_tests.sh full
```

The runner executes these groups in separate pytest processes:

```text
tests/unit
tests/ui
tests/integration
tests/stress
```

In the delivery container, the runner was executed and correctly attempted every group. It could not complete because that container uses Python 3.13.5 and does not contain the project PyQt6 or VisPy dependencies.

Observed collection limitations:

- Unit group: four existing modules blocked by missing PyQt6 or VisPy
- UI group: blocked by missing PyQt6
- Integration group: existing depth lifecycle test blocked by missing PyQt6
- Stress group: existing Qt-based stress tests blocked by missing PyQt6

This is an environment limitation, not a claimed passing full regression. Run the full gate in the configured Python 3.11/3.12 `depth` environment before committing.

## 16. Optional dependency availability check

```bash
python scripts/check_perception_environment.py
```

Executed in the delivery container:

```text
ultralytics-yolo-seg: missing_dependency
ultralytics-yolo-detect: missing_dependency
ultralytics-sam2-prompted: missing_dependency
development-perception-mock: available
```

This confirms the perception registry and application contracts remain importable without the optional runtime.

## 17. Real Ultralytics validation on target hardware

Real Ultralytics, CUDA, RTX 2060 and RTX 5090 inference were not available in the delivery container and are not claimed as passed.

After installing the optional dependency, check availability:

```bash
python scripts/check_perception_environment.py
```

### RTX 2060 initial native-segmentation smoke

```bash
python scripts/run_perception_smoke.py \
  --model ultralytics-yolo-seg \
  --profile efficient \
  --image ~/Downloads/test-pcb.png \
  --overlay outputs/phase75_step2_yolo_overlay.png
```

Initial expected automatic configuration:

```text
YOLO11n-seg
640 px
FP16 when CUDA is ready
retina masks enabled
```

### Profile primary segmentation VRAM

```bash
python scripts/profile_perception_vram.py \
  --model ultralytics-yolo-seg \
  --profile efficient
```

### Profile SAM 2.1 separately

```bash
python scripts/profile_perception_vram.py \
  --model ultralytics-sam2-prompted \
  --profile efficient
```

### Prompted refinement validation

Run only after reviewing individual VRAM measurements and confirming adequate free memory:

```bash
python scripts/run_perception_smoke.py \
  --model ultralytics-yolo-detect \
  --refiner ultralytics-sam2-prompted \
  --profile efficient \
  --allow-unprofiled-co-residency \
  --image ~/Downloads/test-pcb.png \
  --overlay outputs/phase75_step2_prompted_overlay.png
```

The override is intentionally explicit. Do not make it the default on the RTX 2060 until co-residency has been measured safely.

On the RTX 5090, repeat the primary and SAM profiles with:

```bash
--profile ultra
```

Do not populate descriptor VRAM estimates or enable automatic co-residency until measurements are recorded on both target systems.

## 18. Focused manual application validation

Step 2 adds no application UI. In the configured project environment:

1. Launch the application:

```bash
python main.py
```

2. Confirm Live 3D, Stable Video, Snapshot and Benchmark are unchanged.
3. Confirm Reconstruction remains disabled.
4. Confirm no perception or measurement controls appear.
5. Run one Development Mock Live 3D inference.
6. Run one Snapshot export.
7. Close the application.
8. Confirm no depth or perception worker remains:

```bash
ps aux | grep -E 'main.py|eia-perception-worker|depth_3d_model_lab' | grep -v grep
```

9. Run the standalone YOLO segmentation smoke command.
10. Inspect the generated overlay and confirm separate objects and masks are plausible.

Manual Qt application validation was not executed in the delivery container and is not claimed.

## 19. Acceptance checklist before commit

Do not commit until all applicable items pass in the target environment:

- Compile validation
- Step 2 quick gate
- Mock integration gate
- Step 1 regression gate
- Critical depth/IPC gate
- Full split-process regression
- Perception stress sanity
- Optional dependency availability
- RTX 2060 YOLO segmentation smoke
- RTX 2060 primary VRAM profile
- SAM profile before any co-residency decision
- Focused manual application validation
- Clean shutdown with no remaining worker

## 20. Git commands after validation

```bash
git status --short
git add -A
git status --short
git commit -m "feat(perception): add Phase 7.5 segmentation pipeline"
git push
```

No version bump, tag or release is part of Step 2.
