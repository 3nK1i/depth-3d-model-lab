# Phase 9.4G — Performance & GPU Resource Lifecycle

Baseline: clean pushed `phase9-digital-twin` at `8a27b040ffbf25f4ab3f0082c2cdad93166f12f4` (Phase 9.4F closed).

## Scope

9.4G hardens the already-persisted/backend-free interactive Gaussian viewer. It does not add a Gaussian backend, retrain, rerender, change persistence, or promote Gaussian appearance to geometry/metric/measurement/QC authority.

Implemented behavior:

- automatically selects a conservative interactive splat display cap from the cached startup hardware snapshot: CPU/unknown `75k`, ~6 GB devices `100k`, mid-memory devices `150k`, and high-memory devices up to the existing qualified hard ceiling of `200k`;
- keeps the cap explicitly presentation-only. It is not a CUDA/VRAM allocation guarantee or training/render benchmark;
- reports exact application-side NumPy source/display staging bytes and UI-side safe-load/display-apply timings in the Gaussian visualization panel;
- explicitly labels those diagnostics as CPU/UI-side only and never infers OpenGL/CUDA VRAM usage;
- coalesces only active splat-size/opacity slider drags to a bounded ~60 Hz update cadence while preserving immediate discrete/programmatic changes;
- ordinary density/size/opacity updates reuse the existing VisPy Markers visual;
- `Hide`, `Preview`, Gaussian scene-layer OFF, scene switch/clear, and shutdown release the retained `InteractiveGaussianScene`, derived display arrays, and detach/recreate the VisPy Markers node so its GL buffers are not intentionally kept resident;
- re-enabling Interactive performs a fresh integrity-checked CPU load on demand and preserves the user's density/size/opacity settings;
- canonical mesh resources remain loaded through Gaussian release/reload cycles and remain the sole ruler/picking authority.

## Authority and compatibility invariants

- `PersistentReconstructionScene` remains canonical geometry authority.
- Gaussian remains `appearance_only`.
- Gaussian display budgets/timings do not establish metric scale or quality authority.
- measurement/QC authority remains false for Gaussian.
- `certified_metrology=false` unless separately qualified by the existing authority path.
- arbitrary-scale scenes remain arbitrary-scale.
- no gsplat/Graphdeco/NeRF runtime is started by viewer loading, display control changes, lifecycle diagnostics, or resource release.
- persisted `scene.pt` and its integrity references are never modified.

## Automated validation

```bash
./scripts/run_phase9_4g_tests.sh quick
./scripts/run_phase9_4g_tests.sh integration
./scripts/run_phase9_4g_tests.sh stress
./scripts/run_phase9_4g_tests.sh backward
```

The focused suites cover adaptive RTX 2060/RTX 5090 presentation policy, exact CPU staging accounting, diagnostics-only authority, repeated NumPy staging release, UI hide/reload lifecycle, control-setting preservation, Preview/scene-switch release, and affected 9.3I/9.4E/9.4F/interactive-measurement compatibility.

## Real persisted-scene resource check

```bash
python scripts/run_phase9_4g_real_gaussian_resource_check.py \
  --scene-manifest /tmp/phase9_4d_persistent_gaussian_scene/persistent_reconstruction_scene.json \
  --cycles 12
```

This records real safe CPU-load and display-buffer preparation timings for the current machine/run, checks repeated NumPy staging release, reports advisory process RSS, and confirms no heavy Gaussian module starts. No fixed performance number is claimed from CI/synthetic runs.

## Real PyQt/VisPy lifecycle validation

```bash
python scripts/run_phase9_4g_gaussian_lifecycle_preview.py \
  --scene-manifest /tmp/phase9_4d_persistent_gaussian_scene/persistent_reconstruction_scene.json \
  --cycles 12 \
  --show
```

The real production `depth` environment must verify:

1. interactive Gaussian appearance loads and remains navigable;
2. automatic display cap is readable and appropriate for the detected hardware snapshot;
3. repeated Hide→Interactive cycles release and recreate the VisPy Gaussian Markers resource;
4. page-level source/derived NumPy staging references are dropped after every Hide;
5. density/size/opacity settings survive release/reload cycles;
6. canonical mesh remains loaded throughout and ruler picking remains canonical;
7. Scene Layers OFF/ON and Preview/Interactive transitions remain responsive;
8. Gaussian panel resource/performance/lifecycle diagnostics are readable at 1366×768;
9. reported load/apply timings are explicitly UI-side only and RSS is advisory;
10. normal shutdown releases viewer resources without starting gsplat/Graphdeco/NeRF.

Full Phase 9.4 regression remains deferred to 9.4H closure.
