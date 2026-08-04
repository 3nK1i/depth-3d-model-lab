# Phase 6 Step 7 Validation

This checklist covers only the cross-model benchmark, release reporting, release identity, and the `MODEL VRAM` terminology added in Step 7.

## 1. Compile and automated tests

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase6_step7_tests.sh quick
./scripts/run_phase6_step7_tests.sh full
```

## 2. Benchmark-engine smoke test

Run the dependency-light Development Mock in the same isolated-process engine used by production models:

```bash
python scripts/run_phase6_benchmark.py \
  --models development-mock \
  --include-mock \
  --smoke-test \
  --output-dir outputs/phase6/smoke
```

Confirm these files exist and are readable:

```text
outputs/phase6/smoke/benchmark_results.json
outputs/phase6/smoke/benchmark_results.csv
outputs/phase6/smoke/hardware_summary.json
outputs/phase6/smoke/benchmark_summary.md
outputs/phase6/smoke/model_capability_matrix.md
```

## 3. RTX 2060 cross-model benchmark

Close other CUDA applications where practical, then run:

```bash
python scripts/run_phase6_benchmark.py \
  --models all \
  --profile auto \
  --point-budget 100000 \
  --require-all
```

Expected RTX 2060 profile: `efficient`.

Check `outputs/phase6/benchmark_results.json`:

- Seven production models are present.
- Every model status is `passed`.
- Development Mock is absent.
- Video Depth Anything uses workload `sequence` and records multiple frames.
- Marigold is classified `offline_quality_depth`.
- Metric and relative depth labels match registry metadata.
- Intrinsics provenance is `model_predicted` or `estimated`, never `calibrated`.
- Load, warm-up, inference, end-to-end, FPS, output resolution, point count, and VRAM fields are present.

## 4. Report review

Open:

```text
outputs/phase6/benchmark_summary.md
outputs/phase6/model_capability_matrix.md
outputs/phase6/benchmark_results.csv
outputs/phase6/hardware_summary.json
```

Confirm:

- Relative models are not presented as metric measurement systems.
- Estimated-intrinsics models retain the “Estimated — Not Calibrated” warning.
- Marigold is not compared as a real-time model.
- Failed or unavailable models would be listed without deleting successful results.

## 5. UI terminology and release identity

Start the app:

```bash
python main.py
```

Confirm only these Step 7 changes:

- Header displays `v0.6.0`.
- Startup log says `Phase 6 Stable`.
- Footer label is `MODEL VRAM`, while the header remains total device `VRAM`.

## 6. Release gate

Before the commit, run:

```bash
./scripts/run_phase6_release.sh full \
  --require-benchmark
```

After committing, run the final clean-tree gate:

```bash
./scripts/run_phase6_release.sh full \
  --require-benchmark \
  --require-clean-git
```

Expected report:

```text
outputs/release/phase6_release_report.json
```

The gate must report `PASSED` before creating the `v0.6.0` tag.

## 7. RTX 5090 deployment benchmark

Run the identical command on the RTX 5090 machine without editing source or configuration:

```bash
python scripts/run_phase6_benchmark.py \
  --models all \
  --profile auto \
  --point-budget 100000 \
  --require-all \
  --output-dir outputs/phase6/rtx5090
```

Expected profile: `ultra`. Keep the generated JSON and CSV beside the RTX 2060 reports for comparison.

> The deterministic benchmark measures performance, memory, output contracts, and capabilities. It does not claim ground-truth depth accuracy.
