# Phase 6 Release Test Strategy

## Release objective

Release `v0.6.0 — Phase 6 Stable` with a registry-driven multi-model platform and reproducible cross-model benchmark reports while preserving every Phase 5 runtime guarantee.

## Automated gates

### Focused Step 7 gate

```bash
./scripts/run_phase6_step7_tests.sh quick
```

Covers benchmark assets, isolated benchmark execution, report writers, capability metadata, version identity, and the `MODEL VRAM` UI label.

### Full regression gate

```bash
./scripts/run_phase6_step7_tests.sh full
```

Runs unit, UI, integration, and stress suites in separate Python processes.

### Cross-model benchmark

```bash
python scripts/run_phase6_benchmark.py --models all --profile auto --require-all
```

Each model runs in a fresh process. The benchmark records load, warm-up, forward latency, end-to-end latency, FPS, output resolution, point count, and CUDA memory telemetry. Registry metadata determines workload type, iteration count, sequence length, classification, and timeout.

### Release gate

```bash
./scripts/run_phase6_release.sh full --require-benchmark
```

The gate validates Python, layout, version consistency, base dependencies, all production model dependencies and assets, hardware readiness, RTX 2060/RTX 5090 profile simulations, benchmark assets, full regression tests, and the benchmark report.

## Benchmark fairness rules

- All frame models use the same deterministic BGR input.
- Video Depth Anything uses the deterministic temporal sequence and preserves its cache across frames.
- Marigold is classified as offline/quality-oriented, not real-time.
- Metric and relative outputs remain separate categories.
- Metric output with estimated intrinsics is never called calibrated measurement.
- Development Mock validates the engine but is excluded from production reports.
- One process per model prevents allocator, import, and temporal-state leakage.

## Release evidence

```text
outputs/phase6/benchmark_results.json
outputs/phase6/benchmark_results.csv
outputs/phase6/hardware_summary.json
outputs/phase6/benchmark_summary.md
outputs/phase6/model_capability_matrix.md
outputs/release/phase6_release_report.json
```

## Final acceptance

- All automated suites pass.
- All seven production models pass the physical RTX 2060 benchmark.
- The same repository selects Ultra automatically on RTX 5090.
- The Phase 6 release report passes.
- Git worktree is clean before tagging `v0.6.0`.

> The deterministic benchmark measures performance, memory, output contracts, and capabilities. It does not claim ground-truth depth accuracy.
