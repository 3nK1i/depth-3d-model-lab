# Phase 8.7D3 Step 1 Validation — Real-Scene Benchmark Contracts

## Baseline

Apply only on:

- branch: `phase8-reconstruction`
- HEAD: `0f81b528c769528ce9fb03c16d13ea7f9d430cf6`
- clean working tree synchronized with `origin/phase8-reconstruction`
- Phase 8.7D2 real Graphdeco + gsplat qualification complete

## D3 scope

Phase 8.7D3 is the real-scene validation and benchmarking stage:

- Graphdeco vs gsplat;
- PSNR / SSIM / LPIPS on held-out views;
- peak VRAM;
- training time;
- render FPS;
- RTX 2060 validation;
- RTX 5090 profile;
- crash / cancel / recovery.

Step 1 adds only the dependency-light benchmark contracts and split/report harness.
It deliberately does **not** modify the D2 real workers or fabricate benchmark
measurements before the held-out rendering and metric implementations exist.

## Safety rules

- training and validation views must be disjoint;
- every dataset view belongs to exactly one split;
- at least one held-out validation view is mandatory;
- development/mock Gaussian backend cannot be used for D3 comparison;
- benchmark identity includes dataset fingerprint, split, backend set, optimization
  profile, and hardware profile;
- PSNR / SSIM / LPIPS are all-or-none measured evidence;
- comparison readiness requires both image-quality and performance evidence;
- lifecycle evidence includes planned/completed/cancelled/timed-out/crashed/recovered/failed;
- Gaussian benchmark results can never claim canonical geometry authority or certified metrology;
- Reconstruction UI remains disabled;
- no new package is added to the production `depth` environment.

## Hardware profiles

Step 1 reserves explicit identities for:

- `rtx_2060_6gb` — current local validation target;
- `rtx_5090_32gb` — deployment/profile target;
- `custom` — requires an explicit hardware label.

Tuning thresholds are intentionally deferred until measured D3 performance evidence exists.

## Files

This Step 1 overlay contains exactly 9 files:

1. `PHASE8_7D3_STEP1_VALIDATION.md`
2. `core/reconstruction/__init__.py`
3. `core/reconstruction/gaussian_benchmarking.py`
4. `scripts/run_gaussian_benchmark_contracts_smoke.py`
5. `scripts/run_phase8_7d3_tests.sh`
6. `tests/unit/test_reconstruction_gaussian_benchmarking.py`
7. `tests/unit/test_phase8_7d3_backward_compatibility.py`
8. `tests/integration/test_reconstruction_gaussian_benchmark_contracts.py`
9. `tests/stress/test_reconstruction_gaussian_benchmark_contracts_stress.py`

## Validation order

```bash
git diff --check
./scripts/run_phase8_7d3_tests.sh quick
./scripts/run_phase8_7d3_tests.sh integration
./scripts/run_phase8_7d3_tests.sh stress
python scripts/run_gaussian_benchmark_contracts_smoke.py
```

Then run D2 quick as the small backward sanity gate:

```bash
./scripts/run_phase8_7d2_tests.sh quick
```

Do not commit Step 1 until all of those pass and the changed-file set is exactly the
9 files above.

## Expected smoke semantics

The smoke creates a deterministic 5-view dataset with 4 training views and one
held-out validation view, writes the D3 benchmark spec/report, and proves that:

- comparison is not yet ready;
- no fake PSNR/SSIM/LPIPS or performance values are emitted;
- canonical geometry remains authoritative outside the Gaussian representation;
- certified metrology remains false.

Later D3 steps will add real-scene ingestion/held-out rendering, metrics,
performance instrumentation, and crash/cancel/recovery execution.
