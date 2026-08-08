# Phase 8.7D3 Step 4 Validation — Gaussian Backend Robustness

Step 4 validates **isolated Gaussian backend-process robustness only**. Whole reconstruction-engine/session recovery, checkpoint/resume, long-running session recovery, and cross-component cleanup remain Phase 8.7G responsibilities.

## Scope

For each selected Gaussian backend (`gsplat_external`, `graphdeco_reference_external`), the dedicated robustness smoke exercises four deterministic failure scenarios:

1. cancellation of a worker process group containing a child process;
2. timeout of the same process-tree shape;
3. intentional worker crash;
4. intentional invalid partial result artifact followed by cleanup.

Each failure is followed by a real backend recovery run using the already-qualified isolated backend environment. Recovery is accepted only when the worker exits normally and produces a real scene artifact plus held-out render manifest.

Fault injection is disabled by default and requires `EIA_GAUSSIAN_LIFECYCLE_FAULT_INJECTION=1`. Normal benchmark requests are unaffected.

The normal real-scene benchmark launcher now supervises subprocesses in their own process groups, so timeout cleanup applies to regular Step 2/3 backend and metrics-worker execution as well.

## Architectural invariants

- production `depth` environment remains separate from Graphdeco/gsplat runtimes;
- no direct heavy Gaussian imports are added to `core/reconstruction`;
- no Graphdeco/gsplat external source changes are introduced by Step 4;
- canonical geometry remains authoritative;
- Gaussian output remains appearance-only;
- `certified_metrology=False` remains unchanged;
- Reconstruction UI remains untouched/disabled;
- engine-level cancellation/recovery remains deferred to 8.7G.

## Focused gates

```bash
./scripts/run_phase8_7d3_tests.sh quick
./scripts/run_phase8_7d3_tests.sh integration
./scripts/run_phase8_7d3_tests.sh stress
./scripts/run_phase8_7d2_tests.sh quick
```

Expected for this delivery:

- D3 quick: 46 passed
- D3 integration: 4 passed
- D3 stress: 4 passed
- D2 backward quick: 31 passed

## Real backend robustness smoke

```bash
rm -rf /tmp/phase8_7d3_gaussian_backend_robustness

python scripts/run_gaussian_backend_robustness_smoke.py \
  --backend both \
  --recovery-training-steps 16 \
  --cancel-after-s 0.75 \
  --timeout-s 1.0
```

Expected: four passing fault/recovery scenarios for each backend and final:

```text
Phase 8.7D3 Step 4 Gaussian backend robustness: PASS
backend_process_scope_only=True engine_level_recovery_deferred_to_phase8_7g=True
canonical_geometry_authoritative=False certified_metrology=False
```

Evidence:

```text
/tmp/phase8_7d3_gaussian_backend_robustness/step4_robustness_report.json
/tmp/phase8_7d3_gaussian_backend_robustness/step4_summary.json
```

Do not commit until the real both-backend smoke passes and the report confirms cleanup + real recovery for every scenario.
