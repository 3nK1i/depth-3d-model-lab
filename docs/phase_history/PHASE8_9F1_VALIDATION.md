# Phase 8.9F1 — External Backend + Dataset Preflight

Baseline: clean pushed `phase8-reconstruction` at `9debb17936abd1f8ac3b64fbbdb1d0dcdf61fe7e`.

## Scope

F1 is diagnostic-only. It does not modify production code, external checkouts,
Conda environments, datasets, or backend artifacts. It does not start heavy
training/inference.

The preflight verifies current workstation state for:

- external PyCOLMAP;
- isolated Depth Anything 3 runtime + runtime lock + deliberate Turing patch;
- isolated gsplat and Graphdeco Gaussian runtimes/checkouts;
- isolated Nerfstudio/Nerfacto + tiny-cuda-nn runtime/checkouts;
- production `depth` heavy-runtime isolation;
- South Building real image sequence readiness;
- disk/GPU availability;
- source-level SfM/DA3/canonical/appearance/persistence/UI integration contracts.

## Critical source finding

The current DA3 provider does not accept an externally supplied SfM trajectory.
The external DA3 worker infers its own camera extrinsics together with relative
depth. Therefore Phase 8.9F must not claim that PyCOLMAP poses are injected into
DA3.

The current F4 real Gaussian+Nerfacto runner also uses the deterministic F3
fixture. A later F2/F3/F4 validation-only bridge may bind real DA3 evidence into
canonical geometry and then reuse the existing appearance-provider contracts;
this does not require a production architecture change.

## Authority

- DA3: supporting relative/arbitrary geometry until canonicalized;
- canonical geometry: authoritative;
- Gaussian: appearance only;
- NeRF/Nerfacto: appearance only;
- certified metrology: false.

## Gates

```bash
./scripts/run_phase8_9f1_tests.sh quick
./scripts/run_phase8_9f1_tests.sh affected

rm -rf /tmp/phase8_9f1_preflight
python scripts/run_phase8_9f1_external_backend_preflight.py \
  --output-dir /tmp/phase8_9f1_preflight
cat /tmp/phase8_9f1_preflight/summary.json
```

A full F1 PASS requires the repository baseline to remain exact, no unexpected
external checkout drift, required external runtimes to probe successfully, no
heavy runtime importability in `depth`, no direct heavy imports in
`core/reconstruction`, and the South Building sequence to remain available.

Graphdeco is a research/evaluation reference backend. The primary Phase-8
Gaussian release-validation candidate is the qualified isolated gsplat backend.

Do not clean/reset any external checkout during F1.
