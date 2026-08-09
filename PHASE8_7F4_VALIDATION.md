# Phase 8.7F4 — Gaussian + NeRF appearance integration

Baseline: `phase8-reconstruction` at `9074a0c872db42173c780f86b5604ab6b0fe9165`.

## Scope

F4 binds the already-qualified Gaussian and Nerfacto representations into the
Phase 8.7F orchestration graph. Canonical dense/surface geometry remains the
only canonical geometry. Gaussian and NeRF artifacts are always
`appearance_only`, never certified metrology, and the reconstruction UI remains
disabled.

F4 does **not** redo D3/E3/E4 quality/performance qualification. It performs a
bounded real-backend orchestration gate to prove that the qualified external
runtimes can consume one shared F4 neural-scene dataset while canonical geometry
authority remains unchanged.

Deferred:
- `PersistentReconstructionScene`, lifecycle/provenance/fallback finalization → F5
- whole-engine hardening → 8.7G
- UI → 8.8

## Focused tests

```bash
./scripts/run_phase8_7f4_tests.sh quick
./scripts/run_phase8_7f4_tests.sh integration
./scripts/run_phase8_7f4_tests.sh stress
python scripts/run_reconstruction_f4_contracts_smoke.py
```

Expected in the delivery baseline:
- quick: 8 passed
- integration: 1 passed
- stress: 1 passed
- contract smoke: PASS

## Backward sanity

```bash
./scripts/run_phase8_7f3_tests.sh quick
./scripts/run_phase8_7f2_tests.sh quick
./scripts/run_phase8_7f1_tests.sh quick
./scripts/run_phase8_7e4_tests.sh quick
./scripts/run_phase8_7e3_tests.sh quick
```

Expected: `16`, `36`, `32`, `77`, `60` passed.

## Real external-backend integration gate

This gate uses the existing deterministic F3 geometry/semantic/QC fixture but
executes the **real isolated gsplat and Nerfstudio/Nerfacto backends**. This is
an integration gate, not a new visual-quality or performance benchmark.

```bash
rm -rf /tmp/phase8_7f4_real_backends

python scripts/run_reconstruction_f4_real_backends.py \
  --output-dir /tmp/phase8_7f4_real_backends \
  --nerf-training-steps 16
```

The script expects the already-qualified environments/checkouts:
- `eia-gaussian-gsplat`
- `eia-nerf-nerfstudio`
- `~/Desktop/eia-nerf-backends/nerfstudio`
- `~/Desktop/eia-nerf-backends/tiny-cuda-nn`

A passing result must show:
- `status=completed`
- `canonical_geometry_authoritative=True`
- Gaussian authority `appearance_only`
- NeRF authority `appearance_only`
- `certified_metrology=False`
- no new backend quality/performance qualification claims
- UI disabled

Evidence:
- `/tmp/phase8_7f4_real_backends/summary.json`
- `/tmp/phase8_7f4_real_backends/pipeline_result.json`
- `/tmp/phase8_7f4_real_backends/appearance_contact_sheet.png` when both backend previews are available

Open the contact sheet with Chrome for manual validation.

## Isolation checks

```bash
grep -RInE \
  '(^|[[:space:]])(import|from)[[:space:]]+(torch|gsplat|nerfstudio|tinycudann|lpips)' \
  core/reconstruction/pipeline_appearance.py

git diff -- pyproject.toml
```

Both commands must print nothing.
