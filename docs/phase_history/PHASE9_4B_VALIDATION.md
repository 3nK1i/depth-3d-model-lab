# Phase 9.4B Validation — Canonical Scene → Gaussian Input Bridge

## Scope

Phase 9.4B binds exact persistent canonical-scene source/camera evidence into the existing Phase 8 `NeuralSceneDataset` contract consumed by Gaussian workers. It does **not** create a new Gaussian input format and does **not** train, optimize, render, or persist a Gaussian representation.

## Authority boundary

- Canonical scene geometry remains authoritative.
- Persisted camera poses are supporting geometry/context only.
- Gaussian input/output remains appearance-only.
- Gaussian does not gain metric-scale, measurement, QC, reference, or certified-metrology authority.
- Arbitrary-scale scenes stay arbitrary (`unit_to_metre = null`).
- Source images are paired to canonical cameras only by exact persisted source-frame SHA-256.
- Missing or drifted source images, camera poses, fingerprints, DA3 sidecars, or intrinsics fail closed.

## Reuse

The bridge reuses:

- `NeuralSceneDataset` / `NeuralSceneFrame` from Phase 8.7D/E;
- `GaussianWorkerRequest.from_dataset()` from the existing Gaussian pipeline;
- persistent Phase 9.3C canonical camera snapshots;
- retained Phase 9.3H/J DA3 input/result sidecars for the real DA3 adapter.

No `gsplat`, Graphdeco, Nerfstudio, DA3 model, YOLO, or SAM2 backend is started by Phase 9.4B.

## Validation gates

```bash
./scripts/run_phase9_4b_tests.sh quick
./scripts/run_phase9_4b_tests.sh integration
./scripts/run_phase9_4b_tests.sh backward
python scripts/run_phase9_4b_gaussian_input_smoke.py
```

Real already-produced DA3 scene bridge (no DA3/Gaussian execution):

```bash
rm -rf /tmp/phase9_4b_real_gaussian_input
python scripts/run_phase9_4b_real_scene_bridge.py \
  --qualification-summary /tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json \
  --output-dir /tmp/phase9_4b_real_gaussian_input
```

Expected real result: `status=PASS`, exact source/camera binding, retained DA3 intrinsics, no heavy Gaussian modules in `depth`, and no training/rendering.

## Deferred to later 9.4 steps

- Gaussian seed policy and optimization execution: 9.4C.
- Persistent Gaussian representation: 9.4D.
- Interactive Gaussian viewer: 9.4E.
- Gaussian controls/layers UX: 9.4F.
- GPU/resource hardening: 9.4G.
- Real end-to-end Gaussian qualification/closure: 9.4H.
