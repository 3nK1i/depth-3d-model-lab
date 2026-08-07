# Phase 8.3B Validation — SfM Contracts, Workspace, and Mock Backend

## Scope

Phase 8.3B adds the dependency-light foundation required before real PyCOLMAP execution:

- gauge-aware sparse SfM contracts that do not label raw monocular coordinates as metres,
- deterministic COLMAP camera-pose conversion helpers,
- durable keyframe workspace materialization for image sequences and recorded videos,
- external PyCOLMAP environment/capability discovery without importing PyCOLMAP in the main process,
- an isolated requirement pin for the future CUDA 12 reconstruction worker environment,
- a deterministic mock offline SfM backend,
- focused unit/integration tests and a non-UI smoke.

Real feature extraction, matching, incremental mapping, bundle adjustment, and real COLMAP sparse reconstruction are **not** executed in this step. Reconstruction remains disabled in the UI.

## Environment decision

The main `depth` environment remains unchanged and must not receive `pycolmap-cuda12`. The validated worker dependency is pinned separately in:

`requirements/reconstruction-pycolmap-cuda12.txt`

The external environment probe accepts an explicit Python executable and runs discovery in a child interpreter. The normal `core.reconstruction` import path does not directly import PyCOLMAP.

## Coordinate and scale safety

Raw Phase 8.3 sparse output uses:

- `T_reconstruction_from_camera`
- `position_reconstruction`
- `SfmScaleStatus.ARBITRARY`
- `unit_to_metre=None`

Metric fields are not populated until a later scale-establishment stage. COLMAP camera-from-reconstruction transforms are inverted at the adapter boundary to produce the internal camera-to-reconstruction pose.

## Automated validation

From the repository root:

```bash
./scripts/run_phase8_3b_tests.sh quick
./scripts/run_phase8_3b_tests.sh integration
```

Package-preparation baseline:

- quick: 44 passed
- integration: 2 passed
- Phase 8.1 quick regression: 67 passed
- Phase 8.2 quick regression: 58 passed
- non-UI Phase 8.3B smoke: PASS

The complete historical unit/UI/integration/stress regression must be run in the user's `depth` environment because the package-preparation environment does not contain PyQt6 or VisPy.

## Non-UI smoke

Run with no real PyCOLMAP dependency:

```bash
python scripts/run_reconstruction_sfm_foundation_smoke.py
```

Expected characteristics:

- 8 selected keyframes,
- only 2 retained in the bounded Phase 8.2 preview cache,
- all 8 materialized into the durable SfM workspace,
- deterministic mock cameras and sparse landmarks,
- gauge reported as `arbitrary`,
- `unit_to_metre=None`,
- `real_sfm_executed=False`.

Default outputs:

- `/tmp/phase8_3b_sfm_foundation_smoke/workspace/manifest.json`
- `/tmp/phase8_3b_sfm_foundation_smoke/sfm_foundation_report.json`
- `/tmp/phase8_3b_sfm_foundation_smoke/mock_sparse_arbitrary_scale.ply`

To probe the already validated standalone PyCOLMAP environment without installing anything into `depth`:

```bash
python scripts/run_reconstruction_sfm_foundation_smoke.py \
  --sfm-python /home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python
```

The probe should report PyCOLMAP 4.1.1, the `pycolmap-cuda12` distribution, required APIs available, and one CUDA device on the RTX 2060 test machine.

## Full regression

```bash
./scripts/run_phase8_3b_tests.sh full
```

The two existing optional real SAM/Ultralytics hardware skips may remain expected. No other test should fail.

## Targeted manual validation

1. Run the non-UI smoke and inspect the JSON report.
2. Confirm `gauge.scale_status` is `arbitrary` and `unit_to_metre` is `null`.
3. Confirm the workspace contains one durable image for every selected keyframe even though the Phase 8.2 preview cache is smaller.
4. Optionally open the mock PLY in an external point-cloud viewer; it is synthetic contract-validation output, not a real reconstruction.
5. Start the application and confirm Reconstruction is still disabled.
6. Perform one small existing depth/3D sanity check only.

## Git review

```bash
git diff --check
git status --short
git diff --stat

git add -A
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

Do not stage generated `/tmp` smoke outputs, caches, datasets, model weights, or reconstruction workspaces.

Recommended commit after all local validation passes:

```bash
git commit -m "feat(reconstruction): add SfM foundation and workspace contracts"
git push
```

Do not create a tag and do not merge into `main`.

## Next implementation step

After Phase 8.3B is validated and committed, Phase 8.3C should connect the real isolated PyCOLMAP worker: SIFT extraction, sequential/exhaustive matching, incremental mapping, bundle adjustment, structured progress/failures, and conversion into these gauge-aware contracts.
