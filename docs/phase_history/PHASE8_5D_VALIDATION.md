# Phase 8.5D Validation — Global-Consistency / Trajectory Hardening

Phase 8.5D is the final hardening/sign-off step for Phase 8.5. It does not add dense fusion, metric alignment, or Reconstruction UI behavior.

## Locked baseline

- Branch: `phase8-reconstruction`
- Base commit: `d3501081ede16a3878f114e55a9f19b8c4096863`
- Phase 8.5A/B/C behavior must remain backward compatible.
- Production Python remains free of direct `pycolmap` / `stella_vslam` linkage.
- Live pose convention remains `T_reconstruction_from_camera`.
- Monocular live reconstruction remains arbitrary scale with `unit_to_metre=None`.
- Reconstruction UI remains disabled.

## What 8.5D adds

- Worker-generation-aware optimizer epochs.
- Safe native revision reset after a stella worker replacement/restart.
- Previous optimizer/keyframe corrections are invalidated at an epoch boundary while immutable raw tracking poses are preserved.
- Native keyframe IDs may be reused by a replacement worker without colliding with the previous worker's optimizer state.
- Stale worker-generation data is rejected.
- Raw-pose integrity verification and stable trajectory hardening telemetry.
- Final keyframe-match / exact-optimized-pose quality assessment.
- Dependency-light restart/revision integration and stress coverage.
- Real stella endurance smoke covering loop/global optimization, deliberate worker termination/replacement, new optimizer epoch, stale-generation rejection, tracking loss/relocalization, cleanup, and raw trajectory preservation.

## Focused validation

```bash
./scripts/run_phase8_5d_tests.sh quick
./scripts/run_phase8_5d_tests.sh integration
./scripts/run_phase8_5d_tests.sh stress
python scripts/run_live_trajectory_hardening_smoke.py
```

Expected focused counts on the target environment:

- quick: 11 passed
- integration: 1 passed
- stress: 1 passed
- hardening smoke: PASS

## Critical real stella endurance gate

The already-qualified Phase 8.5B native worker is reused; no worker rebuild is required for 8.5D.

```bash
rm -rf /tmp/phase8_5d_live_stella_trajectory_endurance
python scripts/run_live_stella_trajectory_endurance_smoke.py
cat /tmp/phase8_5d_live_stella_trajectory_endurance/summary.json
```

Required invariants include:

- `pass=true`
- two native worker generations are exercised
- deliberate first-worker termination occurs
- `optimizer_epoch=2`
- exactly one optimizer epoch transition
- stale worker-generation status is rejected
- a loop request is accepted in both worker generations
- trajectory quality is accepted in both generations
- old pre-restart optimized corrections are not applied after the restart
- raw tracking poses remain bitwise unchanged
- optimized exact and propagated poses are produced in the replacement epoch
- native relocalization count increases after the deliberate blank interval
- no leaked worker sockets or shared-memory segment
- `T_reconstruction_from_camera`
- arbitrary scale / `unit_to_metre=null`

## Backward gates

```bash
./scripts/run_phase8_5c_tests.sh quick
./scripts/run_phase8_5b_tests.sh quick
./scripts/run_phase8_5a_tests.sh quick
./scripts/run_phase8_4d_tests.sh quick
./scripts/run_phase8_4c_tests.sh quick
./scripts/run_phase8_4b_tests.sh quick
./scripts/run_phase8_4a_tests.sh quick
./scripts/run_phase8_3d_tests.sh quick
./scripts/run_phase8_3c_tests.sh quick
./scripts/run_phase8_3b_tests.sh quick
./scripts/run_phase8_2_tests.sh quick
./scripts/run_phase8_1_tests.sh quick
```

Established target counts before 8.5D:

- 8.5C: 12 passed
- 8.5B: 12 passed
- 8.5A: 12 passed
- 8.4D: 7 passed
- 8.4C: 14 passed on the target PyQt environment
- 8.4B: 21 passed
- 8.4A: 42 passed
- 8.3D: 76 passed
- 8.3C: 63 passed
- 8.3B: 48 passed
- 8.2: 58 passed
- 8.1: 67 passed

## Isolation checks

```bash
grep -RInE \
'^[[:space:]]*(from[[:space:]]+pycolmap|import[[:space:]]+pycolmap)' \
core/reconstruction \
&& echo "ERROR: direct PyCOLMAP import found" \
|| echo "PASS: core/reconstruction remains PyCOLMAP-free"
```

```bash
grep -RInE \
'^[[:space:]]*(from[[:space:]]+stella_vslam|import[[:space:]]+stella_vslam)|ctypes\.(CDLL|PyDLL)' \
core/reconstruction \
&& echo "ERROR: native stella linkage found in Python" \
|| echo "PASS: stella_vslam remains external-process isolated"
```

## Full regression

```bash
./scripts/run_phase8_5d_tests.sh full
python main.py
```

Only the same known optional target-hardware perception skips are acceptable. The application must start/exit normally and Reconstruction must remain disabled.

## Commit gate

Before committing:

```bash
git diff --check
git status --short
git branch --show-current
git rev-parse HEAD
```

The branch must remain `phase8-reconstruction` and HEAD must still be `d3501081ede16a3878f114e55a9f19b8c4096863` until 8.5D is committed.
