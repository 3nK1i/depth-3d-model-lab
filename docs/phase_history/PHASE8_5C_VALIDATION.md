# Phase 8.5C — Optimized Trajectory Propagation

Base: `phase8-reconstruction` at `8df0942141408408fc70fa461d88dc1d7d633cfa`.

## Scope

Phase 8.5C adds a separate live optimized-trajectory layer for reconstruction consumers.

- Raw `LiveSlamObservation.pose` values remain immutable tracking output.
- Native stella optimized keyframe poses are matched through `keyframe_id`.
- Exact optimizer poses use the `optimized_keyframe` layer.
- Non-keyframes may opt into a causal correction from the latest preceding optimized keyframe and are explicitly tagged `optimized_propagated`.
- Stale revisions and same-revision pose mutations are rejected.
- New keyframes may legitimately appear without an optimization-revision bump if existing optimized poses did not change.
- All data remains `T_reconstruction_from_camera`, arbitrary scale, `unit_to_metre=None`.
- No dense fusion, metric alignment, Reconstruction UI enablement, or native worker rewrite is introduced.

The already-built Phase 8.5B native worker is reused unchanged.

## Focused validation

```bash
./scripts/run_phase8_5c_tests.sh quick
./scripts/run_phase8_5c_tests.sh integration
./scripts/run_phase8_5c_tests.sh stress
python scripts/run_live_trajectory_propagation_smoke.py
```

Expected focused results on the target environment:

- quick: `11 passed`
- integration: `1 passed`
- stress: `1 passed`
- propagation smoke: `PASS`

## Real isolated stella gate

The Phase 8.5B worker must already be installed at the normal qualified path.
No rebuild is required for 8.5C.

```bash
rm -rf /tmp/phase8_5c_live_stella_trajectory
python scripts/run_live_stella_trajectory_propagation_smoke.py \
  --base-frames 700 \
  --replay-frames 220 \
  --post-loop-frames 140
cat /tmp/phase8_5c_live_stella_trajectory/summary.json
```

Required invariants include:

- `pass: true`
- real isolated stella execution
- accepted native loop request
- optimization revision advances
- exact optimized keyframes are matched back to raw live keyframes
- at least one propagated non-keyframe pose
- raw tracking poses remain byte-for-byte unchanged
- pose layers remain explicit (`tracking`, `optimized_keyframe`, `optimized_propagated`)
- arbitrary scale / no metric claim

## Backward/full validation

```bash
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

./scripts/run_phase8_5c_tests.sh full
python main.py
```

Reconstruction mode must remain disabled. Do not tag a release in Phase 8.5C.
