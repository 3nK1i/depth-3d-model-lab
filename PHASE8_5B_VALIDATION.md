# Phase 8.5B Validation — Native stella Global Consistency

Phase 8.5B crosses the native global-consistency boundary while preserving the isolated-worker architecture established in Phase 8.4.

## Scope

- Generic live requests may opt into loop closure; global optimization requires loop closure.
- The qualified stella backend accepts either both native loop/global features or neither, avoiding a false feature split for this backend.
- Native stella loop-detector state follows the explicit request and is verified after startup.
- The AF_UNIX protocol exposes a bounded manual loop-closure test/control request and a global-consistency status snapshot.
- Status snapshots contain optimized native keyframe poses as `T_reconstruction_from_camera` in a separate `optimized` pose layer.
- Native tracking recovery telemetry is surfaced without claiming metric scale or matched-keyframe details that the boundary does not provide.
- Existing raw tracking observations remain unchanged and are not overwritten by optimized poses in Phase 8.5B.

## Intentionally deferred

- Applying optimized/global corrections to live frame poses: Phase 8.5C.
- Repeated-loop/relocalization hardening and final Phase 8.5 signoff: Phase 8.5D.
- Dense fusion: Phase 8.6.
- Mesh/texture/semantics: Phase 8.7.
- Reconstruction UI enablement: Phase 8.8.

## Target validation

```bash
./scripts/run_phase8_5b_tests.sh quick
./scripts/run_phase8_5b_tests.sh integration
./scripts/run_phase8_5b_tests.sh stress
```

Rebuild the isolated native worker because `native/live_slam_worker/main.cc` changed:

```bash
./scripts/build_stella_live_worker.sh
```

Then run the real native gate:

```bash
rm -rf /tmp/phase8_5b_live_stella_global_consistency
python scripts/run_live_stella_global_consistency_smoke.py \
  --base-frames 700 \
  --replay-frames 220 \
  --blank-frames 25 \
  --recovery-frames 160
cat /tmp/phase8_5b_live_stella_global_consistency/summary.json
```

The real smoke must report `PASS`, native loop/global enablement, at least one accepted native loop request, optimized pose activity, a deliberate blank-frame tracking loss followed by a new native recovery/relocalization (the relocalization count must increase across that interval), separate raw/optimized pose layers, `T_reconstruction_from_camera`, and arbitrary scale with `unit_to_metre=null`.

Run the backward gates and full regression before commit. Reconstruction must remain disabled in the UI. Do not create a Phase 8 release tag.
