# Phase 8.6A Validation — Camera Model / Calibration Abstraction

Base branch: `phase8-reconstruction`
Required base commit: `be37423b63f12118c48f6066d6a05aab8f4ba207`

## Scope

Phase 8.6A introduces explicit estimated-vs-calibrated camera geometry without adding metric scale or dense fusion.

- Existing calibrated live SLAM stays strict by default.
- Estimated intrinsics are `ESTIMATED` + `NOT_CALIBRATED` and require explicit opt-in.
- Assumed-FOV profiles record focal-length uncertainty and zero-distortion assumption.
- Estimated profiles may be used for provisional visual tracking / relative reconstruction only.
- Calibrated ray geometry still requires explicit calibration.
- Camera calibration does not itself establish metric scale; Phase 8.6B owns scale/reference semantics.
- Reconstruction UI remains disabled.
- No PyCOLMAP or stella_vslam Python import/linkage is introduced.

## Focused tests

```bash
./scripts/run_phase8_6a_tests.sh quick
./scripts/run_phase8_6a_tests.sh integration
./scripts/run_phase8_6a_tests.sh stress
python scripts/run_camera_geometry_smoke.py
```

## Real isolated-stella provisional-camera gate

No native worker rebuild is required. Reuse the qualified Phase 8.5B worker.

```bash
rm -rf /tmp/phase8_6a_estimated_camera_stella
python scripts/run_estimated_camera_stella_smoke.py --max-frames 600
cat /tmp/phase8_6a_estimated_camera_stella/summary.json
```

Acceptance requires `pass=true`, `camera_geometry_mode=estimated`, `calibration_status=not_calibrated`, explicit `allow_estimated` policy, real isolated stella execution, useful tracking/keyframes/landmarks, arbitrary scale, and no metric-accuracy claim.

## Backward/full gates

Run Phase 8.5D → 8.1 quick gates, the PyCOLMAP/stella isolation greps, then:

```bash
./scripts/run_phase8_6a_tests.sh full
python main.py
```

Do not enable Reconstruction UI and do not create a release tag.
