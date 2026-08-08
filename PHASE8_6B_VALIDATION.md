# Phase 8.6B Validation — Scale / Reference Abstraction

## Scope

Phase 8.6B adds explicit scale/reference contracts without changing raw monocular reconstruction semantics. Existing SfM/SLAM geometry remains arbitrary-scale by default. Metric units are introduced only when a traceable scale source is supplied.

This step does **not** add dense fusion, mesh generation, world alignment, certified metrology, robot integration, or Reconstruction UI enablement.

## Safety invariants

- Raw monocular reconstruction remains `SfmScaleStatus.ARBITRARY` with `unit_to_metre=None`.
- Estimated camera intrinsics never imply metric scale.
- A known object dimension produces `ESTIMATED_METRIC` scale and remains provisional.
- A validated reference target may establish `CALIBRATED_METRIC` scale, but calibrated metric geometry still requires calibrated camera intrinsics too.
- Even calibrated camera + calibrated scale does not claim certified metrology in Phase 8.
- Scale and world alignment remain separate. Phase 8.6B does not create a world frame.
- PyCOLMAP and stella_vslam remain externally isolated.
- Reconstruction UI remains disabled.

## Scale modes

- `ARBITRARY`: current/default monocular gauge; no metres conversion.
- `KNOWN_DIMENSION`: user/object known dimension; provisional estimated metric scale.
- `REFERENCE_TARGET`: target/fixture dimension with explicit validation state.
- `EXTERNAL_METRIC_PRIOR`: future-compatible metric prior for external pose/robot systems without adding robot dependencies now.

## Uncertainty

Dimension-based scale uncertainty combines reference-length and reconstruction-measurement relative uncertainty using root-sum-square. The resulting value is stored in `ReconstructionGauge.relative_scale_uncertainty`.

## Validation commands

```bash
./scripts/run_phase8_6b_tests.sh quick
./scripts/run_phase8_6b_tests.sh integration
./scripts/run_phase8_6b_tests.sh stress
python scripts/run_scale_reference_smoke.py
```

Then run all established Phase 8.6A and earlier quick gates plus:

```bash
./scripts/run_phase8_6b_tests.sh full
python main.py
```

## Acceptance

- All focused tests pass.
- Scale/reference smoke prints `PASS`.
- Arbitrary geometry cannot be converted to metres.
- Known-dimension scale is provisional metric with explicit uncertainty.
- Estimated-camera + known-dimension remains provisional metric.
- Estimated-camera + validated target remains provisional because camera intrinsics are still estimated.
- Calibrated-camera + validated target can be classified as calibrated metric geometry, while `certified_metrology` remains false.
- Full regression is green except the same expected optional perception skips.
- Reconstruction UI remains disabled.
