# Phase 8.7G4 — Full Regression + UI-Readiness Gate

## Scope

G4 is the closure gate for Phase 8.7. It does not add reconstruction algorithms or enable the Reconstruction UI.

It verifies:

- the complete unit/UI/integration/stress regression suite;
- Phase 8.7 reconstruction authority and dependency-isolation invariants;
- the existing UI remains backward compatible while Reconstruction stays disabled;
- one bounded real F4 → F5 whole-engine run still completes after the full regression;
- the resulting persistent scene remains canonical-geometry authoritative;
- Gaussian and NeRF remain `appearance_only`;
- the repository is ready to hand off to Phase 8.8 UI/application integration.

## Required baseline

- branch: `phase8-reconstruction`
- HEAD: `bd372b35e2835c101931e7769baca652984a0237`
- clean working tree

## Focused gate

```bash
./scripts/run_phase8_7g4_tests.sh quick
python scripts/run_reconstruction_g4_readiness_smoke.py
```

## Full Phase 8.7 regression

```bash
rm -rf /tmp/phase8_7g4_full_regression
./scripts/run_phase8_7g4_tests.sh full
cat /tmp/phase8_7g4_full_regression/summary.json
```

All four test groups — unit, UI, integration and stress — must pass. Skips are recorded and must be understood; the runner never converts a failing group into a pass.

## Real final non-UI gate

```bash
rm -rf /tmp/phase8_7g4_real_e2e
python scripts/run_reconstruction_g4_real_e2e.py \
  --output-dir /tmp/phase8_7g4_real_e2e \
  --regression-summary /tmp/phase8_7g4_full_regression/summary.json \
  --nerf-training-steps 16
cat /tmp/phase8_7g4_real_e2e/summary.json
```

The real gate must report:

- `phase8_7_complete=true`
- `phase8_8_ui_handoff_ready=true`
- `full_regression_passed=true`
- `pipeline_status=completed`
- `persistent_scene_state=ready`
- `canonical_geometry_authoritative=true`
- `gaussian_authority=appearance_only`
- `nerf_authority=appearance_only`
- `certified_metrology=false`
- `reconstruction_ui_enabled=false`

The UI remains disabled by design until Phase 8.8.
