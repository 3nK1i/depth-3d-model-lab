# Phase 9.1I — Camera Calibration & Metric Qualification Closure

## Scope

Phase 9.1I is a closure/qualification step. It adds no new production feature or
authority path. The inherited backward-compatibility matrix exposed one narrow
Phase 9.1H UI/controller call-shape regression, so 9.1I also carries the minimal
compatibility correction required to restore that established Phase 8 boundary.
The step otherwise closes the software implementation delivered by Phase 9.1A–H
and prepares the branch for Phase 9.2 without changing v0.8.0 release identity or
the Phase 8 persistent-scene schema.

Changed-path scope for the corrected closure candidate:

- `app/widgets/reconstruction_page.py`
- `docs/phase_history/PHASE9_1I_VALIDATION.md`
- `scripts/run_phase9_1i_closure_smoke.py`
- `scripts/run_phase9_1i_tests.sh`
- `tests/integration/test_phase9_1i_phase_closure.py`
- `tests/ui/test_phase9_1h_persisted_authority_ui.py`
- `tests/unit/test_phase8_8b3_runtime_contract.py`

No files under `core/`, `config/`, or `requirements/` are modified by 9.1I.

## Closure invariants

The combined closure smoke reuses the already-qualified 9.1D, 9.1E, 9.1G and
9.1H smokes and requires all of the following:

- checkerboard and ChArUco automatic calibration capability remains available;
- camera calibration alone never creates metric scene authority;
- validated reference-target evidence can qualify metric scale;
- known-size benchmark evidence remains supporting-only;
- future RGB-D evidence remains fail-closed until implemented/qualified;
- calibrated camera intrinsics reach Measure & Inspect without changing depth-scale semantics;
- the Phase 8 persistent reconstruction scene schema remains version 1;
- a legacy scene without a Phase 9 authority sidecar remains loadable without invented authority;
- persisted calibration/metric authority is self-contained and does not require the original profile file;
- `certified_metrology` remains false.

## Automated gates

- `./scripts/run_phase9_1i_tests.sh quick`
  - all Phase 9.1 unit/UI tests plus the 9.1I closure integration contract.
- `./scripts/run_phase9_1i_tests.sh integration`
  - all Phase 9.1 integration tests plus affected camera/scale/measurement/persistence integration gates.
- `./scripts/run_phase9_1i_tests.sh backward`
  - inherited Phase 5, 6, 7, 7.5, Phase 8.8 backward, and MainWindow legacy startup/shutdown sanity.
- `./scripts/run_phase9_1i_tests.sh full`
  - complete current-tree unit, UI, integration, and stress suites followed by the inherited backward matrix.
- `python scripts/run_phase9_1i_closure_smoke.py`
  - combined non-UI authority/reproducibility closure smoke.

## Deferred real-artifact validation

The following items are intentionally **not claimed as completed** because suitable
real artifacts are not currently available:

1. Real physical-camera checkerboard/ChArUco calibration -> accepted calibration
   profile -> real Measure & Inspect use with that profile.
2. Real persistent reconstruction scene carrying actual calibration/metric authority
   -> save -> close/restart -> reload -> verify restored authority and provenance.

These are field-validation carry-forward items. Their absence does not justify
fabricating data, weakening authority rules, or changing the qualified software
contracts. They must be executed later when suitable real artifacts are available.

## Release posture

Phase 9.1I must not change the application release identity from v0.8.0 and must
not create a v0.9.0 tag. Phase 9 continues on `phase9-digital-twin` until the final
Phase 9 release gate.

## Closure-gate backward-compatibility correction

The inherited Phase 8 G2 save-interaction test exposed one UI/controller call-shape
regression from Phase 9.1H: ReconstructionPage always supplied the new
`camera_calibration_profile` keyword when saving, while the established Phase 8
boundary was a one-argument `save_persistent_scene(output_dir)` call. The controller
API itself remained backward compatible, but the page no longer exercised the legacy
shape when no calibration authority was involved.

Phase 9.1I therefore includes one narrow compatibility correction:

- if neither an active nor previously persisted calibration exists, the page uses the
  original one-argument save call;
- if a calibration is selected, the authority-aware keyword call is used;
- if a previously persisted calibration was explicitly cleared, `None` is passed
  explicitly so the old calibration cannot be silently retained.

No scene schema, calibration authority rule, metric qualification rule, model/runtime,
or release identity is changed by this correction.

## Full-regression stale source-text contract correction

The complete unit suite then exposed a second closure-only issue in the Phase 8 B3
runtime contract test. That test still required the literal historical UI strings
`Use calibration file` and `Run Test Mode` to appear in
`app/widgets/reconstruction_page.py`. Phase 9.1F intentionally replaced those
labels with the shared professional Camera Calibration workspace, while preserving
the actual B3 behaviors: explicit non-metric Test Mode, estimated-intrinsics policy,
and the legacy JSON calibration loader fallback. Existing behavioral UI tests also
exercise both calibrated and explicit Test Mode starts.

Phase 9.1I therefore updates that stale source-text assertion to verify the current
semantic boundary instead: the shared calibration dialog is opened with Test Mode
enabled, both `estimated` and `calibrated` choices remain reachable, the legacy JSON
loader/file dialog remains in `_choose_live_camera_calibration`, and the original
non-metric/estimated-intrinsics authority checks remain intact. No production text,
runtime path, or authority rule is changed merely to satisfy an obsolete string
assertion.
