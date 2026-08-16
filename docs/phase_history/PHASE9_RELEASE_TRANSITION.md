# Phase 9 — v0.9.0 Release Transition

## Purpose

Perform the final source-identity transition after Phase 9.6H has passed its real-scene release gate and complete current-tree regression. This step is release metadata/documentation/test alignment only; it does not reopen Phase 9.6 implementation.

## Final identity

```text
core/version.py
  __version__   = "0.9.0"
  RELEASE_NAME  = "Phase 9 Digital Twin Stable"
  RELEASE_PHASE = 9.0

pyproject.toml
  version = "0.9.0"

config/app.yaml
  version: "0.9.0"
  release: "Phase 9 Digital Twin Stable"
```

`main.py` and `core/runtime/release.py` remain consumers of the central metadata and do not receive hard-coded release strings.

## Transition policy

Historical Phase 8 and earlier release notes, phase-history records and frozen pre-release runners remain truthful evidence of the state they qualified. Current-tree compatibility tests that intentionally consume the current central identity advance to `0.9.0`; historical assertions about old documents/runners remain historical.

The transition must not change reconstruction, calibration, semantic, DA3, Gaussian, NeRF, measurement, persistence, export, QC or resource-ownership runtime behavior.

## Authority boundaries

The v0.9.0 release preserves the final H posture:

- canonical geometry is spatial/picking/measurement geometry authority;
- semantics are contextual and non-authoritative for canonical geometry/scale;
- inspection/QC remains separate authority;
- DA3 remains supporting relative geometry and cannot invent physical scale;
- Gaussian and NeRF/Nerfacto remain appearance only;
- estimated ruler values retain explicit uncertainty and provenance;
- intrinsics alone do not establish metric scale;
- `certified_metrology=false` remains required absent separately qualified evidence;
- no release claim of global loop closure or photorealistic convergence is made.

## Qualification

Run:

```bash
./scripts/run_phase9_release_transition_tests.sh quick
./scripts/run_phase9_release_transition_tests.sh affected
./scripts/run_phase9_release_transition_tests.sh full
```

The full mode runs the complete unit, UI, integration and stress suites in separate processes. It does not rerun the real Phase 9.6H DA3/Gaussian/NeRF campaign because this transition contains no corresponding runtime changes.

## Tag boundary

The transition runner must never create a Git tag. After all gates pass, stage only the intended release-transition paths, commit and push them, verify local/remote synchronization, then create the annotated `v0.9.0` tag on that exact release commit and push the tag explicitly.
