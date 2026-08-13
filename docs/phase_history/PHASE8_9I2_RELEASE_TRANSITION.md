# Phase 8.9I2 — v0.8.0 Release Transition

## Purpose

Perform the coordinated final source transition from the Phase 7.5 pre-release identity to the Phase 8 Reconstruction stable identity, then qualify the complete current tree before any tag is created.

## Final identity

```text
core/version.py
  __version__   = "0.8.0"
  RELEASE_NAME  = "Phase 8 Reconstruction Stable"
  RELEASE_PHASE = 8.0

pyproject.toml
  version = "0.8.0"

config/app.yaml
  version: "0.8.0"
  release: "Phase 8 Reconstruction Stable"
```

`main.py`, `core/runtime/release.py`, and the established release-check implementation remain unchanged and consume this central identity.

## Regression transition policy

Historical Phase 8 G/H runners remain frozen as evidence of the pre-release gates they actually executed. Current tests are adjusted so the complete test suite remains valid after the release identity changes to `0.8.0`; they must not rewrite historical evidence to pretend those earlier gates ran under the new version.

The release documentation keeps all qualified authority boundaries:

- canonical geometry is authoritative geometry;
- semantic geometry is context only;
- measurements retain explicit uncertainty and engineering-estimate status;
- inspection/QC remains inspection authority;
- Gaussian and NeRF/Nerfacto representations are appearance only;
- `certified_metrology=false` remains required absent separately qualified calibration/reference evidence;
- monocular reconstruction may remain arbitrary-scale without a valid metric reference;
- heavy reconstruction backends remain isolated from the production `depth` environment.

## Qualification

Run:

```bash
./scripts/run_phase8_9i2_tests.sh quick
./scripts/run_phase8_9i2_tests.sh affected
./scripts/run_phase8_9i2_tests.sh full
```

`full` runs compileall plus the current unit, UI, integration, and stress suites in separate processes. It does not rerun the expensive real SfM, Gaussian, or Nerfacto campaigns because I2 changes release metadata/documentation/test expectations only and does not alter those backend contracts.

## Release boundary

I2 must finish with the local and remote `v0.8.0` tags absent. After all I2 gates pass, the exact I2 scope may be staged and committed. 8.9I3 then verifies the clean synchronized release commit, creates the annotated `v0.8.0` tag, pushes it, and verifies the remote annotated tag points to the release commit.
