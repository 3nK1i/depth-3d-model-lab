# Phase 8.9H2 — Documentation & Release Preparation

## Purpose

Prepare truthful, release-facing Phase 8 documentation while preserving the pre-release runtime/package identity until 8.9I.

## Changes

- Updated `README.md` with a Phase 8 Reconstruction release-candidate overview.
- Added `RELEASE_NOTES_v0.8.0.md` as prepared release notes, explicitly not a completed release claim.
- Documented Reconstruction live/offline workflows, persistence/export and final qualification coverage.
- Added explicit authority boundaries for canonical geometry, semantic context, measurements/QC and Gaussian/NeRF appearance representations.
- Updated calibration wording to post-`v0.8.0` future qualification.
- Clarified that the shared Live 3D `Mesh (Poisson)` control remains disabled even though Reconstruction has its own surface-mesh workflow.
- Documented deliberate exclusions: certified metrology, fully interactive neural free-view, broader Auto Segment/custom SAM workflow and unified digital-twin workspace.

## Release boundary

H2 deliberately keeps these values unchanged:

```text
core/version.py   -> 0.7.5 / Phase 7.5 Stable / 7.5
pyproject.toml    -> 0.7.5
config/app.yaml   -> 0.7.5 / Phase 7.5 Stable
v0.8.0 tag        -> absent
```

The coordinated version transition and annotated tag are reserved for 8.9I after H3 release-candidate qualification.

## Scope

Expected H2 Git changes are exactly:

```text
README.md
RELEASE_NOTES_v0.8.0.md
docs/phase_history/PHASE8_9H1_RELEASE_INVENTORY_AUDIT.md
docs/phase_history/PHASE8_9H2_RELEASE_PREPARATION.md
scripts/run_phase8_9h2_release_prep_tests.sh
tests/unit/test_phase8_9h2_release_prep_contract.py
```

No `app/`, `core/`, `native/`, existing historical release note, version metadata or tag is changed by H2.
