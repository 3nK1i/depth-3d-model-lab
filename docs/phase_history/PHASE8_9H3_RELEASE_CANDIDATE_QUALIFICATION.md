# Phase 8.9H3 — Release-Candidate Documentation Qualification

## Purpose

Qualify the Phase 8 release-preparation documentation produced by H2 without changing production code, canonical runtime/package metadata, or Git tags.

H3 runs on the uncommitted H2 documentation overlay from the clean `phase8-reconstruction` baseline `dff62357874e28de8d0ceadb9a222dd0cd17d999`.

## Qualification boundary

H3 must preserve all of the following until 8.9I:

- `core/version.py` remains `0.7.5` / `Phase 7.5 Stable` / `7.5`.
- `pyproject.toml` remains `0.7.5`.
- `config/app.yaml` remains `0.7.5` / `Phase 7.5 Stable`.
- `main.py` remains unchanged from the H1 baseline.
- local and remote `v0.8.0` tags remain absent.
- no `app/`, `core/`, `native/`, runtime implementation, backend checkout or historical release note is modified.

## Documentation checks

The release candidate must:

1. describe `v0.8.0` as the target release rather than an already-tagged release;
2. document the Reconstruction workspace and its live/offline paths;
3. preserve canonical/semantic/measurement/QC authority separation;
4. state that Gaussian and NeRF/Nerfacto outputs are appearance-only;
5. retain engineering-estimate / `certified_metrology=false` posture without a separately qualified metric reference/calibration path;
6. state that monocular scale can remain arbitrary and camera intrinsics alone do not establish metric scale;
7. keep calibration/metric qualification, broader Auto Segment/custom SAM, unified digital-twin and fully interactive neural free-view work outside `v0.8.0`;
8. preserve heavy-backend isolation from the production `depth` environment;
9. keep the shared Live 3D `Mesh (Poisson)` control distinct from Reconstruction mesh processing;
10. reserve the coordinated version transition and annotated `v0.8.0` tag for 8.9I.

## Expected overlay

Before H3 is committed, the working tree must contain exactly the six H2 files plus the four H3 qualification files. H3 itself is qualification-only.

## Pass gate

Run:

```bash
./scripts/run_phase8_9h3_release_candidate_tests.sh quick
./scripts/run_phase8_9h3_release_candidate_tests.sh affected
```

Both commands must pass with the strict repository pre/post checks unchanged. After that, H2 + H3 may be committed and pushed together as the clean release-preparation checkpoint for 8.9I. Do not create the `v0.8.0` tag in H3.
