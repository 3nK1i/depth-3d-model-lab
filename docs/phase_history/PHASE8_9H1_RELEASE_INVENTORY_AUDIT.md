# Phase 8.9H1 — Release Inventory & Read-Only Audit

## Baseline

- Branch: `phase8-reconstruction`
- HEAD / origin baseline: `dff62357874e28de8d0ceadb9a222dd0cd17d999`
- Runtime/package version before release transition: `0.7.5`
- Release name before release transition: `Phase 7.5 Stable`
- `v0.8.0` local tag: absent
- `v0.8.0` remote tag: absent
- Working tree: clean

The source audit was collected read-only and the final safety check confirmed that branch, HEAD, origin, worktree and tag state were unchanged.

## Findings

1. `core/version.py`, `pyproject.toml` and `config/app.yaml` are mutually consistent at `0.7.5`; they must remain unchanged through 8.9H.
2. `README.md` still described the product primarily as the Phase 7.5 measurement/inspection release and did not provide a Phase 8 Reconstruction release-candidate overview.
3. No `RELEASE_NOTES_v0.8.0.md` existed.
4. The repository contains complete Phase 8 validation history through 8.9G3 plus RTSP/UI post-fix qualification and the final post-fix whole-app requalification.
5. Release-facing authority wording needed to distinguish canonical geometry, semantic context, measurement/QC authority and Gaussian/NeRF appearance-only representations.
6. Known-size calibration language needed to move from "after Phase 8" to explicit post-`v0.8.0` future qualification.
7. The shared Live 3D `Mesh (Poisson)` statement needed clarification because Phase 8 now provides mesh processing inside Reconstruction without enabling that legacy render-card action.

## H2 scope lock

H2 is documentation/release-preparation only:

- modify `README.md`
- add `RELEASE_NOTES_v0.8.0.md`
- add this H1 audit record
- add `PHASE8_9H2_RELEASE_PREPARATION.md`
- add the H2 release-prep test runner and contract test

H2 must not modify production code, canonical version metadata or Git tags.
