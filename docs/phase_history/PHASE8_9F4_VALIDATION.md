# Phase 8.9F4 — Real NeRF / Nerfacto Qualification

## Scope

Release-validation only. No production `core/`, `app/`, configuration, dependency, or external-backend source is changed.

F4 proves that the already-qualified isolated Nerfstudio/Nerfacto backend can consume a real appearance dataset derived from the **real 8.9F2 South Building PyCOLMAP trajectory**, produce held-out real-scene appearance evidence, and produce a hardened renderable checkpoint with measured RTX 2060 backend performance.

## Truthful chain

- 8.9F2 PyCOLMAP SfM is the real pose source for F4.
- 24 registered cameras are selected evenly across the real 80-camera trajectory.
- Intrinsics/distortion are derived from the returned SfM camera model; no calibration values are hardcoded.
- The expected `SIMPLE_RADIAL` model is undistorted to the existing EIA pinhole neural-scene contract and bounded to a 720 px long edge.
- The existing Phase 8.7E3 split contract provides 20 training views and 4 held-out views.
- Existing E3 held-out quality runs for 1000 bounded steps.
- Existing E4 performance/artifact hardening runs for 1000 bounded steps and 8 render repetitions.
- F3 Gaussian must already be qualified, but its `scene.pt` is **not** used as Nerfacto geometry input.
- F2 canonical geometry remains authoritative provenance but is **not** silently used as NeRF metric authority.
- NeRF remains `appearance_only` and `certified_metrology=false`.

## Authority

- canonical geometry: authoritative
- Gaussian: appearance only
- NeRF / Nerfacto: appearance only
- certified metrology: false

## Isolation

Nerfstudio and tiny-cuda-nn execute only in `/home/eiatech/anaconda3/envs/eia-nerf-nerfstudio`. Production `depth` must not import Nerfstudio, tiny-cuda-nn, gsplat, DA3, or PyCOLMAP.

Qualified pins:

- Nerfstudio 1.1.5: `6b60855003011b2ca23c2fe3f8e2ca6314c69924`
- tiny-cuda-nn: `b3473c81396fe927293bdfd5a6be32df8769927c`

## Real acceptance outputs

`/tmp/phase8_9f4_real_nerf/` contains:

- `bridge/dataset.json`
- `bridge/split.json`
- `bridge_summary.json`
- `quality/summary.json`
- `quality/heldout_contact_sheet.png`
- `performance/summary.json`
- a hardened Nerfstudio `config.yml`
- a real Nerfacto checkpoint
- final `summary.json`

The F4 runner requires the real F2 and F3 `/tmp` evidence to still exist. Runtime artifacts are never committed.
