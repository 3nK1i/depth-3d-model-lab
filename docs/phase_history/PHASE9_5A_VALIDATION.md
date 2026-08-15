# Phase 9.5A — NeRF Foundation & Runtime Isolation

## Baseline

Required clean baseline: `7e9ba95897085f00aa034ac2bc628b92eed6f16d` on `phase9-digital-twin`.

## Scope

Phase 9.5A promotes the already-qualified Phase 8 Nerfstudio/Nerfacto backend into an explicit Phase 9 interactive-runtime foundation. It does **not** train NeRF, render a NeRF view, author a checkpoint, persist a new NeRF artifact, or change the Reconstruction UI.

The Phase 8 foundation is reused rather than duplicated:

- isolated environment: `~/anaconda3/envs/eia-nerf-nerfstudio`;
- Nerfstudio checkout: `~/Desktop/eia-nerf-backends/nerfstudio`;
- tiny-cuda-nn checkout: `~/Desktop/eia-nerf-backends/tiny-cuda-nn`;
- Nerfstudio 1.1.5 exact qualified revision: `6b60855003011b2ca23c2fe3f8e2ca6314c69924`;
- tiny-cuda-nn exact qualified revision: `b3473c81396fe927293bdfd5a6be32df8769927c`;
- Python 3.10.x, Torch 2.1.2, CUDA 11.8 runtime family;
- existing Phase 8 process-group supervision with SIGTERM → SIGKILL escalation.

## Authority

NeRF remains `appearance_only`.

It has no canonical-geometry, metric-scale, measurement, QC, reference, or certified-metrology authority. There is no Gaussian↔NeRF fallback substitution. Future Phase 9.5 viewing/picking must retain canonical geometry as the authority.

## Runtime policy

The production `depth` environment must not directly import or install Nerfstudio or tiny-cuda-nn. Runtime probing is read-only and external. Missing runtime evidence yields `UNAVAILABLE`; pinned-version/revision drift, dirty qualified checkouts, interpreter leakage, unsupported compute capability, or production heavy-module leakage yields `REJECTED`.

## 9.5A acceptance

- exact runtime/checkouts are qualified read-only;
- checkout cleanliness is verified;
- process policy is process-group isolated, `shell=False`, SIGTERM → SIGKILL;
- backend/license provenance is explicit;
- capability probe creates no NeRF artifact and performs no optimization/training/rendering;
- no production UI/persistence schema changes;
- focused, integration, backward, deterministic smoke, and real isolated probe pass.
