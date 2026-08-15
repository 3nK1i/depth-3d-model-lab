# Phase 9.5C — Real NeRF Backend Integration / Optimization Execution

Phase 9.5C connects the qualified Phase 9.5A isolated Nerfstudio/Nerfacto runtime to the Phase 9.5B canonical-scene NeRF input bundle through the already-proven Phase 8.7E2 worker.

## Scope

- real isolated Nerfstudio/Nerfacto execution
- bounded qualification optimization and checkpoint creation
- checkpoint reload / preview render proof
- process-group supervision with `SIGTERM -> SIGKILL`
- exact runtime/check-out provenance and artifact integrity evidence
- NeRF remains `appearance_only`

## Explicitly deferred

- persistent NeRF representation: 9.5D
- interactive NeRF viewer: 9.5E
- NeRF controls / scene layers: 9.5F
- performance and GPU resource lifecycle: 9.5G
- final real-scene quality / Phase 9.5 closure: 9.5H

Phase 9.5C does not claim quality convergence, production render performance, metric geometry, measurement authority, QC authority, or certified metrology.

## Target-machine gate

Use the real Phase 9.5B South Building bridge and the isolated `eia-nerf-nerfstudio` environment. The real result must prove CUDA execution, OOM-free bounded training, a non-empty config/checkpoint/preview, exact pinned checkouts, and no heavy NeRF modules in `depth`.
