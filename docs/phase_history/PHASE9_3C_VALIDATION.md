# Phase 9.3C — World Frame + Camera Frustums + Trajectory

## Scope

Phase 9.3C extends the existing Reconstruction viewport with persistent canonical-scene camera context while preserving the Phase 9 authority model and all prior persistence contracts.

Implemented behavior:

- reuses the existing canonical reconstruction origin and viewer-only Z-up transform;
- loads camera poses only when the persistent `camera_trajectory` artifact explicitly contains bounded `T_reconstruction_from_camera` matrices in its existing generic `payload_snapshot`;
- renders dedicated persistent camera frustums, camera centers and trajectory in the existing VisPy `ReconstructionCanvas`;
- includes persistent cameras and the Phase 9.3B canonical surface in Fit-to-scene bounds;
- clears/reloads camera presentation deterministically when persistent scenes change;
- shows truthful World/Cameras status in the existing `CANONICAL SCENE` section;
- keeps legacy persistent scenes valid when camera matrices were not persisted.

No persistent-scene schema version was changed. No reconstruction controller, backend registry, semantic-object, measurement, QC, DA3, Gaussian or NeRF architecture was added or promoted by this step.

## Authority and safety

- Canonical geometry remains the authoritative spatial reference.
- Persistent camera poses/frustums/trajectory are **supporting geometry / presentation context**, never canonical geometry authority.
- Viewer Z-up conversion is presentation-only and does not change the reconstruction/world coordinate contract.
- Camera intrinsics/poses do not establish metric scale.
- Metric qualification remains separate Phase 9.1 authority evidence.
- `certified_metrology` remains `false`.
- Missing legacy camera matrices produce `Camera trajectory not persisted`; no poses are fabricated from frame IDs or provenance metadata.
- Invalid persisted pose matrices fail closed and stale camera visuals are cleared without revoking valid canonical-geometry authority.
- A bounded maximum of 4096 persisted camera poses is enforced; display frustums are additionally decimated for responsive presentation.
- No reconstruction, DA3, Gaussian or NeRF heavy backend is launched by the 9.3C presentation path.

## Focused automated validation

```bash
./scripts/run_phase9_3c_tests.sh quick
./scripts/run_phase9_3c_tests.sh integration
./scripts/run_phase9_3c_tests.sh backward
python scripts/run_phase9_3c_world_camera_smoke.py
```

The focused suites cover explicit camera-pose snapshot validation, legacy compatibility, authority separation, persistent round-trip behavior, scene switching/clearing, controller load/reload, existing trajectory/frustum visibility controls and the 1366x768 sidebar contract.

## Real VisPy manual validation

```bash
python scripts/run_phase9_3c_world_camera_preview.py --show
```

Verify on the production `depth` environment:

- the integrity-verified canonical surface remains visible;
- the canonical reconstruction world origin/axes remain visible and Z-up only in presentation;
- persisted camera frustums and connected camera trajectory are visible in the same 3D scene;
- trajectory and keyframe/frustum toolbar controls hide/show the corresponding persistent camera context;
- Orbit, Pan, Zoom, Fit and Reset remain responsive;
- Fit frames both canonical surface and camera context;
- `CANONICAL SCENE` reports World and Cameras truthfully;
- the synthetic preview remains `VIEWER Z-UP • ARBITRARY GAUGE` and does not claim metric scale or certified metrology;
- no horizontal sidebar overflow appears at 1366x768;
- closing the preview releases both canonical mesh and camera viewer resources;
- no reconstruction/DA3/Gaussian/NeRF backend is started.

Phase 9.3D semantic-object work remains deferred to the next locked roadmap step.
