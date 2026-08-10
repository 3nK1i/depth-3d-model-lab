# Phase 8.8C4 Validation — Live Keyframe / Map Structure Visualization

Baseline: `phase8-reconstruction` at `fc9d426a7454f14dfbf4b1609d91956453a7a1a1` (Phase 8.8C3).

## Scope

C4 adds bounded historical SLAM keyframe camera poses to the existing C1–C3 reconstruction viewer. It does not add dense reconstruction, metric scale, world alignment, DA3, Gaussian Splatting, NeRF, or metrology authority.

The native Stella worker remains isolated and protocol version 1. It may attach the optional `keyframes_reconstruction` array to a live map snapshot together with C3 sparse landmarks. Each sample carries only a keyframe ID, timestamp, `T_reconstruction_from_camera`, and explicit pose convention. The worker emits at most 256 keyframes; the UI renders at most 96 historical frustums with deterministic first/last-preserving decimation.

Historical keyframes use a subdued amber presentation so they remain visually distinct from the current red camera, cyan live trajectory, pale sparse landmarks, true scene-space XYZ origin, and bottom-right orientation gizmo.

## Safety / compatibility

- `LiveSlamObservation` remains unchanged.
- `LiveMapSnapshot` remains unchanged.
- `LiveMapGeometrySnapshot` is backward-compatibly extended with optional keyframe samples.
- Protocol version stays `1`; old workers/observations without keyframe samples remain valid.
- Empty optional keyframe arrays are accepted across the normalized metadata wire round-trip.
- Keyframe poses remain arbitrary-gauge reconstruction coordinates; no metric claims are introduced.
- Keyframe visibility is presentation-only and survives a new session while keyframe geometry itself is cleared.
- Native worker sample cap: 256.
- Viewer frustum cap: 96.

## Required validation

1. `./scripts/run_phase8_8c4_tests.sh quick`
2. `./scripts/run_phase8_8c4_tests.sh protocol`
3. `./scripts/run_phase8_8c4_tests.sh backward`
4. `python scripts/run_phase8_8c4_keyframe_preview.py`
5. Rebuild isolated worker with `./scripts/build_stella_live_worker.sh` and confirm C4 keyframe metadata.
6. `python scripts/run_phase8_8c4_live_keyframe_smoke.py --max-frames 500 --playback-speed 2.0`
7. Normal `python main.py` sanity, then `git diff --check`.
