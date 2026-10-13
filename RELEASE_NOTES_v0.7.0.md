# Depth & 3D Model Lab v0.7.0

## Phase 7 Stable

Phase 7 promotes the application from a Live 3D model viewer to a multi-mode depth-analysis workstation.

### Added

- Interactive Benchmark Mode using the Phase 6 isolated-process benchmark engine
- Snapshot Mode for one frozen frame with atomic RGB, depth, point-cloud and metadata export
- Stable Video Mode for Video, USB Camera and RTSP sources
- Native temporal-model support and generic stabilization for stateless models
- Latest-frame backpressure, dropped-frame reporting and synchronized RGB/depth video export
- Release-time validation of real Benchmark, Snapshot and Stable Video outputs

### Preserved

- All seven production depth models
- Registry-driven adapter construction
- RTX 2060 and RTX 5090 automatic profiles
- Worker crash, timeout and OOM recovery
- Image, Video, USB Camera and RTSP source lifecycle
- Phase 5 and Phase 6 automated regression coverage

### Deferred

Reconstruction, camera pose estimation, persistent mapping, SLAM, SfM, calibration and multi-camera fusion remain planned for later phases.
