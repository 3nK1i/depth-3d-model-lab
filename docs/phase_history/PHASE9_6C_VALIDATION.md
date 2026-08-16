# Phase 9.6C — Cross-Representation Navigation & Synchronization

Phase 9.6C adds a read-only navigation/synchronization projection over the
qualified Reconstruction viewport and the Phase 9.6B scene composition.

## Scope

- The existing Reconstruction/VisPy turntable camera remains the single shared presentation-navigation source.
- Interactive Gaussian co-navigates because it is rendered in the existing canonical VisPy scene graph.
- Interactive NeRF continues to derive render requests from `ReconstructionCanvas.interactive_render_camera_state()`.
- The unified navigation status projection feature-detects that camera-snapshot API and fails closed without blocking canonical scene loading when a legacy/degraded canvas or test double does not expose it.
- Static Gaussian/NeRF previews are explicitly reported as static; they are not falsely described as camera-synchronized.
- Same-scene reload keeps the existing Phase 9.3I orbit/pan/zoom preservation policy.
- Different-scene binds keep the existing one-final-fit policy after scene-bound resources are ready.
- Scene switching continues to release/hide Gaussian and NeRF before persistent scene identity changes.
- Canonical geometry remains the only spatial, picking, and measurement authority.
- Persisted camera poses remain supporting context and are not scale evidence.
- No metric-scale, QC, appearance-authority, or certified-metrology claim is introduced.
- No reconstruction controller, canvas renderer, persistence schema, backend runtime, measurement, or QC contract is changed.

## Validation gates

```bash
./scripts/run_phase9_6c_tests.sh quick
./scripts/run_phase9_6c_tests.sh integration
./scripts/run_phase9_6c_tests.sh backward
```

Manual validation is limited to orbit/pan/zoom/fit/reset across Canonical,
interactive Gaussian and interactive NeRF, plus one same-scene reload and one
different-scene switch when suitable persisted scenes are available.
