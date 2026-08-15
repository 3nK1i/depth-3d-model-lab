# Phase 9.3I — Performance + Resource Lifecycle + UX Hardening

Baseline: clean pushed `phase9-digital-twin` at `5f28daab483067fdd4010db92b0538b4731643e8` (Phase 9.3H closed).

## Scope

9.3I hardens the already-qualified Phase 9.3A-H canonical viewer. It does not add a reconstruction provider, persistence schema, calibration requirement, measurement algorithm, or new authority source.

Implemented behavior:

- coalesces the former canonical-mesh Fit + camera-context Fit pair into **one final Fit** after a different persistent scene is fully bound;
- exact same-scene reloads preserve the user's orbit/pan/zoom state and restore the selected semantic object when that object still exists;
- a successfully validated non-empty canonical surface replaces the prior non-empty surface **in the existing VisPy `MeshVisual`**, avoiding a detach/recreate allocation cycle on ordinary reload/switch;
- the qualified 9.3B detach/recreate lifecycle is still used for a genuine canonical-surface clear, so no zero-vertex `MeshVisual.set_data()` call is introduced;
- switching directly between two valid non-empty semantic object surfaces reuses the selected-highlight `MeshVisual`; detach/recreate remains reserved for a real clear/unavailable surface;
- adds one collapsed `SCENE HEALTH` group with actual UI-side persistent-scene bind latency, currently materialized viewer resource counts, and lifecycle state;
- scene-health diagnostics are explicitly presentation-only and never benchmark DA3/reconstruction/GPU backends or alter authority;
- no calibration is required. Arbitrary-scale/legacy scenes continue to load and operate unchanged.

## Authority / compatibility invariants

- `PersistentReconstructionScene` remains canonical geometry authority.
- Viewer resource reuse never changes canonical geometry identity or scale semantics.
- Semantic selection preservation remains semantic context only.
- Measurement/QC evidence and uncertainty/rejection semantics are unchanged.
- `certified_metrology=false` unless separately qualified by the existing authority path.
- DA3 remains supporting geometry evidence only; no DA3/Gaussian/NeRF runtime is launched by scene binding or scene-health diagnostics.
- Same-scene UX preservation is additive. Loading a different scene still clears stale semantic selection and performs a fresh bounded Fit.

## Automated validation

```bash
./scripts/run_phase9_3i_tests.sh quick
./scripts/run_phase9_3i_tests.sh integration
./scripts/run_phase9_3i_tests.sh backward
python scripts/run_phase9_3i_scene_health_smoke.py
```

The focused suites cover bind-policy correctness, diagnostics-only authority, actual persistent canonical resource counts, one-fit scene binding, same-scene view/selection preservation, different-scene stale-selection clearing, 1366x768 sidebar containment, and affected 9.3C/9.3F/9.3H compatibility.

## Real PyQt/VisPy lifecycle/performance validation

```bash
python scripts/run_phase9_3i_scene_lifecycle_preview.py --reload-cycles 12 --show
```

The preview uses the qualified real VisPy runtime and deterministic persistent canonical artifacts. It must verify:

1. the first mesh+camera scene bind performs one final Fit rather than two sequential Fits;
2. repeated exact same-scene reloads keep the same canonical `MeshVisual` object;
3. same-scene reloads preserve camera center/distance and semantic selection;
4. object-to-object canonical semantic highlight switching reuses its non-empty highlight `MeshVisual`;
5. binding a different scene clears stale semantic selection while replacing the canonical surface in place;
6. `SCENE HEALTH` remains readable and horizontally unclipped at 1366x768;
7. reported median/p95 values are identified strictly as **UI bind time**, not backend/GPU reconstruction performance;
8. layer controls, provider inspection, Orbit/Pan/Zoom/Fit/Reset and semantic navigation remain responsive;
9. normal shutdown releases viewer/controller resources without a VisPy crash;
10. no YOLO/SAM2/reconstruction/DA3/Gaussian/NeRF heavy backend starts.

No fixed latency number is claimed from synthetic/CI environments. Real values printed by the production `depth` environment are evidence for that machine/run only.

## Deferred to 9.3J

- real heavy DA3 execution and real-scene reconstruction quality/performance qualification;
- real physical calibration/metric-scale qualification where available;
- uncertainty-aware measurement-frame → canonical-frame registration and genuine canonical 3D dimension placement;
- final real-scene end-to-end Phase 9.3 qualification/closure.
