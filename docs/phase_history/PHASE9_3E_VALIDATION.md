# Phase 9.3E — Object Inspection + Measurement Overlays

## Scope

Phase 9.3E adds a bounded 3D evidence callout for the currently selected Phase 9.3D semantic object. It reuses the established Phase 8.8 E1/E2/E3/E4 selection, measurement, validity/uncertainty and inspection/QC presentation contracts; it does not create a new measurement or QC engine.

Implemented behavior:

- binds the viewport overlay to the exact selected persistent semantic object and the active canonical scene/session;
- requires the exact E2 source/observation binding and E3 validity evidence before any numeric measurement value is exposed in the 3D viewport;
- renders available measurements as source-authored value ± expanded uncertainty, source bounds when present, and source reliability;
- renders rejected/unavailable measurements as status + source reason only, with no numeric value;
- presents only the exact selected-object E4 QC result; an attached report with no exact object result is shown as `NOT EVALUATED` and never inherits the scene decision;
- anchors the evidence callout to the validated Phase 9.3D canonical semantic object without altering the canonical surface;
- follows the existing semantic-object visibility/selection lifecycle and clears on scene replacement/viewport shutdown;
- preserves existing Reconstruction sidebar E2/E3/E4 information unchanged.

No measurement dimension box/arrow is generated in this step. The existing Phase 7.5 measurement box is camera/depth-frame evidence and the current E2/E3 presentation contract does not carry an authoritative transform from that frame into the persistent canonical reconstruction frame. Reusing semantic support bounds as dimensions would also fabricate measurement geometry. 9.3E therefore shows truthful source evidence as a 3D callout only.

## Authority and safety

- Canonical surface geometry remains the authoritative spatial scene.
- Semantic objects remain `semantic_context` only.
- E2 alone never places numeric values in the viewport; numeric display requires E3 source validity + uncertainty.
- Rejected/unavailable E3 values never display a numeric value.
- E4 object QC is copied only for an exact source `candidate_object_id`; otherwise the object is `NOT EVALUATED`.
- Measurement/QC evidence does not promote canonical-scene scale.
- Metric source measurements may be shown while the canonical scene remains arbitrary gauge, with both statuses stated explicitly.
- No semantic support bounds are reinterpreted as measurement geometry.
- No camera-space Phase 7.5 box is placed into canonical 3D without a qualified transform.
- `certified_metrology` remains `false`.
- No YOLO/SAM2, reconstruction, DA3, Gaussian or NeRF backend is launched by persistent scene loading or overlay construction.

## Focused automated validation

```bash
./scripts/run_phase9_3e_tests.sh quick
./scripts/run_phase9_3e_tests.sh integration
./scripts/run_phase9_3e_tests.sh backward
python scripts/run_phase9_3e_canonical_inspection_smoke.py
```

Coverage includes exact scene/object/provenance binding, E2 fail-closed numeric behavior, E3 uncertainty/rejection semantics, exact-object E4 QC behavior, persistent canonical-semantic integration, visibility/selection clearing, and focused E2/E3/E4 + 9.3D backward sanity.

## Real VisPy manual validation

```bash
python scripts/run_phase9_3e_canonical_inspection_preview.py --show
```

Verify in the production `depth` environment:

- the canonical mesh and Phase 9.3D selected semantic surface remain unchanged and aligned;
- E2-only evidence produces no numeric 3D callout;
- after E3 attachment, available values show `value ± uncertainty` and rejected values show reason without a number;
- the callout states source scale and canonical-scene scale separately;
- exact selected-object QC appears, while a different selected object without a source feature result shows `NOT EVALUATED`;
- changing object, Clear, Prev, Next and Focus update/clear the callout correctly;
- the existing semantic visibility control hides/shows the callout with semantic presentation;
- Orbit, Pan, Zoom, Fit and Reset remain responsive;
- layout remains usable at 1366×768;
- no measurement arrows/boxes appear from semantic support bounds;
- scene replacement and viewport shutdown clear all 9.3E overlay resources without a crash;
- no heavy backend starts during persistent scene loading.

Professional independent scene-layer controls remain Phase 9.3F scope.
