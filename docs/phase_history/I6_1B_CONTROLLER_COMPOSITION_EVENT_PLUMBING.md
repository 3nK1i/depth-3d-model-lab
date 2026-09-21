# I6.1B — Controller Composition & Event Plumbing

## Purpose

I6.1B connects the dependency-light I6.1A offline reconstruction workspace contract to the existing application-level `ReconstructionController` without creating a second controller, changing reconstruction algorithms, or activating production Auto.

## Scope

The existing `ReconstructionController` now exposes a passive `offline_workspace_snapshot` and an `offline_workspace_changed` signal. The snapshot is recomposed from already-authoritative state owned by the existing session model, offline worker controller, and persistence controller.

The controller retains only the latest offline worker protocol mapping for workspace stage/progress presentation. The event remains forwarded unchanged through the pre-existing `offline_event_received` signal. Repeated snapshots are deduplicated, and a new `STARTING` state clears stale stage/progress from a previous run before workspace recomposition.

Workspace recomposition responds to:

- existing application/session-model changes,
- accepted offline-result evidence,
- persistent authority changes, and
- offline worker progress/status events.

Session/result identity conflicts remain explicit through the I6.1A contract and are never merged or hidden.

## Explicit non-goals

I6.1B does not:

- modify `ReconstructionPage` or any UI widget,
- activate, qualify, or authorize production Auto,
- select or rank dense providers,
- change PyCOLMAP camera/global-frame authority,
- modify reconstruction algorithms or thresholds,
- modify persistent-scene/checkpoint formats,
- add pause/resume behavior,
- change calibration or metric/metrology claims, or
- implement I6.2 One Start Reconstruction.

## Test strategy

The step is test-first. A static red state demonstrates the missing controller signal/property/event-composition seam before implementation. Focused Qt integration tests then cover initial composition, source selection, passive worker-event forwarding and deduplication, stale-progress clearing on a new run, sparse partial-result composition, identity-conflict visibility, and persistent-authority-driven scale recomposition.

The validation runner also retains I6.1A regression coverage, dependency-free G1–G4 predecessor coverage, affected controller/UI predecessors, compile/import probes, and an optional full unit-suite closure mode.

## Authority invariants

- Canonical camera/global-frame authority remains `pycolmap_sfm`.
- Production Auto scope remains `offline_shadow_only`.
- Production Auto remains unqualified and unauthorized.
- Certified metrology remains false.
- Negative/conflicting evidence remains visible.
