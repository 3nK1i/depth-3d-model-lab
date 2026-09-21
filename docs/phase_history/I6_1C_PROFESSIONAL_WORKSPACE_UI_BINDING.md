# I6.1C — Professional Workspace UI Binding & I6.1 Closure Gate

## Purpose

I6.1C binds the passive I6.1A offline-workspace snapshot, already composed and
emitted by I6.1B, into the existing professional Reconstruction page. It does
not create another page/controller, execute reconstruction, change provider
selection, alter persistence, or activate production Auto.

## Product presentation

The existing top-level hierarchy is preserved. Two compact status rows are
added rather than introducing a developer/debug panel:

- **Workspace** in `SOURCE & WORKFLOW` presents the product-level offline state
  (`Offline idle`, `Ready to reconstruct`, `Reconstructing`, `Partial result`,
  `Scene ready`, or a fail-closed warning/error state).
- **Authority** in `RUNTIME STATUS` presents the canonical camera/scale boundary
  (`PyCOLMAP` plus unqualified/arbitrary/metric/calibrated-metric scale).

Tooltips progressively disclose stage/progress, partial-vs-persistent scene
meaning, checkpoint linkage, PyCOLMAP camera authority, the lack of certified
metrology, and the fact that production Auto remains deferred/offline-shadow
only.

## Scientific and architectural boundaries

I6.1C preserves all I5/I6.1 invariants:

- PyCOLMAP SfM remains the sole canonical camera/global-frame authority.
- Sparse arbitrary-scale output is never presented as metric/certified.
- Partial output is never presented as a persistent canonical scene.
- Identity conflicts are visible and fail closed.
- Production Auto is not activated, qualified, or authorized.
- Existing Start/Pause/Stop, source selection, persistence, provider selection,
  recovery, viewer, reconstruction algorithms, defaults, and thresholds are
  unchanged.

## Test-first evidence

The pre-implementation focused tests failed because
`app.reconstruction_offline_workspace_presentation` did not exist. After the
narrow implementation, dependency-light focused/regression validation passes.
The authoritative Qt/UI and full-unit closure run must be executed in the
project `depth` environment.

## Manual UI acceptance for I6.1C

This is the first I6 step requiring a real application UI check. It is a
presentation/binding acceptance only; a new heavy reconstruction experiment is
not required here. The first deliberate real end-to-end Start Reconstruction
exercise belongs to I6.2.

Run at the normal target desktop resolution (1366×768 is the minimum reference
used by prior Reconstruction UX qualification):

1. Launch `python main.py`; switch to **Reconstruction**.
2. Confirm no backend starts merely by entering the page.
3. Confirm `Workspace` initially reads **Offline idle** and `Authority` reads
   **PyCOLMAP • scale unqualified**.
4. Hover `Authority`; verify the tooltip states PyCOLMAP camera/global-frame
   authority, **Certified metrology: no**, and **Production Auto: deferred**.
5. Choose a valid **Image sequence** (or Recorded video) but do not press Start.
   Confirm `Workspace` becomes **Ready to reconstruct** and Start becomes
   available through the pre-existing control logic.
6. Confirm source selection alone does not launch the offline worker.
7. Switch to **Live source** and verify the offline workspace returns to
   **Offline idle** rather than implying an offline reconstruction is active.
8. Return to the offline source and confirm the two new rows remain readable,
   do not clip, do not create excessive blank vertical space, and leave the
   Reconstruction viewport visibly usable.
9. Open the Authority/Workspace tooltips and verify no debug-heavy raw payloads
   are exposed.
10. Close the application normally.

I6.1C may be closed only after the automated closure receipt passes and this
manual checklist is reported PASS.
