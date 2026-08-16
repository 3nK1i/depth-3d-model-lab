# Phase 9.6G — GPU / Resource Coordination

Baseline: `8f7fee07a53b35fb30d52cba222f7b001e58272a`

## Scope

Phase 9.6G closes the application-wide resource-ownership gap between the existing `ReconstructionController` interlock and Phase 9.6 Digital Twin runtimes that intentionally live outside that controller. It adds one dependency-light ownership snapshot shared by the Reconstruction page and the existing MainWindow admission boundary.

Heavy/exclusive owners are serialized:

- active Reconstruction runtime;
- active E1 external Digital Twin build (DA3/NeRF);
- isolated interactive NeRF renderer.

Interactive Gaussian is tracked as bounded dynamic viewer residency rather than an external heavyweight CUDA backend, but it still counts as a Reconstruction-page resource owner for cross-mode admission until explicitly released. Static Gaussian/NeRF previews and persisted scene metadata do not reserve runtime resources.

Starting Reconstruction or an E1 DA3/NeRF build performs one centralized, explicit appearance-resource handoff. Interactive NeRF is stopped through the existing blocking isolated-worker shutdown path, Gaussian staging/display buffers are released, and the heavyweight operation fails closed if release cannot be confirmed. Conversely, interactive Gaussian/NeRF admission is blocked while Reconstruction or an E1 build owns resources.

MainWindow now observes these Phase 9.6 page-owned resources in the same reciprocal application resource boundary already used for Benchmark, Snapshot, Stable Video and inspection/export work.

The Build & Processing card surfaces the current resource owner and a clearly labelled startup GPU snapshot. The snapshot is not live process telemetry and does not create a CUDA monitoring loop in the UI process.

## Authority invariants

- resource coordination never changes canonical geometry authority;
- DA3 remains supporting geometry only;
- Gaussian and NeRF remain appearance-only;
- resource state never establishes metric scale, measurement authority, QC authority or certified metrology;
- external DA3/Nerfstudio runtime isolation remains unchanged;
- no DA3, gsplat, Nerfstudio, PyCOLMAP or CUDA backend is imported by the new coordinator.

## Manual acceptance

1. Load an existing persistent NeRF Digital Twin and expand `BUILD & PROCESSING` plus Digital Twin details. With no dynamic renderer/job active, confirm `Resource owner` reports available/idle and `GPU snapshot` is explicitly a startup snapshot.
2. Start **Interactive NeRF**. Confirm the resource owner changes to interactive NeRF and other long-running application modes (for example Benchmark) are navigation/admission locked while the isolated renderer is active.
3. From the same Reconstruction page start **Rebuild NeRF** with the already-qualified Efficient profile. Confirm G first releases/stops the interactive NeRF renderer, then the E1 build becomes the sole owner. No co-resident interactive NeRF worker may remain.
4. Cancel the rebuild after it reaches a live build state; wait for cancellation/terminal cleanup. Confirm the owner returns to available/idle and application mode navigation is released.
5. Show only the static NeRF Preview and confirm it does not become a resource owner or lock other modes.
6. Confirm the persistent canonical scene, scale wording, measurement authority and QC/metrology wording remain unchanged throughout the handoffs.

A full DA3 rerun or completed NeRF retraining is not required for G; those external workflows were already qualified in E2/E3. G qualifies ownership, release, admission and cleanup behavior.
