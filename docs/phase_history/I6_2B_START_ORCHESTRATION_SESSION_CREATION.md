# I6.2B — Start Orchestration & Session Creation

## Purpose

I6.2B wires the closed I6.2A One-Start preflight contract into the existing
`ReconstructionController.start_selected_offline()` path. It does not create a
new worker, new Reconstruction page, new reconstruction algorithm, dense
provider, production Auto policy, persistence format, or camera authority.

The existing Reconstruction UI continues to call the same public controller
entrypoint. The existing isolated offline worker remains the execution owner.

## Admission / launch sequence

One offline Start action now uses this order:

1. Allocate one session UUID.
2. Run passive I6.2A source/runtime/output/storage preflight.
3. Reject before resource handoff if passive evidence is blocked.
4. Evaluate the existing application offline-resource guard exactly once.
5. Re-run I6.2A preflight with the same session UUID, source and output root.
6. Preserve a returned application-resource conflict as preflight evidence.
7. Build `OfflineReconstructionConfiguration` from the committed preflight.
8. Delegate through the existing `start_offline()` seam.
9. Start the existing `OfflineReconstructionController` worker.
10. Existing session/workspace signals expose the same session/output identity.

The worker still rechecks output-directory emptiness at the final process
boundary, preserving the previous stale-result race protection.

## Authority boundaries

- PyCOLMAP SfM remains sole canonical camera/global-frame authority.
- Execution scope remains `sparse_sfm_only`.
- Dense execution is not authorized.
- Sparse results are not promoted to persistent canonical scenes.
- Production Auto remains deferred, unqualified and unauthorized.
- No provider selection/ranking or global scalar provider score is introduced.
- Certified metrology is not claimed.
- The established 8 GiB storage safety floor from I6.2A is unchanged.
- Hardware adaptation remains owned by the existing offline worker pre-SfM path.

## Compatibility

I6.2B preserves the historical public `start_offline()` delegation used by the
B3 runtime-control contract. The old B3 executable fixture is updated to carry
an executable bit because I6.2A formally qualified executable-runtime checking.
The historical I6.2A test asserting "not wired yet" is advanced to verify that
the A contract remains dependency-light while B consumes it only in the
controller; the Reconstruction page still does not import preflight logic.

## Qualification gates

Automated closure requires:

- I6.2B focused static/controller orchestration tests;
- I6.2A regressions;
- I6.1 contract/presentation regressions;
- Start-workflow static and Qt predecessors;
- full unit regression;
- import/compile/shell/hygiene/scope checks.

I6.2B also requires one real UI-launched image-sequence reconstruction before
closure. That real run is evidence of orchestration only; it does not qualify
dense reconstruction, universal SfM robustness, metric scale, or production
Auto.
