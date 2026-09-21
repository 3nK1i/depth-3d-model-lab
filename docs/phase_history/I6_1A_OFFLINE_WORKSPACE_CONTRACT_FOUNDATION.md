# I6.1A — Offline Reconstruction Workspace Contract Foundation

## Purpose

I6.1A introduces the dependency-light, immutable product-layer workspace contract for I6 Offline Reconstruction Productization. It composes existing reconstruction state; it does not introduce a second controller, persistence format, provider system, viewer, or runtime.

## New contract

`app/reconstruction_offline_workspace.py` defines `OfflineReconstructionWorkspaceSnapshot` and `build_offline_reconstruction_workspace_snapshot()`.

The snapshot passively exposes:

- existing reconstruction session identity and explicit identity conflicts;
- offline workflow/source readiness for image sequences and recorded video;
- offline lifecycle, output identity, latest stage/progress/message;
- partial sparse-result versus persistent-scene availability;
- checkpoint presence, next stage, and existing resume-allowed evidence;
- fixed PyCOLMAP canonical camera-authority ownership;
- camera-pose availability and canonical-geometry authority;
- conservative scale state: unqualified, arbitrary, metric, or calibrated metric;
- explicit non-certification of metrology;
- explicit production-Auto deferral (`offline_shadow_only`, not qualified, not authorized).

## Authority and safety boundaries

I6.1A is descriptive only. It does not:

- start, cancel, pause, resume, or recover reconstruction jobs;
- modify `ReconstructionController` or `ReconstructionPage`;
- modify persistence/checkpoint formats;
- select or rank dense providers;
- activate or qualify production Auto;
- modify reconstruction algorithms, thresholds, defaults, cameras, or poses;
- claim certified physical metrology.

The contract fails closed on conflicting session/result identities and refuses direct construction that attempts to change PyCOLMAP camera authority, authorize production Auto, or claim certified metrology.

## Test-first evidence

The first focused run was intentionally executed before the production module existed. Collection failed with `ModuleNotFoundError: app.reconstruction_offline_workspace`, establishing the expected red state.

After the narrow implementation, the focused I6.1A unit/integration suite passed with 15 tests. Four dependency-free predecessor suites covering the existing Phase 8.8 G1-G4 session/checkpoint/recovery composition also passed with 32 tests.

The current inspection container does not provide PyQt6. Therefore the PyQt/controller/persistence predecessor batch cannot collect here and is intentionally preserved as environment-limited evidence rather than reported as a product failure. `scripts/run_i6_1a_tests.sh all` must be executed in the project `depth` environment before I6.1A is formally closed.

## Validation entrypoints

Focused contract validation:

```bash
bash scripts/run_i6_1a_tests.sh focused
```

Dependency-free predecessor validation:

```bash
bash scripts/run_i6_1a_tests.sh predecessors-pure
```

Full validation in the project `depth` environment, with formal receipt:

```bash
bash scripts/run_i6_1a_tests.sh all /path/to/i6_1a_validation_receipt.json
```

I6.1A is not the controller/event-plumbing step. Controller composition and event caching belong to I6.1B; PyQt workspace presentation belongs to I6.1C.
