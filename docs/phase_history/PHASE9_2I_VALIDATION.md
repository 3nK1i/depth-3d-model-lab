# Phase 9.2I — Custom Segmentation / Auto Segment Workspace Closure

## Scope

Phase 9.2I is a qualification/closure step. It adds no production feature, UI,
semantic authority, model runtime, persistence format, or release identity. It
closes the software implementation delivered by Phase 9.2A-H and prepares the
branch for Phase 9.3 — Interactive Canonical 3D Scene.

Changed-path scope:

- `docs/phase_history/PHASE9_2I_VALIDATION.md`
- `scripts/run_phase9_2i_closure_smoke.py`
- `scripts/run_phase9_2i_tests.sh`
- `tests/integration/test_phase9_2i_phase_closure.py`

No files under `app/`, `core/`, `config/`, `requirements/`, or release metadata are
modified by 9.2I.

## Closure invariants

The combined closure smoke reuses the already-qualified Phase 9.2A, B, C, E, F,
G and H non-UI smokes and requires all of the following:

- accepted semantic objects preserve their reviewed masks;
- class identity and instance identity remain separate;
- segmentation projects round-trip without an external database;
- Auto Segment proposals remain genuinely unlabeled until reviewed;
- prompted SAM2 creation/refinement remains semantic-only and refinement resets review to Pending;
- deterministic multi-mask merge produces one semantic union, keeps provenance, and resets review to Pending;
- accepted reviewed masks can feed Measure & Inspect directly while perception/YOLO inference is skipped;
- depth still executes when reviewed masks are the semantic source;
- workspace/project state persists and restores with reviewed identities intact;
- used custom-class retirement returns affected objects to Unlabeled/Pending;
- retired stable class IDs are never silently reused;
- semantic state never becomes geometry, metric, measurement, QC, or certified-metrology authority.

Phase 9.2D and later UI behavior is covered by the Phase 9.2 UI suites and the real
manual validation already performed during the feature steps. 9.2I does not add a
new UI and does not repeat the entire manual application regression.

## Automated gates

- `./scripts/run_phase9_2i_tests.sh quick`
  - all Phase 9.2 unit/UI tests plus the 9.2I closure integration contract.
- `./scripts/run_phase9_2i_tests.sh integration`
  - all Phase 9.2 integration tests plus affected calibration/alignment/semantic-geometry integration gates.
- `./scripts/run_phase9_2i_tests.sh backward`
  - inherited Phase 5, 6, 7, 7.5, Phase 8 backward gates and the complete Phase 9.1 quick gate.
- `./scripts/run_phase9_2i_tests.sh full`
  - complete current-tree unit, UI, integration, and stress suites followed by the inherited backward matrix.
- `python scripts/run_phase9_2i_closure_smoke.py`
  - combined dependency-light semantic-workspace closure smoke.

## Real validation evidence carried into closure

Phase 9.2 closure relies on the real validations already executed during A-H rather
than fabricating or unnecessarily repeating them. These include real RTX-2060 SAM2
Auto Segment execution, professional Auto Segment UI use, interactive prompted
refinement, deterministic plant/pot mask merge, reviewed-mask Measure & Inspect
handoff with perception inference skipped, and persisted workspace/custom-class
management including retired-ID non-reuse.

No new real-artifact claim is introduced by 9.2I.

## Authority posture

Semantic labels, masks, project classes, reviewed objects, merge provenance and
persisted semantic workspaces are context/evidence only. They do not establish
canonical geometry, metric scale, measurement validity, inspection/QC authority or
certified metrology. Depth, calibration, scale evidence and the existing measurement
validity/uncertainty pipeline retain their established authority roles.

## Release posture

Phase 9.2I must not change the application release identity from v0.8.0 and must not
create a v0.9.0 tag. Phase 9 continues on `phase9-digital-twin` until the final Phase 9
release gate.
