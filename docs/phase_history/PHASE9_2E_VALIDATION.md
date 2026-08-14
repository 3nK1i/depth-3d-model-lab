# Phase 9.2E — Interactive SAM2 Prompt Refinement

Phase 9.2E extends the Phase 9.2D Auto Segment workspace with explicit point-prompt interaction while preserving all semantic/geometry authority boundaries.

## Scope

- `+ Segment Object`: one positive image click creates a new unlabeled SAM2 proposal.
- Selected-mask refinement: `+ Include` and `− Exclude` points are accumulated and applied through SAM2.
- Refinement automatically includes an interior anchor from the selected mask so exclude-only edits remain object-anchored.
- Refined masks preserve class and instance identity but return to `Pending` review.
- Prompt execution remains isolated in `QProcess`; the Qt GUI thread does not own SAM2 inference.
- New-instance prompted requests are explicitly parentless instead of fabricating a fake parent detection/mask.
- Existing prompted-refinement calls keep their original parent-required contract.

## Authority invariants

Interactive SAM2 masks and semantic labels remain semantic evidence only. They do not establish canonical geometry authority, metric scale, measurement authority, inspection/QC authority, or certified metrology.

## Deferred

- Box-prompt drawing and brush/polygon manual editing are not introduced in 9.2E.
- Workspace/mask persistence remains deferred to the dedicated Phase 9.2 persistence step.
- Accepted semantic objects are not yet injected into Measure & Inspect analysis in this step.

## Validation

Run:

```bash
./scripts/run_phase9_2e_tests.sh quick
./scripts/run_phase9_2e_tests.sh integration
./scripts/run_phase9_2e_tests.sh backward
python scripts/run_phase9_2e_prompt_smoke.py
python scripts/run_phase9_2e_prompt_ui_preview.py
```

Real target validation should additionally use the existing `sam2.1_t.pt` runtime on the RTX 2060 and verify one-click creation plus positive/negative refinement inside the desktop application.
