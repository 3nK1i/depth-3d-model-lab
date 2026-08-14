# Phase 9.2C — Auto Segment Engine / Proposal Generation Foundation

Phase 9.2C adds a synchronous core Auto Segment engine above the existing perception registry.
It does not add UI, training, semantic geometry authority, metric authority, or measurement authority.

## Scope

- Adds an explicit automatic-segmentation adapter capability/task.
- Extends the existing Ultralytics SAM 2.1 adapter with no-prompt segment-everything proposal generation.
- Keeps the existing prompted SAM refinement path intact.
- Adds dependency-light raw automatic mask proposal contracts.
- Adds deterministic proposal filtering/bounding and conversion to unlabeled `SemanticWorkspace` objects.
- Reuses the Phase 9.2B project class catalog without assigning fake classes to proposals.
- Keeps accepted proposals compatible with the existing `ObjectObservation` contract.
- Keeps the existing perception model IDs and runtime/environment requirements unchanged.

## Authority

Automatic proposals are semantic review candidates only:

- geometry authority: false
- metric authority: false
- measurement authority: false
- certified metrology: false by implication; this phase introduces no metrology authority

## Resource posture

The SAM descriptor remains non-live and prefers isolated execution. Phase 9.2C does not introduce
YOLO+SAM co-residency or change measured-VRAM policy. UI execution/off-GUI-thread orchestration is a
later Phase 9.2 step.

## Validation

Run:

```bash
./scripts/run_phase9_2c_tests.sh quick
./scripts/run_phase9_2c_tests.sh integration
./scripts/run_phase9_2c_tests.sh backward
python scripts/run_phase9_2c_auto_segment_smoke.py
```

A real SAM 2.1 offline smoke can additionally be run with an actual image and the efficient profile.
