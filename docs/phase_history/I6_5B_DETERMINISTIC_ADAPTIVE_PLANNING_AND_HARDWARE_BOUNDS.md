# I6.5B — Deterministic Adaptive Planning & Hardware Bounds

## Scope

I6.5B resolves the frozen I6.5A frame/view request contract into a deterministic,
resource-aware product plan. It reuses the already-qualified I4.3F2 pre-SfM
resource heuristic rather than changing or duplicating that historical planner.

This step is execution-free. It does not sample source frames, mutate ingestion,
modify the controller/UI/worker, start reconstruction, retry until a result
passes, select a dense provider, activate production Auto, or add progressive
SfM visualization.

## Resolution semantics

- **Adaptive** — uses the I4.3F2 resource target for the current source/resource
  snapshot. The target is a per-run planning decision, never a permanent global
  cap.
- **Manual limit** — the requested value is a maximum. The effective target is
  deterministically bounded by known source count and the resource planning
  target when available.
- **All eligible** — never receives a silent adaptive cap. Resource pressure is
  recorded explicitly and execution remains subject to the existing resource
  preflight boundary.

## Hardware provenance

`FrameViewHardwareBounds` records the resource snapshot, raw I4.3F2 target,
source-clamped resource target, planning capacity units, the transparent work
coefficient, and a deterministic SHA-256 hardware-context identity. These are
planning heuristics, not empirical runtime/memory guarantees.

`DeterministicFrameViewPlan` combines the frozen I6.5A `plan_input_sha256` with
that hardware-context identity to produce `planning_provenance_sha256`.

## Safety / authority

- no permanent global frame/view cap;
- no GPU-brand-specific policy branching;
- no request-mode substitution;
- no retry-until-pass or silent post-failure replanning;
- planned views are not PyCOLMAP-registered cameras;
- PyCOLMAP remains sole canonical camera/global-frame authority;
- no metric-scale or certified-metrology claim;
- no provider-selection or production-Auto effect.

## Qualification

Focused tests cover deterministic resolution, source/resource clamping,
manual-limit semantics, all-eligible resource-pressure reporting, unknown
resource behavior, hardware-provenance digests, source-size safety, and stress
ranges. I6.5A, I4.3F, and I6.4 safety boundaries are retained as regressions.
