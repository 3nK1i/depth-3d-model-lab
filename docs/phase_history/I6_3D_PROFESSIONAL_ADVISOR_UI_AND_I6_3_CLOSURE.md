# I6.3D — Professional Reconstruction Advisor UI & I6.3 Closure

## Scope

I6.3D is the presentation-only closure step for the I6.3 Reconstruction Advisor. It exposes the already-qualified I6.3A contract, I6.3B deterministic evidence-bound reasoning, and I6.3C passive controller/live-state composition in the existing Reconstruction settings sidebar.

The Advisor remains guidance only. It does not select or activate a dense provider, mutate settings, trigger Start, execute a worker, authorize production Auto, create a global provider ranking, create a scalar provider score, change PyCOLMAP camera/global-frame authority, or claim certified metrology.

## Product presentation

A compact `RECONSTRUCTION ADVISOR` disclosure group is placed immediately after the existing Workflow group so guidance is visible without creating a third top-level runtime card or widening the 1366×768 layout. It contains no buttons or provider selectors.

The group presents:

- **Status** — `Recommendation available` or `No recommendation`.
- **Guidance** — the exact-context provider label for a recommendation, or an intentional abstention/waiting state.
- **Evidence** — qualitative support only (`Exact qualified`, `Bounded`, `Insufficient`, or `Conflicting`).
- **Why** — the evidence-bound engine explanation.
- **Boundary** — `Advisory only • no settings changed`.

Abstention is not presented as a reconstruction runtime failure. Waiting states are neutral; evidence ambiguity, unavailability, integrity blockers, or bounded support are warnings.

## Safety and authority boundaries

- Exact unique PASS evidence remains required for a recommendation.
- Ambiguous PASS evidence remains abstention; UI does not rank or tie-break.
- Provider availability can block a recommendation but cannot rewrite evidence.
- Hardware observations remain descriptive and cannot break ties.
- The live evidence-context resolver remains unqualified.
- Production Auto remains unqualified, unauthorized, and inactive.
- PyCOLMAP SfM remains sole canonical camera/global-frame authority.
- Advisor state has no provider-selection, execution, Start, or settings effect.

## Qualification

Focused unit/static tests cover presentation semantics and source-level no-action binding. Qt UI tests cover the initial no-source state, ready-source/SfM waiting state, exact-evidence recommendation, stale-source invalidation, and ambiguous-evidence abstention. Closure also reruns I6.3A-C, I6.2D UI, page/runtime predecessors, and the complete unit suite in the authoritative `depth` environment.

A real UI qualification is required for final I6.3 closure. Because no production live-context resolver is qualified, the default production UI is expected to abstain unless an explicit frozen evidence binding is supplied. The recommendation UI path is qualified through controller-driven Qt tests without granting production effects.
