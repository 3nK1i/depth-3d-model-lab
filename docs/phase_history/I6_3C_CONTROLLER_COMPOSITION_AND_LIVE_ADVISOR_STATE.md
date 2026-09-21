# I6.3C — Controller Composition & Live Advisor State

## Scope

I6.3C composes the frozen I6.3A request/evidence contract and I6.3B pure advisory engine into the existing `ReconstructionController` as passive, deduplicated live product state.

The controller now exposes an immutable Advisor state and a change signal. Current source/workspace/admitted sparse-result facts are read from existing I6.1/I6.2 authorities. Frozen provider evidence, descriptive resources, provider availability, and the evidence-context token enter through an explicit `ReconstructionAdvisorEvidenceBinding`.

## Locked boundaries

- No live scene/context classifier is introduced or qualified.
- No provider is selected or activated.
- No reconstruction setting is mutated.
- No Start action or worker execution is triggered.
- No production Auto behavior is introduced, qualified, or authorized.
- No provider ranking or scalar scoring is introduced.
- Hardware/resources remain descriptive only.
- PyCOLMAP SfM remains canonical camera/global-frame authority.
- A prior result is reused only when its recorded source URI matches the currently selected source and workspace result identity is consistent.
- SfM usability is true only when the admitted matching result explicitly reports a usable quality assessment.

## Live recomputation

Advisor state is recomputed when the passive offline workspace or admitted offline result changes, and when an explicit frozen evidence binding is installed/cleared. Equal snapshots are deduplicated before signal emission.

## Deferred to I6.3D

No Advisor widget/panel or user-facing recommendation presentation is added in I6.3C. I6.3D owns professional UI binding and manual UX qualification.
