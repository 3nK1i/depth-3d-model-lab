# I6.4C — Production Auto Gate & Qualification Boundary

## Purpose

I6.4C exposes one production-facing Auto qualification boundary for the later
I6.4D UI without enabling Auto execution.  It preserves the final I5 result as
the authority for what is and is not qualified.

The bounded conclusion remains:

- offline/shadow Auto policy: **qualified**;
- offline frozen-evidence replay: **qualified**;
- production Auto: **not qualified**;
- production Auto authorization: **no**;
- production Auto selectable: **no**.

This is a successful fail-closed product gate, not a failure of the offline Auto
policy.

## Source authority preserved

I6.4C intentionally binds its product semantics to the frozen I5 contracts:

- policy profile: `evidence_bound_conservative_v1`;
- qualification authority: `I5.14B` final I5 closure;
- decision: `defer_production_auto`.

The I5 production blockers remain explicit:

1. live-context -> frozen-evidence-token mapping is not qualified;
2. a production scene/context resolver is not implemented/qualified;
3. adaptive production planning remains explicit-provider-only;
4. production UI Auto activation is not implemented;
5. the offline Auto engine is not wired into the application;
6. no production Auto orchestration entry point exists.

Future production Auto work must separately qualify a live context resolver,
abstention outside the frozen evidence envelope, full decision/evidence
provenance through orchestration, and shadow validation before any live
activation.

## Product contract

`core/reconstruction/reconstruction_production_auto_gate.py` adds an immutable
`ProductionAutoQualificationBoundary` and builder.  It consumes the current
I6.4A/I6.4B descriptive selection snapshot only.

Manual provider selection is allowed to coexist with this boundary but never
becomes Auto.  Reading the boundary grants no provider selection, activation,
execution, settings mutation, Start action, or production-default change.

`ReconstructionController.reconstruction_production_auto_gate` exposes the
boundary read-only for later presentation.  There is deliberately no
`select_reconstruction_auto`, `activate_reconstruction_auto`, or Auto execution
method, and the application does not import the I5 Auto engine.

## TDD / compatibility evidence

The I6.4C tests were introduced before the implementation and initially failed
because the production-Auto gate module did not exist.  After the narrow
implementation, focused pure/static/stress tests pass.

Regression coverage includes:

- I6.4B manual provider intent and controller composition;
- I6.4A provider availability/qualification contract;
- I5.13D production Auto integration decision;
- I5.14B final I5 closure.

The I5 production-surface regression remains important: I6.4C must not make the
old evidence-only Auto engine appear as a live production entry point.

## Explicit non-goals

I6.4C does not:

- implement a live context resolver;
- classify production scenes;
- import/evaluate the I5 Auto engine from the application;
- select or activate a provider;
- execute dense reconstruction;
- change provider settings;
- start reconstruction;
- change production defaults;
- create a global provider ranking or scalar score;
- change PyCOLMAP camera/global-frame authority;
- claim metric scale or certified metrology;
- change the Reconstruction UI.

## Next step

After authoritative validation and immutable checkpointing, I6.4D may present
manual provider selection and the disabled/deferred Auto gate professionally in
the Reconstruction UI.  The UI must not broaden the I6.4C qualification state.
