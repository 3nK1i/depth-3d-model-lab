# I6.4D — Professional Provider / Auto UI & I6.4 Closure

## Purpose

I6.4D productizes the already-qualified I6.4A-C provider-selection contracts in
the Reconstruction page without broadening their authority.

The page gains a distinct **PROVIDER & AUTO** disclosure group.  This group is
separate from the older Phase 9.3G **ADVANCED RECONSTRUCTION** provider inspector:

- the Phase 9.3G control inspects persisted/known provider provenance;
- the I6.4D control records explicit **future production provider intent** only.

The two controls must never be conflated.

## Manual provider UX

The manual selector contains only concrete providers that the I6.4A/B state
marks both:

1. explicitly runtime-available; and
2. qualified for the exact frozen evidence context.

The selector never infers either fact.  A provider recommendation from the I6.3
Advisor may be marked in the selector, but recommendation does not become
selection.  An explicit user choice calls only
`ReconstructionController.select_reconstruction_provider()` and records intent;
it does not activate or execute the provider, change settings, or start work.

The UI also presents:

- current selection lifecycle/status;
- current Advisor recommendation, when one exists;
- eligible-provider count with per-provider status available by tooltip;
- an explicit boundary statement;
- a Clear action that clears manual intent only.

Source/context invalidation remains fail-closed through the I6.4B controller.

## Production Auto UX

Production Auto is visible so the user can understand the product boundary, but
it remains disabled.  I6.4D preserves I6.4C exactly:

- offline/shadow Auto: qualified;
- production Auto: not qualified;
- production Auto authorized: no;
- production Auto selectable: no;
- live context resolver qualified: no;
- Auto engine wired into app: no;
- Auto orchestration entry point: no.

There is no Auto click handler and no production Auto activation API.

## Compatibility

I6.4D does not modify the I6.4A contract, I6.4B controller/state contract, or
I6.4C Auto gate.  It also preserves the frozen I6.3D first-six primary sidebar
ordering.  The new **PROVIDER & AUTO** group is appended immediately after that
frozen primary stack, before the remaining legacy settings groups.

No worker, reconstruction algorithm, dense provider implementation, PyCOLMAP
camera/global-frame authority, scale semantics, or certified-metrology claim is
changed.

## Qualification

Automated qualification covers:

- pure presentation mapping for no-source, eligible, recommended, selected and
  invalidated states;
- visible fail-closed Auto state;
- static proof that the new manual handler contains no Start/activation path;
- Qt binding of the controller selection signal;
- explicit manual choice and clear behavior;
- source-change invalidation;
- Auto remaining disabled even when a manual provider is eligible/selected;
- frozen I6.3D sidebar order and predecessor UI/runtime controls;
- I6.4B/I6.4C regression boundaries;
- the full unit suite.

Manual UI qualification is required because the Reconstruction page changes.
The normal production application is expected to show zero selectable providers
until an explicit qualified evidence binding exists; I6.4D does not invent a
live evidence-context resolver merely to make the selector active.

## I6.4 closure boundary

If automated and manual qualification pass, I6.4A-D may close as one milestone.
That closure means the application can represent and record safe manual provider
intent and can honestly expose the current Auto qualification state.  It does
**not** mean production Auto or dense-provider execution has been authorized.
