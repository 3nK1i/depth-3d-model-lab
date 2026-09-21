# I6.4B — Manual Provider Selection State & Controller Composition

## Purpose

I6.4B composes the execution-free I6.4A provider availability/qualification contract into the existing `ReconstructionController` as explicit **manual user intent**. It does not activate a dense provider and does not change reconstruction execution.

## Locked semantics

- Recommendation, selection, and execution remain three separate concepts.
- The selectable catalog is composed only from the explicit I6.3 frozen evidence binding; I6.4B performs no runtime provider probing and introduces no live evidence-context resolver.
- A concrete provider is manually selectable only when the current source is ready, the evidence context is explicit, runtime availability is explicit, and exact-context evidence qualifies that provider.
- Advisor recommendation is surfaced in the provider-selection snapshot but is never copied into manual selection.
- A valid manual selection survives passive state refreshes while it remains eligible.
- Source changes clear previous manual intent fail-closed.
- Evidence/context/availability changes clear a now-invalid selection fail-closed rather than allowing it to reappear later.
- `Auto` remains outside the concrete provider catalog and production Auto remains unqualified, unauthorized, and non-selectable.

## Controller additions

The existing controller now exposes:

- `reconstruction_provider_selection_changed`
- `reconstruction_provider_selection_state`
- `reconstruction_provider_selection_snapshot`
- `select_reconstruction_provider(provider_id)`
- `clear_reconstruction_provider_selection()`

These APIs record and publish product intent only. They are not referenced by the offline worker launch path or dense-provider execution paths.

## Authority / safety boundaries

I6.4B does not:

- execute or activate a dense provider;
- mutate provider settings or reconstruction defaults;
- start/cancel/retry reconstruction;
- qualify or authorize production Auto;
- create provider rankings or scalar scores;
- change PyCOLMAP SfM camera/global-frame authority;
- claim physical scale or certified metrology.

## Validation intent

Focused tests cover contract composition, exact-context qualification, recommendation/selection separation, manual selection, stale-context invalidation, source-change invalidation, deterministic composition, controller signal deduplication, and preservation of predecessor controller behavior. Full unit regression remains part of the authoritative closure runner.
