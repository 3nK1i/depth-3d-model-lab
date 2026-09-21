# I6.5C — Controller Composition & Explicit User Override Semantics

## Scope

I6.5C composes the frozen I6.5A frame/view provenance contract and I6.5B deterministic resource-aware planner into the existing `ReconstructionController`. It introduces controller-owned planning **intent only**. No worker sampling, SfM execution, provider activation, production Auto, or UI behavior is changed in this step.

## Product semantics

The controller owns a `FrameViewPlanRequest` with default `ADAPTIVE` intent. An explicit user request may switch to `ALL_ELIGIBLE` or `MANUAL_LIMIT`. The requested intent and the effective deterministic plan remain separate facts.

A source change preserves the user's requested mode but recomputes source provenance and the effective plan. The old plan digest is never retained for the new source. A resource-snapshot change likewise recomputes the deterministic plan without changing the request.

`MANUAL_LIMIT` remains a maximum, not an exact execution count. `ALL_ELIGIBLE` remains uncapped and can expose resource pressure without being silently converted to Adaptive.

## Resource input

I6.5C does not probe hardware. The controller accepts an explicit `AdaptiveSelectionResourceSnapshot`; until one is supplied, a valid offline source reports `WAITING_RESOURCES`. I6.5D will own the professional UI wiring/qualification of this state.

## Source identity

For controller invalidation, I6.5C creates a deterministic metadata-manifest fingerprint (`selection_metadata_manifest_v1`) from the selected offline source. This is planning provenance, not a content-evidence hash and not a PyCOLMAP camera-registration fingerprint.

## Authority and non-effects

- planned views are not registered cameras;
- PyCOLMAP remains the sole canonical camera/global-frame authority;
- no source sampling is executed;
- no worker or ingestion behavior is changed;
- no provider selection/activation or production Auto behavior is added;
- no scientific thresholds, metric scale, or certified metrology claims are changed;
- progressive SfM visualization remains I6.6.
