# I6.5A — Adaptive Frame/View Contract & Provenance

## Purpose

I6.5A freezes the product-facing contract for adaptive pre-SfM frame/view planning without changing the existing I4.3F production planner, ingestion behavior, controller, worker, or Reconstruction UI.

The purpose is to make the existing source-planning behavior explicit and auditable before I6.5B productizes deterministic planning and I6.5C introduces user override semantics.

## Four separate facts

The contract intentionally keeps these concepts distinct:

1. **Source provenance** — immutable source identity, source kind/URI, SHA-256 source fingerprint, estimated source-view count and expected frame rate.
2. **Requested plan** — what the product/user asked for: `adaptive`, `all_eligible`, or `manual_limit`.
3. **Effective plan resolution** — the effective target resolved for this source/run, including any resource-derived target and engineer-visible reason codes.
4. **Realized view decisions** — per-source-view provenance recording whether a view was planned, eligible but not planned, or ineligible, with explicit reasons.

A requested plan is not silently treated as the effective plan, and a planned view is not silently treated as a successfully registered camera.

## Planning modes

### Adaptive

`adaptive` carries no user max-view cap. A later resolver may produce a run-specific effective target from explicit resource/source evidence. I6.5A does not compute that target.

### All eligible

`all_eligible` carries no cap. A realization using this mode must plan every observed eligible source view. This remains subject to ordinary ingestion validity: an ineligible/decode-failed frame is not fabricated into an eligible view.

### Manual limit

`manual_limit` requires an explicit requested maximum of at least three views. The effective target may be equal to or lower than the request, but never higher. The reason for any reduction remains explicit.

No mode introduces a permanent global view cap.

## Provenance identity

`FrameViewPlanResolution.plan_input_sha256` is a deterministic SHA-256 identity over the source provenance, requested mode, effective/resource targets, reason codes, and I6.5A schema/phase identity.

`FrameViewPlanRealization.plan_provenance_sha256` extends that identity with every realized per-view record in authoritative source order.

Changing a reason code, source fingerprint, target, disposition, locator, timestamp, or source-view identity changes the corresponding digest. These hashes are provenance identities; they are not quality scores or provider rankings.

## Relation to the existing I4.3F adaptive planner

I4.3F already provides the production `AdaptivePreSfmFramePlan` and resource-aware target computation. I6.5A does not replace or modify it.

The integration regression proves that an existing I4.3F2 target can be represented by the new I6.5A resolution contract without changing the legacy planner. I6.5B may then productize a deterministic resolver around this frozen contract.

## PyCOLMAP authority boundary

The frame/view flow is:

```text
source views
    ↓
I6.5 requested/effective frame-view plan
    ↓
planned views submitted to SfM
    ↓
PyCOLMAP SfM registration
    ↓
registered cameras / canonical global frame
```

Therefore:

- `planned_view_count` is **not** `registered_keyframes`;
- a `PLANNED` record is **not** a camera-pose authority claim;
- I6.5A contains no registered-camera count;
- PyCOLMAP remains the sole canonical camera/global-frame authority.

## Non-effects

I6.5A does not:

- execute frame sampling;
- modify ingestion or keyframe selection;
- alter I4.3F resource heuristics or scientific thresholds;
- introduce a fixed global frame/view cap;
- modify the controller, worker, or UI;
- select/activate/execute a dense provider;
- activate production Auto;
- start reconstruction;
- add progressive SfM visualization (I6.6 scope);
- claim metric scale or certified metrology.

## Next

I6.5B may implement deterministic adaptive planning and hardware bounds on top of this contract while preserving full-span sampling, explicit provenance, source-order semantics, and the no-retry-until-pass rule.
