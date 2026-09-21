# I6.4A — Production Provider Availability & Selection Contract

## Purpose

I6.4A freezes the execution-free product contract for production dense-provider availability and selection. It does not wire controller state, probe runtimes, expose UI controls, or execute a dense provider.

The contract intentionally keeps these facts separate:

1. **Provider identity** — one of the ten canonical concrete dense-provider implementations already established by I4.3/I5.
2. **Runtime availability** — an explicit observation (`unknown`, `available`, `unavailable`).
3. **Context qualification** — an explicit evidence claim (`unresolved`, `qualified_for_context`, `not_qualified_for_context`) bound to an exact context and SHA-256 evidence provenance.
4. **Advisor recommendation** — passive guidance only.
5. **User selection** — explicit intent only; still not provider activation or execution.

`Auto` is not inserted into the concrete provider catalog. It is represented by a separate fail-closed gate.

## Canonical provider identities

The contract preserves the ten concrete identities already used by the I5 evidence/Auto work:

- COLMAP MVS
- Depth Anything 3 (DA3)
- RollingDepth
- Video Depth Anything
- Metric3D v2
- UniDepthV2
- MoGe-2
- Depth Pro
- Depth Anything V2
- Marigold

The canonical provider-ID mapping is regression-checked against the frozen I5.13B mapping. No global ranking or universal winner is created by catalog order.

## Selection eligibility

`manual_selection_allowed=True` is valid only when the provider is both:

- explicitly runtime-available, and
- qualified for the exact evidence context.

Availability alone is not qualification. Qualification alone is not runtime availability. Recommendation does not imply selection. Selection does not imply execution.

I6.4A permits a snapshot to carry both a recommended provider and a different manually selected provider so later product layers cannot accidentally collapse Advisor guidance into user intent.

## Auto boundary

The I5 offline/shadow Auto policy remains qualified. Production Auto remains:

- not qualified,
- not authorized,
- non-selectable.

Attempting to construct an enabled production Auto gate fails closed.

## Authority / non-effects

I6.4A does not:

- probe a provider runtime;
- select or activate a provider in production;
- execute dense reconstruction;
- mutate settings;
- start reconstruction;
- modify a production default;
- create a global provider rank or scalar score;
- establish metric scale or certified metrology;
- change PyCOLMAP SfM camera/global-frame authority.

## Next

I6.4B may compose explicit manual selection state into the existing controller while preserving the I6.4A non-execution boundary.
