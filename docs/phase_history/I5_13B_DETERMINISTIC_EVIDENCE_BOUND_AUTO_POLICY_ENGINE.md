# I5.13B — Deterministic Evidence-Bound Auto Policy Engine

## Status
Implementation step for the first executable Auto policy **in offline/shadow mode only**.

## Purpose
I5.13B implements the I5.13A policy design without integrating Auto into production execution. The engine consumes explicit frozen evidence context tokens and produces the existing `DenseProviderSelectionDecision` contract plus an explainability/provenance record.

## Locked rules
- Auto is a selection policy, never an executable provider.
- PyCOLMAP remains the sole camera/pose authority.
- Automatic selection requires a clean authoritative SfM gate, clean runtime integrity, a recognized frozen context, and exactly one clean PASS provider in that exact evidence context.
- Multiple clean PASS providers are an ambiguity and therefore abstain; availability is not used as an invented evidence tie-break.
- DEGRADED/REJECTED/unknown evidence is never promoted to automatic primary.
- Historical exact-scenario rejections remain active.
- Forward-motion sparse rejection hard-abstains before dense-provider selection.
- Hybrid selection and automatic matching-policy changes are forbidden.
- Engineer override is explicit, non-automatic, provenance-preserving, and cannot bypass missing/invalid authoritative SfM or runtime-integrity gates.
- Fallback never means retry-until-pass and cannot silently promote degraded evidence.

## Offline evidence tokens
The engine compiles only evidence already frozen by I5.13A, including:
- bounded current MVS envelope;
- current admitted Castle temporal envelope (RollingDepth PASS, VDA DEGRADED);
- historical temporal motion/static-sweep terminal evidence;
- historical G4 single-image complete-room / low-texture / weak-overlap matrices.

This is **not** a live scene classifier. Production scene/context classification is deferred and not authorized by I5.13B.

## Deterministic self-probes
The formal B freeze verifies:
1. unique MVS evidence selects COLMAP MVS;
2. unique clean temporal evidence selects RollingDepth over degraded VDA;
3. ambiguous single-image low-texture evidence abstains;
4. forward-motion rejected sparse context hard-abstains;
5. unknown context abstains;
6. explicit engineer override remains non-automatic.

Every probe is executed twice with identical inputs and must produce byte-equivalent decision payloads.

## Scope exclusions
No provider execution, SfM/MVS execution, scene rematerialization, production Auto wiring, production default modification, provider promotion, runtime/GPU ranking, global score, hybrid execution, matcher change, metric-scale claim, measurement authority, or metrology claim is permitted here.

## Next
A clean I5.13B freeze authorizes **I5.13C frozen-evidence offline replay only**. Production integration remains deferred to I5.13D.
