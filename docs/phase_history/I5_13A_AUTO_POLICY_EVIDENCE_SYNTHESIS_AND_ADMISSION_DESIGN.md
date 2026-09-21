# I5.13A — Evidence Synthesis + Auto-Policy Admission / Design

## Purpose

I5.13A is the first boundary at which the project may design an evidence-driven `Auto` dense-provider selection policy. It is still execution-free: it does not run SfM, MVS, DA3, temporal depth, single-image depth, Hybrid, matched-view rendering, or any provider inference, and it does not select a provider or change the production default.

The step binds the frozen I4.3G cross-provider qualification closure and the I5.1–I5.12 robustness/closure evidence into one conservative policy-design contract. `Auto` remains a **selection policy**, never an executable provider ID. PyCOLMAP remains the sole authoritative camera/global reconstruction frame.

## Evidence basis

The admission review binds:

- I4.3G5B formal cross-provider qualification closure;
- I5.1C difficult-handheld closure;
- I5.2C low-overlap / weak-parallax closure;
- I5.3C blur / exposure / clutter / symmetry closure;
- I5.4C forward-motion / KITTI closure;
- I5.5C dynamic-scene proxy closure;
- I5.6C COLMAP MVS robustness closure;
- I5.7C RollingDepth robustness closure;
- I5.8C Video Depth Anything robustness closure;
- I5.9C remaining single-image depth-provider closure;
- I5.10D MP-SfM/matching-policy closure;
- I5.11D bounded failure/recovery closure;
- I5.12A Hybrid evidence-justification review;
- the frozen post-I5.12A source checkpoint;
- an I5.13A regression/admission receipt.

No terminal quality event is converted into a score, win count, or global ranking.

## Conservative v1 policy principle

The admitted policy profile is `evidence_bound_conservative_v1`:

> **Select only when exact frozen evidence leaves one defensible automatic primary; otherwise abstain.**

Abstention is therefore a valid and expected Auto result. The policy must not force a provider merely because `Auto` was requested.

### Hard gates

Automatic selection requires:

1. authoritative PyCOLMAP SfM is available and production-usable;
2. system/runtime integrity is clean;
3. the provider is available;
4. the evidence context is recognized and bounded;
5. the candidate is not the Hybrid family.

A sparse-SfM failure is never masked by choosing a dense provider. In particular, the admitted I5.4 forward-motion case remains a hard abstention boundary, and I5.10 does not authorize Auto to switch the SfM matcher.

### Evidence eligibility

- `PASS` evidence may make a provider eligible only within the exact admitted evidence context.
- `DEGRADED` evidence is not a clean automatic-primary qualification.
- `REJECTED` evidence is ineligible for automatic primary selection.
- preserved historical rejections remain active;
- later PASS evidence from a distinct scenario does not overwrite an earlier exact-scenario rejection;
- conflicting/mixed evidence blocks automatic-primary use until a later boundary resolves it;
- unknown or unqualified contexts abstain.

Examples preserved by the design include RollingDepth's historical temporal-motion rejection despite a current Castle PASS envelope, VDA's capacity-bounded degraded current envelope, and the historical low-texture rejections for Depth Anything V2 and Marigold.

### No ranking or hidden tie-break

The first Auto policy may not create:

- a global scalar score;
- provider win counts;
- a cross-family ranking;
- a universal provider winner;
- a runtime-performance tie-break;
- a GPU-model performance ranking.

If more than one provider remains equally admissible and there is no frozen direct scenario-specific preference that resolves the choice, Auto abstains.

### Fallback and engineer override

Fallback must be explicit, must be allowed by the request, and must be independently evidence-eligible for the same context. There is no implicit global fallback chain and no retry-until-PASS behavior.

An engineer override remains distinct from automatic selection. It must be explicit and provenanced and cannot rewrite the evidence record.

## I5.13B/C/D boundaries

A passing I5.13A admission authorizes only **I5.13B — deterministic evidence-bound Auto policy engine**. B is offline/shadow only: it may consume frozen evidence-context tokens but may not execute providers or integrate Auto into production.

I5.13C remains unauthorized until B is frozen. If authorized, C replays the policy offline against frozen evidence only and must demonstrate deterministic same-input/same-decision behavior without provider execution or result-driven rule tuning.

I5.13D remains the only boundary allowed to decide whether production Auto integration is justified. I5.13A does not authorize production integration.

## Authority and claim boundaries

I5.13A preserves:

- PyCOLMAP camera/global-frame authority;
- arbitrary SfM scale unless later calibration establishes physical scale;
- no provider promotion or production-default change;
- no Hybrid execution;
- no matcher/default changes;
- no metric-scale, measurement-authority, metrology, or surface-mesh claim;
- no universal reconstruction/provider robustness claim.
