# I5.13D — Auto Policy Production Integration Qualification / Decision

## Purpose

I5.13D is the final boundary of I5.13. It consumes the qualified I5.13C frozen-evidence replay and inspects the current production integration surface. It does **not** wire Auto into production, execute a provider, rerun SfM/MVS, classify a live scene, rematerialize a scenario, tune policy rules, or modify a production default.

The question is narrower: **does the evidence accumulated through I5.13C justify production Auto activation in the current application architecture?**

## Evidence result entering D

I5.13C established that the frozen `evidence_bound_conservative_v1` policy is deterministic over the admitted replay envelope:

- 10 predeclared frozen-evidence cases;
- 2 exact unique selections;
- 8 evidence-driven abstentions;
- exact provider IDs and ordered reason-code sequences reproduced;
- same input -> same decision;
- no provider/SfM/MVS execution;
- no result-driven rule tuning;
- no production effect.

That is sufficient to qualify the **offline/shadow policy behavior**. It is not, by itself, evidence that a live production reconstruction can be mapped safely to the frozen evidence-context tokens consumed by the engine.

## Production integration inventory

The current production surface preserves the pre-Auto architecture:

- `DenseProviderSelectionMode.AUTO` exists as an architecture contract;
- the I5.13B engine consumes explicit frozen sparse/dense evidence-context IDs;
- no production scene/context classifier is implemented or qualified;
- no live reconstruction -> evidence-context-token mapping is qualified;
- `prepare_adaptive_provider_execution()` still builds `EXPLICIT_PROVIDER` requests and records `provider_selected_by_auto=False`;
- the Reconstruction provider control remains inspection-only and explicitly states that changing it never activates a provider;
- the application does not import/wire the I5.13B Auto engine as a production orchestration entry point.

## Qualification decision

I5.13D therefore closes I5.13 with:

- **offline/shadow Auto policy: QUALIFIED**;
- **production Auto integration: NOT QUALIFIED / DEFERRED**;
- **production Auto authorization: NO**;
- **production default modification: NO**.

This is a successful evidence-only decision, not an I5.13 failure. Enabling Auto without a qualified live-context resolver would require generalizing frozen scenario evidence to unknown live inputs, which the I5.13A policy explicitly forbids.

## Blocking gaps before any future production Auto authorization

A later production-integration effort must separately qualify at least:

1. a deterministic live-context resolver that maps production inputs to admitted evidence contexts and abstains outside the frozen envelope;
2. a production orchestration bridge that preserves the full Auto decision/reason-code/evidence provenance;
3. shadow validation of that live context-resolution + orchestration path before activation;
4. preservation of PyCOLMAP camera/pose authority, abstention-first behavior, historical rejections, no result-driven tuning, no global score/rank, and no hidden fallback/retry-until-pass.

I5.13D does not implement or pre-authorize those mechanisms.

## Roadmap

A passing D artifact marks I5.13 complete and authorizes **I5.14 final I5 qualification and closure**. I5.14 must preserve the bounded conclusion: the first evidence-driven Auto policy is qualified offline/shadow, while production Auto remains deferred/unqualified.
