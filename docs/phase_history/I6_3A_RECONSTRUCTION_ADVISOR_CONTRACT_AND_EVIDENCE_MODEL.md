# I6.3A — Reconstruction Advisor Contract & Evidence Model

## Purpose

I6.3A opens the **Guided Reconstruction Advisor** as an execution-free product
contract.  It defines what later I6.3 steps may observe and what an Advisor may
return, without wiring any provider choice into production execution.

The Advisor is deliberately separate from production Auto:

- Advisor: explains a defensible evidence-bound choice, or abstains;
- Auto: would select/activate a provider in production and remains unqualified
  and unauthorized.

I6.3A modifies no controller, worker, provider adapter, runtime action, UI,
persistence format, scientific threshold, or production default.

## Evidence boundary inherited from I5

I5 closed with a deterministic offline/shadow Auto policy qualified against
frozen evidence, while production Auto remained deferred because live context
resolution and production orchestration were not qualified.

I6.3A preserves those boundaries exactly:

- exact scenario/context evidence remains required for a provider recommendation;
- PASS, DEGRADED and REJECTED outcomes remain distinct and preserved;
- a later PASS in a different scenario does not erase historical negative evidence;
- unresolved, bounded or conflicting evidence can be represented explicitly;
- no provider win-count, global score, cross-family rank, runtime-performance
  tiebreak or GPU-performance ranking is introduced;
- PyCOLMAP SfM remains the sole canonical camera/global-frame authority;
- production Auto remains `offline_shadow_only`, unqualified and unauthorized.

## New contract

`core/reconstruction/reconstruction_advisor_contract.py` defines immutable
contracts for four concerns.

### 1. Runtime/source context

`ReconstructionAdvisorRequest` carries only descriptive context:

- offline reconstruction mode;
- selected source workflow, readiness, item count and source fingerprint;
- workspace state;
- authoritative SfM usability;
- runtime-integrity state;
- observed GPU/VRAM/RAM resources;
- concrete currently available provider IDs;
- frozen provider-evidence observations.

Resource observations are descriptive facts only and explicitly serialize
`is_quality_score = false`.

### 2. Context resolution

`AdvisorContextResolution` has two I6.3A states:

- `unresolved` — no frozen evidence-context token is claimed;
- `explicit_frozen_token` — a caller supplies an explicit frozen context token.

I6.3A does **not** claim that a production scene/context resolver exists or is
qualified.  An unresolved context therefore carries no evidence-context ID.

### 3. Provider evidence/provenance

`AdvisorEvidenceObservation` binds one concrete provider implementation to:

- its canonical provider family;
- PASS / DEGRADED / REJECTED / UNQUALIFIED outcome;
- one exact evidence-context ID;
- matched scenario IDs;
- reason codes;
- bounded limitations;
- one or more SHA-256 evidence bindings.

Provider family/implementation consistency is checked against the canonical
I4.3 provider architecture.

### 4. Advisory decision

`ReconstructionAdvisorDecision` permits only:

- `recommend` — one concrete provider, only with `exact_qualified` support,
  matched scenarios and content-addressed provenance;
- `abstain` — no hidden provider choice.

The decision carries qualitative evidence support, reason codes, explanations,
limitations, considered providers and reason-coded exclusions.  It explicitly
forbids a numeric confidence/quality score.

## Non-effect guarantees

Both request and decision contracts make the following boundaries explicit:

- provider activation authorized: **NO**;
- provider-selection effect: **NO**;
- provider execution: **NO**;
- reconstruction execution: **NO**;
- settings mutation: **NO**;
- Start action: **NO**;
- production Auto qualified: **NO**;
- production Auto authorized: **NO**;
- global provider ranking: **NO**;
- global scalar score: **NO**;
- cross-family rank: **NO**;
- universal provider winner: **NO**.

## TDD / validation

The focused tests were written first and initially failed at collection because
the new contract module did not exist.  The implementation then satisfied the
contract without adding runtime dependencies.

I6.3A validation includes:

- focused unit/integration contract tests;
- I5.13A-D Auto-policy evidence-boundary regressions;
- I5.14A/B final-I5 closure regressions;
- I6.1/I6.2 pure predecessor regressions;
- import/compile probes;
- full unit regression in the authoritative `depth` environment before closure.

Manual UI validation is not required because I6.3A changes no UI.

## Deferred to I6.3B

I6.3B may implement the deterministic **advisory reasoning engine** on top of
this contract.  It must remain non-executing and must use the contract's
abstention path when exact, unique, provenance-backed evidence is not available.
It must not wire production Auto or silently create a live context classifier.
