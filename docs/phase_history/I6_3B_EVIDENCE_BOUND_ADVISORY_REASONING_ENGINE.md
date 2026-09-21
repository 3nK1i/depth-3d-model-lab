# I6.3B — Evidence-Bound Advisory Reasoning Engine

## Scope

I6.3B implements the deterministic, execution-free reasoning layer for the Reconstruction Advisor introduced by I6.3A.

The engine consumes only `ReconstructionAdvisorRequest` and returns `ReconstructionAdvisorDecision`. It does not alter the reconstruction controller, UI, workers, providers, settings, persistence, SfM, dense execution, or production Auto.

## Frozen decision principle

The Advisor may recommend one concrete provider only when the exact frozen evidence context contains exactly one clean `PASS` provider and that provider is currently available.

Otherwise the Advisor abstains explicitly.

Availability and hardware resources may block a recommendation, but they never resolve evidence ambiguity. Multiple exact clean PASS candidates therefore remain an abstention even if only one is currently available. GPU/RAM observations are descriptive only and are not a quality score or tie-break.

## Fail-closed conditions

The Advisor abstains when any of the following holds:

- source is not ready;
- authoritative PyCOLMAP SfM is not usable;
- runtime integrity is not clean;
- no concrete provider is available;
- evidence context is unresolved;
- no exact frozen-context evidence exists;
- exact evidence contains no clean PASS provider;
- more than one exact PASS provider exists;
- the unique exact PASS provider is unavailable.

Negative, degraded, rejected, and unqualified evidence is preserved in provider exclusions and never promoted.

## Boundaries preserved

- PyCOLMAP SfM remains sole canonical camera/global-frame authority.
- No live scene/context classifier is implemented or qualified.
- No provider selection or activation side effect exists.
- No reconstruction Start action exists.
- No settings mutation exists.
- No fallback/retry-until-pass loop exists.
- No global/cross-family ranking exists.
- No scalar provider score exists.
- Hardware is not used as a tie-break.
- Production Auto remains unqualified and unauthorized.
- Certified metrology is not claimed.

## Validation strategy

I6.3B validation includes focused unit/integration/stress tests, I6.3A contract regression, I5 evidence-bound Auto/closure regressions, I6 predecessor regressions, compile/import probes, and the full unit suite in the authoritative project environment.

No manual UI validation is required because I6.3B has no UI or controller wiring. Those belong to I6.3C and I6.3D.
