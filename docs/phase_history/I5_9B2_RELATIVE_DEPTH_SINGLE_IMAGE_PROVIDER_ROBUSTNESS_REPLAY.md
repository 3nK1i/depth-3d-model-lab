# I5.9B2 — Relative-Depth Single-Image Provider Robustness Replay

## Scope

I5.9B2 replays the two remaining single-image relative-depth production providers across the six frozen I5.9A Castle/PyCOLMAP foundations:

- Depth Anything V2
- Marigold

The campaign contains 12 provider/scenario attempts and executes each admitted attempt at most once.

## Frozen predecessor state

I5.9B2 requires:

- the exact I5.9A admission and I5.6A frozen SfM admission;
- the closed I5.9B1 replay with the observed 24 PASS / 0 DEGRADED / 0 REJECTED result;
- a passing I5.9B2 replay validation receipt.

The I5.9B1 replay is bound by caller-supplied SHA256 and its full record matrix is validated before B2 preflight or inference.

## Provider-specific alignment semantics

Depth Anything V2 keeps its existing per-view affine inverse-depth alignment to authoritative SfM. Marigold keeps its existing per-view affine native prediction-domain alignment. These domains are not interchangeable and are guarded separately by tests.

Neither provider's native depth nor estimated intrinsics becomes canonical authority. PyCOLMAP remains the sole camera/global-frame authority.

## Historical Low Texture anchor

I4.3G4 recorded genuine Low Texture production rejections for both Depth Anything V2 and Marigold. I5.9B2 preserves those historical verdicts as separate frozen evidence. New Castle results may differ, but they do not overwrite or reinterpret the historical G4 outcome.

## Status semantics

Existing production PASS, DEGRADED, and recognized quality REJECTED terminal semantics are preserved. Fusion-capacity drops are recorded as evidence only and do not automatically change a single-image provider status.

A runtime or integrity failure is not converted to a scientific rejection. It produces failure evidence and stops the batch nonzero. No retry-until-pass, threshold relaxation, provider substitution, capacity increase, or result-driven tuning is permitted.

## Preflight

`preflight` is execution-free. It binds predecessor evidence, probes the two real production runtimes, validates source/checkpoint provenance, revalidates all six frozen SfM reports and staged inputs, and resolves all 12 existing I4.3F adaptive execution plans without starting provider inference.

## Production-native replay

`production-native` reruns the same strict preflight inside the production evidence directory, then executes the 12 attempts sequentially. Every attempt revalidates the frozen scenario after execution and records runtime status, accepted views, alignment metrics, resource evidence, fusion-capacity observations, artifacts, and authority assertions.

## Prohibited conclusions

I5.9B2 does not rank providers, select a universal winner, establish physical metric scale, authorize metrology, start MP-SfM, or implement Auto/Hybrid policy.

On evidence-complete terminal completion, I5.9C may be authorized for evidence-only remaining-provider qualification and closure.
