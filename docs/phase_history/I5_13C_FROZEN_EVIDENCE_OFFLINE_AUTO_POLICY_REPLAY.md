# I5.13C — Frozen-Evidence Offline Auto-Policy Replay

## Purpose
I5.13C qualifies the deterministic I5.13B Auto policy engine by replaying only frozen evidence-context tokens against predeclared expected decisions. It does not execute providers, SfM, MVS, scene classification, scenario rematerialization, or production Auto integration.

## Frozen replay matrix
The replay manifest is declared in code before evaluation and contains ten automatic cases: two uniquely-supported selections and eight evidence-driven abstentions. The matrix covers every dense evidence context compiled by I5.13B plus the forward-motion and degraded sparse-SfM abstention boundaries.

## Pass contract
Every case must match the predeclared decision class, selected provider ID, and exact ordered reason-code sequence. Each case is evaluated three times and must be byte-stable under canonical JSON serialization. Any decision drift, reason-code drift, engine-module hash drift, or design-hash drift rejects the replay.

## Scientific boundaries
- PyCOLMAP remains camera/pose authority.
- No global provider score or ranking is created.
- No cross-family or runtime/GPU tie-break is introduced.
- No rejected/degraded evidence is promoted to an automatic primary.
- No hybrid family is selected.
- No matching-policy change is authorized.
- Abstention remains a valid terminal Auto decision.
- Passing I5.13C authorizes only the I5.13D production-integration qualification/decision review; it does not authorize production Auto by itself.
