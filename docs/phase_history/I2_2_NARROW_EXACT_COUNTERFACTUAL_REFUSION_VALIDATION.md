# I2.2 — Narrow Exact Counterfactual Refusion

## Scope

I2.2 follows the I2.1 support-sensitivity diagnosis. It does **not** change the source I1/G scene. It reuses the already-saved aligned Dense RGB source views and performs isolated exact counterfactual experiments for only the strongest I2.1 single-view and support-2 pair hypotheses.

Each counterfactual performs:

1. remove one or two suspect workspace orders;
2. rerun the existing `DenseVoxelFusion` at the **exact baseline voxel size**;
3. rerun the unchanged formal F confidence/consistency/artifact filter;
4. recompute Core/Extended spatial quality against the fixed baseline Core anchor;
5. compare artifact reduction against Core/Extended coverage loss.

SfM and DA3 are not rerun. G persistence is not replaced or mutated. Metric scale, measurement authority, certified metrology and mesh authority remain unpromoted.

## Default scenarios

The default scenario list is data-driven from I2.1:

- top 5 single-view dependencies;
- top 5 support-2 co-support pairs.

For the current full-video scene this is expected to target the evidence around orders `5,1,4,30,31` and pairs such as `30-31`, `25-26`, `3-5`, `4-5`, `1-2`, but the runner consumes the I2.1 summary rather than hard-coding those values.

## Decision semantics

I2.2 is diagnostic. The ranking and Pareto front are triage aids only. No view removal is promoted into production geometry automatically.

A scenario is marked `promising_for_manual_exact_review` only when it is relatively surgical (<=5% Core loss, <=10% Extended loss, no >15% robust-span regression) and materially reduces far/peripheral geometry. Even then, manual RGB/region inspection is required before any correction is authorized.

## Authority

- source geometry changed: **false**
- source G bundle changed: **false**
- SfM rerun: **false**
- DA3 rerun: **false**
- exact fusion rerun per scenario: **true**
- formal F rerun per scenario: **true**
- metric scale claimed: **false**
- measurement authoritative: **false**
- surface mesh generated: **false**
