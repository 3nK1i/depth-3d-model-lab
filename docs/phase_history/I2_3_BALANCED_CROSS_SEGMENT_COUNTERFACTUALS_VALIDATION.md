# I2.3 — Balanced Cross-Segment Counterfactuals

## Purpose

I2.3 is the final narrow diagnostic search after I2.2 showed two partially independent artifact families: early views strongly influence far-peripheral tails, while late views can improve the global full/robust spatial envelope. I2.3 tests only cross-segment pairs so one candidate can attack each family simultaneously.

## Scope

- Reuse the exact passing I1 aligned source clouds.
- Reuse I2/I2.1/I2.2 evidence.
- Derive up to three promising early singles and two promising late singles from I2.2.
- Form at most six cross-segment two-view exclusions.
- Delegate exact voxel refusion and unchanged formal F filtering to the proven I2.2 engine.
- Compare artifact reduction, Core/Extended loss, robust span, full/robust ratio, dominant-component retention, and combined-vs-single synergy.
- No SfM rerun, no DA3 rerun, no source G mutation, no global F threshold change.

## Balanced triage gate

A scenario may be marked `balanced_candidate_for_manual_review` only when all diagnostic conditions hold:

- Core loss <= 7.5%.
- Extended loss <= 10%.
- Peripheral reduction >= 15%.
- Far-peripheral reduction >= 25%.
- Full/robust ratio improves versus baseline.
- Extended robust span does not regress by more than 15%.
- Dominant Extended component remains >= 90% and within 5 percentage points of baseline.

This is a manual-review gate only. It does not authorize production view exclusion.

## Synergy evidence

I2.3 additionally reports:

- `core_overlap_savings_fraction`: positive when valid Core losses overlap rather than add linearly.
- `far_gain_over_best_single_fraction`: incremental far-tail reduction versus the stronger component single.
- `ratio_gain_over_best_single_fraction`: incremental spatial-envelope improvement versus the stronger component single.

## Authority

I2.3 is diagnostic only. Metric scale, measurement, persistence promotion, I2 closure, I3 progression, and mesh reconstruction remain unpromoted until manual review and an explicit correction/closure decision.
