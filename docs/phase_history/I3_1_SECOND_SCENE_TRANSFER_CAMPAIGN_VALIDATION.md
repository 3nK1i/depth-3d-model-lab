# I3.1 — Second Indoor/Static Scene Transfer Campaign

## Purpose

Start the second real-scene Dense RGB campaign without assuming that the I2
manifold correction generalizes automatically. I3.1 first proves that the new
generic scene-relative evidence builder exactly reproduces the closed I2.6
`d12_sparse` production mask. Only then does it run/reuse a different
indoor/static sequence and evaluate the same rule as a counterfactual.

I3.1 does not modify formal F/G for the new scene. Production promotion remains
blocked until second-scene manual review succeeds.

## Locked authority

- SfM gauge remains arbitrary.
- No physical metric scale claim.
- No measurement authority.
- No certified metrology.
- No surface mesh.
- No peripheral/far, extremum-driver, source-view blacklist, or metric-distance
  predicate is allowed in the d12 rejection decision.

Peripheral/extremum diagnostics may only protect the scene-relative Core
reference used to derive manifold/KNN thresholds.

## I2 transfer preflight

Before heavy second-scene work, I3.1 reconstructs the manifold evidence directly
from the original I2 F Extended cloud and requires:

1. exact d12 rejection-mask equivalence with I2.6;
2. exact scene-relative anchor-distance q90 equivalence with I2.5.2; and
3. exact scene-relative KNN-spacing q90 equivalence with I2.5.2.

Any mismatch fails closed.

## Second-scene flow

1. Run/reuse I1 end-to-end on a different static indoor sequence.
2. Run ordinary I2-style baseline quality qualification.
3. Rebuild the scene-relative manifold evidence.
4. Apply locked d12 as a counterfactual only.
5. Compare baseline and candidate topology using exact 26-neighbour lineage.
6. Save baseline-aligned decision diagnostics plus candidate Extended/Core NPZs.
7. Require conservative transfer-safety gates before manual review.

## Conservative transfer gates

- Extended loss <= 1.0%.
- Core loss <= 0.5%.
- Robust-diagonal shrink <= 3.0%.
- Full/robust ratio may not regress materially (>1%).
- Dominant lineage remains exact.
- Newly detached voxels = 0.
- Removed baseline-dominant voxels = 0.
- Largest-component fraction does not degrade.
- Detached voxel count does not increase.

These are safety gates for manual review, not production-promotion criteria.

## Validation commands

```bash
bash scripts/run_i3_1_tests.sh focused
bash scripts/run_i3_1_tests.sh quick
bash scripts/run_i3_1_tests.sh affected
```

Local packaging validation:

- focused: 14 passed
- quick: 241 passed
- affected: requires the project PyQt6 environment; packaging environment lacks PyQt6

## Real run

```bash
python scripts/run_i3_1_second_scene_transfer_campaign.py \
  --video /path/to/SECOND_STATIC_INDOOR_VIDEO.mp4 \
  --output-dir /tmp/eia_i3_second_scene_campaign
```

Expected end state when all automatic gates pass:

`READY_FOR_MANUAL_TRANSFER_REVIEW`

Formal F/G remains unchanged for the second scene until the manual transfer
review is accepted in the next I3 step.
