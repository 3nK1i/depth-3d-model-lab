# Phase 7.5 Thin-Structure Preservation Fix Validation

## Scope

Focused pre-commit correction for object-specific point extraction. The change preserves mask-supported thin or articulated parts such as chair legs, rails, handles and narrow supports without disabling background rejection.

## Changes

- Replaced the maximum-neighbour discontinuity rule with eight-neighbour depth consensus.
- Added a relative depth threshold so smooth metric depth ramps survive.
- Added source-image support-retention diagnostics after 3D filtering stages.
- Added an outlier-filter fallback when strict MAD trimming collapses valid image support.
- Added support-aware merging of image-adjacent, depth-consistent 3D clusters.
- Kept depth-distant secondary clusters rejected as probable background contamination.
- Added processing-step diagnostics and explicit warnings when preservation guards intervene.
- Added a standalone visual smoke test.

## Validation completed

- Python compilation: passed.
- New thin-structure unit tests: 4 passed.
- Object geometry filter/clustering/pipeline focused suite: 23 passed.
- Phase 7.5 Steps 1-10 quick regressions: 245 passed.
- Object extraction integration/stress focus: 6 passed.
- Synthetic visual smoke: passed.

Synthetic smoke result:

```text
status=success raw=8450 filtered=8306 vertical_min=19 vertical_max=164 contamination=0
```

The synthetic chair-like object retained both lower supports while the injected far-depth contamination patch contributed zero output points.

## Environment limitation

The complete unit/UI suite was not run in the delivery container because PyQt6 and VisPy are unavailable. Run the full split-process Step 10 suite in the configured `depth` environment.

## Real-scene validation required

Re-run the chair image in Measure & Inspect and compare Raw and Filtered points. Confirm that lower legs/seat support survive and that obvious wall/floor contamination is not introduced. This is a geometry-completeness correction, not a calibration claim.
