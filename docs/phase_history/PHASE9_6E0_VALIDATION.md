# Phase 9.6E0 — Estimated Metric Ruler Readout & Provenance

## Scope

Phase 9.6E0 adds a truthful pre-calibration metric readout for the canonical two-point ruler without changing canonical scale authority.

The existing ruler already reports metres ± uncertainty when the loaded scene carries provisional or qualified metric-scale evidence. Arbitrary-scale scenes such as the real South Building DA3 scene intentionally have `unit_to_metre=null`; DA3 cannot supply a physical scale by itself. E0 therefore does **not** reinterpret reconstruction units as metres.

Instead, E0 adds one optional, scene-bound `EstimatedMetricScaleHint`. The current UI can create it from the last canonical ruler segment plus an approximate real-world distance and an explicit user-selected ± uncertainty. The same dependency-light API can later consume an automatic metric-depth-derived scale estimate without changing ruler authority or UI semantics.

## UX

A collapsed **RULER METRIC ESTIMATE** group exposes:

- estimate state;
- estimated metres per reconstruction unit ± relative scale uncertainty;
- exact ruler reference used to create the estimate;
- explicit authority wording;
- **Set from last ruler** and **Clear estimate** actions.

With an estimate active, the viewport ruler shows the estimated metre distance as the primary result with explicit ± uncertainty while retaining the native reconstruction-unit distance in the secondary/provenance text.

## Provenance / refinement

The first metric estimate for a ruler segment is retained in `ReconstructionRulerMetricRecord`. If a better scene-bound provisional/qualified scale later becomes available, the current value is recomputed from that evidence while the original estimated value, uncertainty, source and provenance remain available in the record/tooltip.

The estimate is session-side only in E0. Phase 9.6F owns persistence/restore and must preserve this provenance without mutating historical scene evidence.

## Authority rules

- canonical geometry remains the only picking/measurement geometry authority;
- `EstimatedMetricScaleHint` is explicitly non-authoritative and uncalibrated;
- the hint never writes or changes canonical `unit_to_metre`;
- camera intrinsics/calibration alone still do not establish scene scale;
- DA3, Gaussian and NeRF do not establish metric scale;
- clearing the hint returns an arbitrary scene to native reconstruction-unit-only measurement;
- rejected/unavailable measurements never expose a trusted metric number;
- `certified_metrology=false` throughout E0.

## Automated gates

```bash
./scripts/run_phase9_6e0_tests.sh quick
./scripts/run_phase9_6e0_tests.sh integration
./scripts/run_phase9_6e0_tests.sh backward
```

## Manual validation

On an arbitrary-scale canonical scene:

1. use the ruler to pick Point A and Point B; confirm the initial result remains native reconstruction units / metric unavailable;
2. expand **RULER METRIC ESTIMATE** and choose **Set from last ruler**;
3. enter an approximate physical distance and a conservative uncertainty percentage;
4. confirm the active ruler immediately changes to `Estimated ... m ± ...` while the raw reconstruction-unit distance remains visible in the secondary/provenance text;
5. measure a second canonical segment; confirm the same scene-bound estimate is applied automatically;
6. clear the estimate and confirm metric values disappear rather than being fabricated;
7. if a scene with real provisional/qualified metric evidence is available, confirm scene scale takes precedence and the original E0 estimate is retained in the ruler provenance tooltip.

No backend, DA3 rerun, Gaussian/NeRF optimization, calibration mutation, persistence migration or release tag belongs to E0.
