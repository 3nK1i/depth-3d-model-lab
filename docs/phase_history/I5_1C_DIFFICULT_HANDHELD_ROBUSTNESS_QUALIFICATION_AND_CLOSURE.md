# I5.1C — Difficult-Handheld Robustness Qualification + Closure

## Purpose

I5.1C is the evidence-only closure boundary for **I5.1 — Difficult handheld captures**. It consumes the frozen I5.1A historical baseline, the successful corrected I5.1B Horse replay, the preserved pre-correction OOM diagnostic, and a narrow closure regression receipt.

It does **not** rerun PyCOLMAP, DA3, localization, dense-pair qualification, fusion, filtering, persistence, or manual review.

## Qualified result

The fresh replay establishes the current difficult-handheld operating envelope:

- fresh PyCOLMAP global reconstruction completed before dense degradation;
- global dense qualification selected 2 connected views;
- decoupled localization completed across the full candidate set;
- `uniform_localized_dense_pool` reached 7 connected qualified views;
- `global_anchors_plus_local_neighbors` reached 6 connected qualified views;
- the locked production gate remains >=10 connected qualified views;
- therefore the Horse scenario is formally **DEGRADED** and outside the current production dense-robustness envelope.

The +5 selected-view improvement over the two-view historical/global baseline is material robustness evidence, but it is not sufficient to claim production qualification.

## OOM event integrity

The first I5.1B replay attempt is preserved as failed/incomplete supporting resilience evidence. It completed I4.1.6A and reached I4.1.6B candidate 147/176 before Linux killed the Python process for host-memory exhaustion.

The subsequent correction changed only bulk dense-record memory lifetime: canonical clouds remain persisted, while full `points` / `colors` / `aligned_depth` arrays are not retained for every candidate simultaneously. Pair checks hydrate only the records they need.

The corrected replay is **not** a retry-until-PASS maneuver. The infrastructure defect was corrected once, the failed attempt was preserved, and the first valid post-correction quality result was accepted unchanged as `DEGRADED`.

This does not close I5.11 OOM/crash/recovery work early; the event remains useful later resilience context.

## Authority and guardrails

I5.1C preserves all frozen authority contracts:

- PyCOLMAP remains the sole global camera / scene-frame authority;
- reconstruction scale remains arbitrary;
- `permanent_global_view_cap=None` remains the production contract;
- no dense-consistency threshold relaxation;
- no production-default modification;
- no provider selection, promotion, ranking, winner, or scalar score;
- no Auto policy;
- no Hybrid authorization;
- no metric-scale, authoritative-measurement, certified-metrology, or mesh authority.

The experiment-level `recommended_plan` is scenario evidence only and is not a production default or provider ranking.

## Closure decision

A passing I5.1C evidence record means:

```text
i5_1a_complete = true
i5_1b_complete = true
i5_1c_complete = true
i5_1_complete = true
i5_1_quality_status = degraded
i5_2_authorized = true
i5_2_started = false
```

## Next

**I5.2 — Low overlap / weak parallax**
