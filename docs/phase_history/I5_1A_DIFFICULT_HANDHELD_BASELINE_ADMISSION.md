# I5.1A — Difficult Handheld Historical Baseline Admission

## Purpose

I5.1A starts **I5.1 — Difficult handheld captures** by admitting the preserved Horse robustness history into the I5 evidence chain without rerunning reconstruction.

The repository already records why Horse belongs here rather than in clean I4 qualification:

- I4.1.4 records that full-span sampling reached 36/40 DA3/SfM alignment while dense consistency still collapsed to two views;
- I4.1.5 records a mixed failure: the Horse capture regime is difficult and local temporal density materially affects dense-consistency success;
- I4.1.6B records the real `uniform_localized_dense_pool` and `global_anchors_plus_local_neighbors` plans and preserves the locked 10-view production gate;
- I4.1.7 records that decoupled sampling materially improved real dense consistency on Horse but still did not reach that 10-view gate;
- Horse is therefore preserved for I5 robustness rather than rewritten as a clean I4 PASS.

I5.1A **does not turn the historical Horse failure into a PASS**. The failure is the baseline.

## Provenance recovery

The standalone historical Horse `i4_1_4_summary.json` / `i4_1_6b_summary.json` files are not required by I5.1A. When they are not retained, I5.1A must not synthesize replacements or invent hashes.

Instead, the formal baseline SHA-binds the committed I4 phase-history records that preserve the historical conclusions:

1. `I4_1_4_DECOUPLED_GLOBAL_COVERAGE_LOCAL_DENSITY_SAMPLING_DIAGNOSIS.md`;
2. `I4_1_5_DECOUPLED_GLOBAL_LOCAL_SAMPLING_COUNTERFACTUAL.md`;
3. `I4_1_6B_REAL_DECOUPLED_DENSE_AB_VALIDATION.md`;
4. `I4_1_7_CLEAN_STATIC_OBJECT_ORBIT_QUALIFICATION.md`.

This is intentionally a narrower provenance claim than rebinding the original runtime JSON. I5.1A records that no historical standalone summary JSON, raw Horse video, or nested native artifact tree was rehashed.

## Scope

I5.1A verifies and SHA-binds:

1. the final passing I4.4D closure;
2. the four committed Horse-relevant I4 phase-history records above;
3. the I5.1A regression receipt.

It does not rerun PyCOLMAP, DA3, dense-pair qualification, fusion, filtering, persistence, or the source video.

## Authority

Locked throughout:

- PyCOLMAP remains the sole authoritative global camera / scene frame owner.
- Historical fixed caps remain historical evidence only; `permanent_global_view_cap=None` remains the production contract.
- No metric scale, authoritative measurement, certified metrology, or mesh authority is created.
- No provider is selected, promoted, ranked, or declared a winner.
- I5.12 Hybrid remains conditional and unstarted.
- The first evidence-driven Auto policy remains deferred to I5.13.

## Decision

A passing I5.1A result means only:

```text
historical_horse_baseline_admitted = true
i5_1_started = true
i5_1a_complete = true
i5_1_current_replay_authorized = true
i5_1_complete = false
i5_2_authorized = false
```

## Next

**I5.1B — Current frozen-foundation difficult-handheld replay**

That step must establish fresh source identity and run the current frozen I4 foundation on a difficult handheld capture. PASS / DEGRADED / REJECTED are all legitimate robustness evidence. A quality failure must not be hidden by threshold relaxation or repeated retries until success.
