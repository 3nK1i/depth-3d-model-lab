# I2.5.3 — Conservative Manifold-Attachment Filtering Counterfactual

Status: counterfactual validation step. No production F/G policy or authority is changed.

## Purpose

I2.5.2 showed a strong local-manifold signal: extremum drivers and stubborn survivors are strongly enriched for large distance to strong multi-view geometry and local sparsity. I2.5.3 tests whether that signal can remove the envelope-driving tails without repeating the robust-denominator damage seen in I2.5.

## Locked constraints

The candidate filter rule may use only scene-relative local-manifold evidence derived from the central-Core I2.5.2 reference. It must not use peripheral/far labels, I2.5.1 extremum labels, source-view/camera-order blacklists, metric distance, or whole-view deletion. Those labels are evaluation-only.

The fixed narrow ladder is:

- `d4_sparse`
- `d6_sparse`
- `d8_sparse`
- `d12_sparse`
- `d4_sparse_linear`

All distance/spacing/linearity thresholds are multiples of the I2.5.2 central-Core q90 references.

## Acceptance evidence

A candidate is only flagged for manual review when it preserves Core/central-Core geometry, removes a large fraction of extremum and stubborn survivors, materially reduces far-tail geometry, improves the full/robust envelope ratio, avoids excessive robust-diagonal shrink, and preserves the dominant component.

This step does not authorize production filtering, I2 closure, I3, metric scale, measurement authority, or mesh reconstruction. Promotion belongs to I2.6 only after a numerically and visually qualified candidate exists.

## Commands

```bash
bash scripts/run_i2_5_3_tests.sh focused
bash scripts/run_i2_5_3_tests.sh quick
bash scripts/run_i2_5_3_tests.sh affected
```

Real campaign:

```bash
python scripts/run_i2_5_3_conservative_manifold_filter_counterfactual.py \
  --i1-dir /tmp/eia_i1_offline_dense_campaign_recovered \
  --i2-dir /tmp/eia_i2_full_video_quality \
  --i2-5-1-dir /tmp/eia_i2_5_1_spatial_extremum_survivor_diagnosis \
  --i2-5-2-dir /tmp/eia_i2_5_2_extremum_local_geometry_manifold_diagnosis \
  --output-dir /tmp/eia_i2_5_3_conservative_manifold_filter_counterfactual
```
