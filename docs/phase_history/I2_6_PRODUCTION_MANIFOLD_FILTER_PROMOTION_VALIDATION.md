# I2.6 — Production Manifold Filtering Promotion & Canonical Scene Regeneration

## Purpose
Promote the manually reviewed I2.5.3 `d12_sparse` manifold-attachment rule into formal F filtering and regenerate the self-contained G Dense RGB scene bundle.

## Locked production rule
The promoted rule rejects a previously retained F Extended voxel only when both are true:

- nearest strong-manifold distance > `12 ×` the I2.5.2 central-Core non-driver q90 reference;
- local KNN median spacing > `1 ×` the corresponding central-Core q90 reference.

The production rule does **not** accept peripheral/far labels, extremum-driver labels, source-view blacklists, or metric distances as inputs.

## Safety / authority
- requires explicit manual-review confirmation;
- reproduces the reviewed I2.5.3 rejection mask exactly, otherwise fails closed;
- no PyCOLMAP rerun;
- no DA3 rerun;
- no fusion rerun;
- formal F rejection provenance records `weak_manifold_attachment`;
- G persistence is regenerated and relocation-tested;
- SfM scale remains arbitrary;
- metric, measurement, certified-metrology and mesh authority remain unpromoted.

## Validation gates
```bash
bash scripts/run_i2_6_tests.sh focused
bash scripts/run_i2_6_tests.sh quick
bash scripts/run_i2_6_tests.sh affected
```

Real promotion:
```bash
python scripts/run_i2_6_manifold_filter_promotion.py \
  --i1-dir /tmp/eia_i1_offline_dense_campaign_recovered \
  --i2-5-2-dir /tmp/eia_i2_5_2_extremum_local_geometry_manifold_diagnosis \
  --i2-5-3-dir /tmp/eia_i2_5_3_conservative_manifold_filter_counterfactual \
  --output-dir /tmp/eia_i2_6_manifold_filter_promotion \
  --scenario d12_sparse \
  --confirm-manual-review-pass
```

A passing I2.6 authorizes only I2.7 final full-video requalification. It does not close I2 or authorize mesh reconstruction.
