# I2.7.1 — Detached Component Qualification

## Purpose

Investigate the single failed I2.7 automated closure gate (`largest_component_preserved`) without changing the promoted `d12_sparse` filter, F/G geometry, persistence, or authority state.

## Method

I2.7.1 recomputes 26-neighbour voxel connected components for both:

- the original I2 baseline Extended cloud; and
- the promoted I2.6/I2.7 Extended cloud.

It then traces exact voxel-key lineage through the 412-point production deletion and separates promoted non-dominant geometry into:

- surviving pre-existing detached geometry;
- newly detached geometry that belonged to the baseline dominant component;
- newly detached Core geometry;
- d12-removed voxels.

The diagnostic also measures support/confidence/dispersion, peripheral/far/weak membership, nearest distance to the promoted dominant component, component-size distribution, and whether newly detached components touch removed voxels.

## Fail-closed prerequisites

The runner requires:

1. an automated-passing original I2 baseline;
2. a degraded I2.7 summary;
3. `largest_component_preserved` to be the **only** failed I2.7 automated closure gate;
4. promoted geometry to be an exact voxel-key subset of baseline geometry;
5. all surviving coordinates, RGB, support, confidence, dispersion, and Core membership to match baseline exactly.

## Authority

Diagnostic only:

- no geometry changes;
- no filter-policy changes;
- no persistence changes;
- no SfM/DA3/fusion rerun;
- arbitrary SfM scale remains arbitrary;
- no metric, measurement, or mesh authority promotion;
- no automatic I2 closure-gate relaxation.

## Validation

```bash
bash scripts/run_i2_7_1_tests.sh focused
bash scripts/run_i2_7_1_tests.sh quick
bash scripts/run_i2_7_1_tests.sh affected
```

Real diagnosis:

```bash
python scripts/run_i2_7_1_detached_component_qualification.py \
  --i2-baseline-dir /tmp/eia_i2_full_video_quality \
  --i2-7-dir /tmp/eia_i2_7_final_requalification \
  --output-dir /tmp/eia_i2_7_1_detached_component_qualification
```

Do not alter the I2.7 98% component gate based on assumptions. Review the exact baseline-vs-promoted component evidence first.
