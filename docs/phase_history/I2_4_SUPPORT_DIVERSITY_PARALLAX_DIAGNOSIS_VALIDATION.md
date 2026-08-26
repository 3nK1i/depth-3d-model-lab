# I2.4 — Support-Diversity / Parallax Artifact Diagnosis

## Purpose

I2.4 investigates the remaining full-video peripheral/tail geometry after I2.1–I2.3 showed that deleting whole views or small view pairs removes too much valid Core geometry.

The hypothesis under test is narrower: retained voxels can satisfy a simple multi-view support count while their supporting cameras still have very little geometric diversity. Two nearby views can therefore agree on the same incorrect depth sheet and make it look locally consistent.

I2.4 measures that hypothesis directly from the exact retained-voxel membership and the already-saved SfM camera poses.

## Diagnostic metrics

For every retained Extended voxel I2.4 computes:

- exact supporting view count;
- maximum pairwise viewing-ray parallax angle;
- maximum camera-baseline / mean-range ratio (scale invariant);
- maximum absolute supporting-camera baseline in reconstruction units;
- selected-view rank span;
- workspace-order span.

The maximum pair evidence is deliberately conservative: if any supporting pair gives meaningful parallax/baseline, the voxel is not treated as narrow-view solely because another pair is nearly co-located.

## Reference thresholding

The diagnostic low-diversity threshold is not an absolute world-scale threshold. It is derived from the lower quartile of central Core geometry with comparable low support (default support <=3). If too few such points exist, all central Core points are used as a fallback.

A voxel is marked `low_support_diversity` only when:

1. its support is <=3 views; and
2. its maximum pair parallax is at or below the central-Core reference threshold; and
3. its maximum baseline/range ratio is at or below the corresponding reference threshold.

This classification is diagnostic only.

## Support-2 pair analysis

I2.4 also groups exact support-2 voxels by the two supporting workspace orders and reports for each pair:

- Extended/Core/peripheral/far counts;
- low-diversity peripheral/far counts;
- camera baseline;
- parallax quantiles;
- baseline/range quantiles;
- temporal/order gaps.

This directly tests whether I2.1's strong local pairs (for example neighboring trajectory views) are geometrically weak as well as statistically influential.

## Authority boundaries

I2.4 does **not**:

- change geometry;
- change F filtering thresholds;
- rerun SfM;
- rerun DA3;
- rerun fusion;
- rewrite G persistence;
- promote metric or measurement authority;
- generate or authorize a surface mesh.

A strong I2.4 signal only justifies a later targeted diversity-aware filtering counterfactual. It is not itself a production filtering rule.

## Validation commands

```bash
bash scripts/run_i2_4_tests.sh focused
bash scripts/run_i2_4_tests.sh quick
bash scripts/run_i2_4_tests.sh affected
```

Expected project counts on the current H/I stack:

- focused: 14 passed
- quick: 115 passed
- affected: 177 passed

Real diagnosis:

```bash
rm -rf /tmp/eia_i2_4_support_diversity_parallax_diagnosis

python scripts/run_i2_4_support_diversity_parallax_diagnosis.py \
  --i1-dir /tmp/eia_i1_offline_dense_campaign_recovered \
  --i2-dir /tmp/eia_i2_full_video_quality \
  --i2-1-dir /tmp/eia_i2_1_targeted_peripheral_diagnosis \
  --i2-3-dir /tmp/eia_i2_3_balanced_cross_segment_counterfactuals \
  --output-dir /tmp/eia_i2_4_support_diversity_parallax_diagnosis
```

Recommended manual diagnostics after inspecting the terminal summary:

```bash
python scripts/run_i2_4_support_diversity_preview.py \
  --i2-4-dir /tmp/eia_i2_4_support_diversity_parallax_diagnosis \
  --representation extended \
  --color-mode diversity \
  --view oblique \
  --fit full \
  --show

python scripts/run_i2_4_support_diversity_preview.py \
  --i2-4-dir /tmp/eia_i2_4_support_diversity_parallax_diagnosis \
  --representation extended \
  --color-mode parallax \
  --view top \
  --fit full \
  --show
```

## Decision

I2 remains open. Mesh reconstruction remains unauthorized until the full-video quality issue is resolved and I2 is formally closed.
