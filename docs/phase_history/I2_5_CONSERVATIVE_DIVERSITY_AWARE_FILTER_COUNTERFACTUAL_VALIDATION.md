# I2.5 — Conservative Diversity-Aware Filtering Counterfactual

## Scope

I2.5 evaluates whether the strong I2.4 support-diversity/parallax signal can be
used safely as a *post-F diagnostic counterfactual* without modifying the
production F policy.

The counterfactual rules consume only:

- retained multi-view support count;
- maximum supporting-view parallax;
- maximum baseline/range ratio;
- existing F high-support-neighbor evidence;
- existing F local radius-2 connectivity;
- existing F confidence where explicitly requested by the policy.

I2 peripheral/far labels are **evaluation only**. They are never part of the
filter rule.

No SfM, DA3, fusion, persistence, metric scale, measurement, UI, or mesh
authority is changed.

## Default policy ladder

1. `s2_lowdiv_anchor1` — support 2, I2.4 low diversity, at most one nearby >=3-view anchor.
2. `s2_lowdiv_anchor2` — support 2, I2.4 low diversity, at most two nearby >=3-view anchors.
3. `s2_lowdiv_anchor2_radius12` — policy 2 plus local radius-2 count <= 12.
4. `s2_075_anchor2` — support 2, 0.75x I2.4 diversity thresholds, at most two high-support anchors.
5. `s2_verylow_anyanchor` — support 2 and only 0.50x I2.4 diversity thresholds.
6. `s23_075_anchor1_lowconf25` — support <=3, 0.75x thresholds, at most one high-support anchor, confidence <= central-Core reference P25.

This is intentionally a small policy ladder, not a fine threshold sweep.

## Automated gates

```bash
bash scripts/run_i2_5_tests.sh focused
bash scripts/run_i2_5_tests.sh quick
bash scripts/run_i2_5_tests.sh affected
```

Expected on the production `depth` environment:

- focused: 14 passed
- quick: 129 passed
- affected: 191 passed

The packaging environment does not include PyQt6, so the affected Qt gate is
authoritative only on the production `depth` environment.

## Real counterfactual run

```bash
rm -rf /tmp/eia_i2_5_conservative_diversity_filter_counterfactual

python scripts/run_i2_5_conservative_diversity_filter_counterfactual.py \
  --i1-dir /tmp/eia_i1_offline_dense_campaign_recovered \
  --i2-dir /tmp/eia_i2_full_video_quality \
  --i2-4-dir /tmp/eia_i2_4_support_diversity_parallax_diagnosis \
  --output-dir /tmp/eia_i2_5_conservative_diversity_filter_counterfactual
```

The run must report:

- `source_geometry_changed=False`
- `production_F_policy_changed=False`
- `sfm_rerun=False`
- `da3_rerun=False`
- `fusion_rerun=False`
- `i2_closure_authorized=False`
- `production_filter_authorized=False`
- `mesh_authorization=False`

## Manual-review triage

A counterfactual is marked `balanced_for_manual_review` only when it preserves
Core/central-Core geometry and the robust spatial extent while also reducing
peripheral/far geometry and improving the full/robust scene ratio.

`preferred_for_manual_review` is deliberately stricter. Neither flag authorizes
production F changes.

Only the top 1–3 policies should be previewed after reviewing the terminal
ranking. Example:

```bash
python scripts/run_i2_5_policy_preview.py \
  --i2-5-dir /tmp/eia_i2_5_conservative_diversity_filter_counterfactual \
  --scenario <BEST_SCENARIO> \
  --representation extended \
  --color-mode rgb \
  --view oblique \
  --show
```

For an explicit decision map:

```bash
python scripts/run_i2_5_policy_preview.py \
  --i2-5-dir /tmp/eia_i2_5_conservative_diversity_filter_counterfactual \
  --scenario <BEST_SCENARIO> \
  --representation extended \
  --color-mode decision \
  --view top \
  --include-rejected \
  --show
```

Decision colors: kept Core = green, rejected Extended-only = red, rejected Core
= magenta.

## Authority

I2.5 is counterfactual only. Promotion of a successful diversity-aware policy
into formal F belongs to I2.6 and requires an evidence-backed manual review.
