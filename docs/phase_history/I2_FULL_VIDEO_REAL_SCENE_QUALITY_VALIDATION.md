# I2 — Full-Video Real-Scene Visual & Spatial Quality Qualification

I2 consumes a completed I1 full-video campaign. It does not rerun PyCOLMAP,
DA3, fusion, F filtering, or G persistence, and it never modifies retained
geometry. The purpose is to determine whether the first complete-video Dense
RGB reconstruction is numerically healthy enough for structured manual visual
review before the campaign proceeds to additional scenes or mesh work.

## Automated evidence

The I2 runner validates:

- I1 pipeline completion and explicit I2 authorization;
- F/G/fusion pass state and exact retained point counts;
- SfM registration and dense-alignment coverage;
- selected PASS-only path coverage;
- exact F observation accounting;
- Core and Extended retention fractions;
- largest supported component fraction;
- G self-contained persistence and authority boundaries.

It then computes diagnostic-only spatial evidence from the persisted F
Extended/Core geometry:

- robust Core anchor envelope;
- Extended robust and full scene span;
- retained peripheral and far-peripheral masks;
- weak-peripheral evidence based on support/confidence/dispersion relative to
  Core;
- dominant-component retention;
- directional peripheral sectors in the arbitrary reconstruction frame;
- per-view attribution of retained peripheral/far voxels using the exact saved
  selected-view voxelization used by F.

No I2 diagnostic mask deletes or changes a point.

## Status semantics

`ready_for_manual_visual_review` means all automated gates passed. It does not
mean that the scene has been visually accepted. I2 deliberately keeps:

- `authorizes_i3_second_scene_campaign = false`;
- `authorizes_mesh_reconstruction = false`;
- metric scale, measurement authority and mesh authority unpromoted.

The final I2 closure is evidence-based after fixed-view RGB and diagnostic
previews are inspected.

## Validation commands

```bash
bash scripts/run_i2_tests.sh focused
bash scripts/run_i2_tests.sh quick
bash scripts/run_i2_tests.sh affected
```

Real current-video analysis:

```bash
rm -rf /tmp/eia_i2_full_video_quality
python scripts/run_i2_full_video_quality_qualification.py \
  --i1-dir /tmp/eia_i1_offline_dense_campaign_recovered \
  --output-dir /tmp/eia_i2_full_video_quality
```

Useful fixed-view previews:

```bash
python scripts/run_i2_quality_preview.py --i2-dir /tmp/eia_i2_full_video_quality \
  --representation extended --color-mode rgb --view oblique --fit full --show

python scripts/run_i2_quality_preview.py --i2-dir /tmp/eia_i2_full_video_quality \
  --representation extended --color-mode region --view oblique --fit full --show

python scripts/run_i2_quality_preview.py --i2-dir /tmp/eia_i2_full_video_quality \
  --representation extended --color-mode region --view side --fit full --show

python scripts/run_i2_quality_preview.py --i2-dir /tmp/eia_i2_full_video_quality \
  --representation extended --color-mode region --view top --fit full --show
```

`region` colors are diagnostic only: central retained geometry is green,
peripheral retained geometry yellow, weak-peripheral orange, and far-peripheral
red. Use `support`, `confidence`, `dispersion`, `component`, and `direction` for
follow-up diagnosis. The reconstruction remains in the arbitrary SfM gauge.
