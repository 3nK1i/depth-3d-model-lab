# I2.7.2 — Baseline-Aware Component Gate Correction + Final Requalification

## Purpose

Correct the single invalid I2.7 closure assumption discovered by I2.7.1 without weakening the promoted `d12_sparse` geometry or lowering the 98% absolute threshold to an arbitrary value.

I2.7.1 proved, with identical fresh 26-neighbour voxel connectivity, that the original qualified I2 baseline had a dominant-component fraction of about 97.05%, while the promoted scene improved it to about 97.35%. The dominant component remained exactly 128,490 voxels; all 412 production removals came from pre-existing detached geometry; no promoted voxel became newly detached.

## Gate correction

The historic 98% value remains recorded as a diagnostic threshold. When an explicitly reviewed I2.7.1 qualification is supplied, I2.7 now uses strict baseline-relative topology preservation instead:

- I2.7.1 status must be `DIAGNOSIS_READY`;
- scene ID and fingerprint must match the promoted scene;
- failure classification must be `preexisting_baseline_fragmentation`;
- surviving XYZ/RGB/support/confidence/dispersion/Core evidence must be exactly equivalent;
- promoted dominant component size must not be smaller than baseline;
- promoted dominant lineage must be exactly 100% from one baseline component;
- newly detached voxels must be zero;
- newly detached components must be zero;
- removed baseline-dominant voxels must be zero;
- promoted dominant fraction must not regress relative to baseline;
- promoted detached count must not increase;
- the I2.7.1 lineage visualization must have an explicit manual PASS confirmation.

No numeric component threshold is lowered.

## Authority

Validation-only correction:

- no geometry change;
- no `d12_sparse` change;
- no F/G regeneration;
- no SfM, DA3, or fusion rerun;
- arbitrary SfM scale remains arbitrary;
- no metric, measurement, certified-metrology, or mesh promotion.

## Validation

```bash
bash scripts/run_i2_7_2_tests.sh focused
bash scripts/run_i2_7_2_tests.sh quick
bash scripts/run_i2_7_2_tests.sh affected
```

Final automated requalification:

```bash
rm -rf /tmp/eia_i2_7_final_requalification

python scripts/run_i2_7_final_requalification.py \
  --i1-dir /tmp/eia_i1_offline_dense_campaign_recovered \
  --i2-baseline-dir /tmp/eia_i2_full_video_quality \
  --i2-6-dir /tmp/eia_i2_6_manifold_filter_promotion \
  --i2-7-1-dir /tmp/eia_i2_7_1_detached_component_qualification \
  --output-dir /tmp/eia_i2_7_final_requalification \
  --confirm-i2-7-1-lineage-review-pass
```

A successful run must report `component_gate=baseline_aware_lineage`, `topology_pass=True`, `automated_gates_pass=True`, and `READY_FOR_FINAL_MANUAL_REVIEW`. It still does not close I2 by itself; fixed-view review and Reconstruction UI sanity remain required before the existing final closure command.
