# I4.1.7A — Final Geometry Diagnosis + Saturation-Safe Dense A/B

## Purpose

I4.1.7A is a bounded corrective sub-step after a clean box capture completed the full I4.1.7 pipeline but remained DEGRADED at the final orbit gates.

The real box evidence showed:

- global SfM: 40/40 registered;
- I1 selected dense views: 40;
- both I4.1.6A plans localized at 100% with full-span coverage;
- recommended local-neighbor plan: 118/118 aligned, 118 selected, pair-PASS = 1.0;
- global baseline pair-PASS = 1.0;
- final provisional-target orbit: about 87 degrees / 4 sectors / 86 degrees opposed.

This step does **not** lower any dense or orbit threshold and does not authorize production binding by itself.

## Correction 1 — saturated baseline comparison

The prior I4.1.6B comparison required both:

1. more selected qualified views; and
2. a strictly larger pair-PASS fraction.

That rule becomes mathematically impossible when the global-only baseline is already 1.0. The corrected comparison therefore keeps the strict rule for every unsaturated baseline, but for a saturated 1.0 baseline accepts **no pair-quality regression + strictly more selected qualified views**.

This is not a threshold relaxation. A candidate below the saturated baseline still fails.

## Diagnosis — saved-pose orbit geometry only

`run_i4_1_7a_final_geometry_diagnosis.py` reuses the existing completed I4.1.7 artifacts. It performs no SfM, DA3, dense A/B, fusion, filtering, or persistence rerun.

It compares the same registered/selected camera poses against:

- the saved I4.1.7 provisional optical-axis-consensus target;
- a freshly recomputed optical-axis-consensus target;
- a registered-trajectory circle-fit target;
- retained I1 Core geometry when available;
- target-independent trajectory-centred angular spread.

Each target is evaluated under multiple coordinate-invariant orbit-plane constructions. The diagnostic can support a provisional-target-collapse hypothesis only when alternative target evidence robustly recovers the locked orbit geometry. It never authorizes an orbit-metric change automatically.

## Run

```bash
rm -rf /tmp/eia_i4_1_7a_final_geometry_diagnosis

python scripts/run_i4_1_7a_final_geometry_diagnosis.py \
  --i4-1-7-dir /tmp/eia_i4_1_7_box_clean_static_orbit_03_final \
  --output-dir /tmp/eia_i4_1_7a_final_geometry_diagnosis
```

## Guardrails

- one authoritative PyCOLMAP SfM frame remains unchanged;
- no threshold relaxation;
- no production sampling integration;
- no production target authority;
- SfM scale remains arbitrary;
- no metric-scale claim;
- no authoritative measurement;
- no mesh authorization;
- I4.1.8 remains unauthorized until a later bounded correction/requalification legitimately passes all locked I4.1.7 gates.
