# I4.1.7 — Clean Static Object / Connector Orbit Qualification

## Purpose

I4.1.7 validates the decoupled global/local dense-candidate architecture on a deliberately clean static object-orbit capture. It follows the difficult Horse I4.1.6B result, which proved that decoupled sampling materially improves real dense consistency but did not reach the 10-view production gate.

This step is **qualification only**. It does not bind a production planner, relax dense-consistency thresholds, change fusion/filtering/persistence policy, claim metric scale, authorize measurement, or generate a mesh.

## Pipeline

1. Validate prior I4.1.6B guardrails.
2. Generate a uniformly sampled capture contact sheet for manual review.
3. Run a fresh global I1 baseline with full-span sampling and exhaustive matching.
4. Run I4.1.6A candidate localization into the same authoritative SfM frame.
5. Run I4.1.6B real DA3 + dense-consistency A/B.
6. Use the chosen plan's real selected path for orbit qualification.
7. Compute a **provisional optical-axis-consensus target** from the authoritative global SfM cameras.
8. Apply the locked I4.1 orbit gates.

The provisional target is intentionally not production authority. If I4.1.7 passes, I4.1.8 must bind the validated plan, run bounded fusion/F/I2, and recompute final object-orbit qualification from retained Core geometry.

## Locked gates

- registered global SfM fraction >= 80%
- candidate plan covers the full source span
- dense alignment fraction >= 80%
- selected qualified views >= 10
- real dense plan increases selected qualified views and improves pair-PASS fraction over global-only baseline; if the baseline is already saturated at pair-PASS = 1.0, the candidate must preserve that saturated pair quality with no regression while still increasing selected qualified views
- selected azimuth coverage >= 150 degrees
- selected occupied azimuth sectors >= 5/12
- selected opposite-side view angle >= 120 degrees
- selected target-facing fraction >= 75% using <= 60 degree facing angle
- selected standoff p90/p10 <= 2.5
- explicit manual confirmation that the object/environment were static enough for the intended static-world model

No threshold relaxation is allowed.

## Capture guidance

Preferred first qualification object: matte, rigid, textured, asymmetric object such as a multicolored shoe or textured tool body. Record roughly 35–45 s at 1080p/30 FPS, normal 1x lens, landscape, smooth camera motion around the fixed object, approximately 210–240 degrees of orbit, stable distance/exposure, no cuts, minimal blur, and no moving people/vehicles/background elements.

## Quick capture preflight

```bash
rm -rf /tmp/depth3d_i4_1_7_capture_preflight

python scripts/run_i4_1_7_capture_preflight.py \
  --video /path/to/clean-static-orbit.mp4 \
  --output-dir /tmp/depth3d_i4_1_7_capture_preflight
```

Review the generated contact sheet before the heavy campaign.

## Real campaign

```bash
rm -rf /tmp/depth3d_i4_1_7_clean_static_orbit

python scripts/run_i4_1_7_clean_static_orbit_campaign.py \
  --video /path/to/clean-static-orbit.mp4 \
  --confirm-static-capture \
  --output-dir /tmp/depth3d_i4_1_7_clean_static_orbit
```

Default runtime assumptions preserve the established isolated environments:

- PyCOLMAP: isolated PyCOLMAP runtime
- DA3: isolated Depth Anything 3 runtime
- DA3 checkout: external Depth Anything 3 source checkout
- exhaustive SfM matching
- 40 global anchors
- local dense target 2 FPS
- dense candidate cap 120
- final production fusion is **not executed** in I4.1.7

## Decision

A clean PASS emits:

```text
I4.1.7 Clean Static Orbit Qualification: QUALIFIED
i4_1_8_production_binding_authorized=True
production_sampling_integration_authorized=False
```

A failure remains fail-closed and authorizes neither production binding nor I4.2. Diagnose the failed capture/geometry/dense gate rather than relaxing thresholds.
