# I4.1.7B — Bounded Provisional Orbit-Target Correction

## Purpose

I4.1.7B is a narrow requalification step after I4.1.7A showed that a clean box orbit was being collapsed by the provisional optical-axis-consensus target while the same authoritative camera trajectory supported a robust trajectory-circle-center orbit.

It does not rerun SfM, DA3, dense A/B, fusion, filtering, or persistence. It does not relax any locked I4.1.7 threshold and does not make the trajectory target production authority.

## Bounded fallback rule

The registered-camera trajectory-circle target may replace the optical-axis target for **I4.1.7 provisional qualification only** when all of the following are true:

- the source I4.1.7 campaign completed DEGRADED;
- I4.1.7A reproduces the saved orbit metric exactly;
- source and recomputed I4.1.7A evidence both classify `provisional_target_collapse_supported`;
- the registered-trajectory-circle target is a robust alternative in I4.1.7A evidence;
- registered and selected pose sets both satisfy the unchanged locked orbit geometry under invariant plane constructions;
- target-independent selected trajectory coverage is broad;
- no unrelated I4.1.7 source gate failed;
- static-capture confirmation and prior decoupled guardrails passed.

If any precondition fails, I4.1.7B stays fail-closed.

## Unchanged locked gates

- SfM registration >= 80%
- dense alignment >= 80%
- selected qualified views >= 10
- azimuth coverage >= 150 degrees
- occupied azimuth sectors >= 5/12
- opposed-view angle >= 120 degrees
- target-facing fraction >= 75% with <= 60 degree facing angle
- standoff p90/p10 <= 2.5
- dense comparison uses the saturation-safe I4.1.7A rule, with no pair-quality regression permitted at a saturated baseline

## Authority

A qualified I4.1.7B result may authorize entering I4.1.8, but:

- production sampling is not bound in I4.1.7B;
- the trajectory-circle target is provisional only;
- SfM scale remains arbitrary;
- metric scale and measurement authority remain false;
- no mesh is authorized;
- I4.1.8 must recompute the target from retained Core geometry and perform final I4.1 requalification.

## Replay

```bash
python scripts/run_i4_1_7b_bounded_target_correction.py \
  --i4-1-7-dir /tmp/eia_i4_1_7_box_clean_static_orbit_03_final \
  --i4-1-7a-dir /tmp/eia_i4_1_7a_box_final_geometry_diag_01 \
  --output-dir /tmp/eia_i4_1_7b_box_bounded_target_01
```
