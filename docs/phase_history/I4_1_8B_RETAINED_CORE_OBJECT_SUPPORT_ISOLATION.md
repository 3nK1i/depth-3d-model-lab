# I4.1.8B — Retained-Core Object Support Isolation

## Purpose

I4.1.8A showed that the bound production camera trajectory is broad, but the
trajectory-circle center is not yet supported as the reconstructed object's
final target. I4.1.8B therefore isolates object-support evidence **inside the
retained production Core** before any final target binding.

This step is intentionally diagnostic/isolation-only. It does not change the
production target, production sampling, locked I4.1 gates, metric scale,
measurement authority, or mesh state.

## Evidence used

I4.1.8B consumes only already-saved I4.1.8/I4.1.8A artifacts:

- retained Core points, confidence and saved support counts;
- the 40 bound production camera poses/source indices;
- the 40 saved bound source clouds;
- completed I4.1.8 stage/authority evidence;
- completed I4.1.8A diagnosis evidence.

The runner reconstructs the exact per-Core-voxel view-observation matrix from
the saved source clouds using the same saved-depth replay used by F/E2. The
reconstructed support count must match `view_observation_counts` for every Core
voxel exactly or the step fails closed.

## Isolation policy

Object-support selection does **not** consult azimuth coverage, occupied bins,
opposed-view angle, target-facing fraction, standoff ratio, or any other I4.1
pass/fail metric.

Per Core voxel, the step derives:

- exact supporting-view count;
- per-view optical-axis attention rank;
- central-observation fraction;
- support rank span;
- F filter confidence.

High-evidence Core seeds are spatially clustered in Euclidean reconstruction
space. Candidate components are ranked only by retained-Core evidence:
attention, provenance breadth, compactness, 3-D structure and component size.
The selected component must also be unique enough, bounded relative to the full
Core, broadly attributed across production views/source span, and produce a
stable geometric-median target under deterministic bootstrap resampling.

Only after the component is selected does the step evaluate its target against
the locked orbit policy, and that result is recorded as **post-hoc diagnosis
only**. It cannot change isolation readiness.

## Authority

Always false in I4.1.8B:

- `production_target_changed`
- `production_sampling_integration_authorized`
- `orbit_thresholds_changed`
- `dense_consistency_threshold_relaxation_authorized`
- `metric_scale_claimed`
- `measurement_authoritative`
- `certified_metrology`
- `surface_mesh_generated`

No PyCOLMAP, DA3, dense A/B, fusion, F filtering, G persistence, I2 quality, or
production-binding stage is rerun.

## Outcomes

`retained_core_object_support_isolated`
: Independent retained-Core object-support evidence is strong enough to design
  I4.1.8C final target binding. This still does not promote the target.

Other classifications remain fail-closed, including insufficient attention
seeds, no compact component, ambiguous support, inadequate view/source breadth,
non-compact scene-scale support, or unstable target evidence.

## Next step

When `object_support_isolation_ready=True`, I4.1.8C may bind the final target
from the independently isolated retained-Core support, rerun the unchanged final
I4.1 gates, perform final manual review, and either close I4.1 or fail closed.
