# I4.1.8C — Final Object-Target Binding + Orbit Causality + I4.1 Closure Decision

## Purpose

I4.1.8C consumes the independently isolated retained-Core object support from I4.1.8B and performs the final object-centric orbit causality check before I4.1 closure.

It compares the **same retained-Core-supported target** against:

1. the complete qualified decoupled dense path from I4.1.7B; and
2. the actual bounded production path from I4.1.8/I4.1.8B.

This distinguishes capture/reconstructed-trajectory insufficiency from production-binding loss without changing any locked I4.1 threshold.

## Non-goals / authority guardrails

I4.1.8C does **not** rerun PyCOLMAP, DA3, dense pair tests, fusion, F filtering, G persistence, I2 quality, or production binding. It does not search for a camera subset or target that passes. It does not change orbit thresholds, claim metric scale, authorize measurements, or generate a mesh.

The final target is independently reproduced as the geometric median of the exact isolated retained-Core points saved by I4.1.8B. The saved target must reproduce to <= 1e-6 voxel drift or the step fails closed.

## Causality outcomes

### `final_object_orbit_qualified`
The actual production-bound path passes every unchanged object-orbit gate against the retained-Core-supported target. Automated I4.1 qualification passes and final manual orbit review is required. I4.1 remains open until that review is explicitly confirmed.

### `production_binding_lost_object_angular_coverage`
The complete qualified path passes but the production-bound path fails. The capture/reconstruction evidence is sufficient, but production binding lost object-centric coverage. Only production view-binding policy may be corrected; threshold or target changes remain forbidden.

### `qualified_path_object_orbit_insufficient`
The complete qualified path also fails against the independently isolated object target. This establishes that the current reconstructed capture does not satisfy the locked object-centric orbit gate. No code/threshold workaround is authorized; acquire one better-controlled orbit capture and rerun qualification.

## Explicit manual closure

If and only if automated qualification passes, rerun I4.1.8C with `--confirm-manual-orbit-review` after visually reviewing the final object/orbit evidence. This changes the I4.1.8C result to `i4_1_closed` and authorizes entry to I4.2. The flag cannot override an automated failure.

## Locked gates

No changes:

- registered fraction >= 0.80
- dense alignment fraction >= 0.80
- selected views >= 10
- azimuth coverage >= 150 deg
- occupied sectors >= 5/12
- opposed-view diversity >= 120 deg
- target-facing fraction >= 0.75 using the existing 60 deg facing limit
- standoff p90/p10 <= 2.5

## Validation

Assistant isolated environment:

- focused: 41 passed
- quick: 174 passed
- integration: 37 passed
- stress: 21 passed
- affected: collection blocked only because PyQt6 is not installed in the isolated assistant runtime; authoritative result must come from the project `depth` environment.

A synthetic end-to-end runner replay also verified the expected fail-closed `qualified_path_object_orbit_insufficient` path without any heavy-stage reruns.
