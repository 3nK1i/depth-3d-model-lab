# I5.1B — Current Frozen-Foundation Difficult-Handheld Replay

## Purpose

I5.1B performs the first fresh runtime robustness replay of the difficult Horse
capture after I4 was formally frozen. It consumes the SHA-bound I5.1A historical
baseline and establishes a fresh SHA256 identity for the replay video before any
runtime work begins.

The replay deliberately reuses the already-qualified reconstruction primitives:

1. current I1 global offline Dense RGB campaign;
2. I4.1.6A decoupled dense-candidate localization into the same PyCOLMAP frame;
3. I4.1.6B real DA3 + dense-consistency A/B.

This is a robustness replay, not new provider development.

## Quality semantics

`PASS`, `DEGRADED`, and `REJECTED` are legitimate replay-quality outcomes. The
I5.1B evidence record itself may be valid even when the capture quality remains
degraded or rejected. A poor Horse result must never be converted to PASS by
relaxing dense-consistency, alignment, or 10-connected-view requirements.

I5.1B does not close I5.1. It authorizes I5.1C to qualify the difficult-handheld
operating envelope from the historical baseline plus this fresh replay.

## Controlled replay bounds

The replay defaults to the historical-like bounded campaign settings:

- exhaustive matching;
- 40 global keyframes;
- 40 global dense views;
- 120 total decoupled candidate bound;
- 2 FPS global/local sampling targets;
- DA3 process resolution 392.

These are **controlled experiment bounds only**. They do not become production
limits and do not change the frozen `permanent_global_view_cap=None` authority
contract. MP-SfM/matching-strategy evaluation remains I5.10.

## Provenance caveat

I5.1A proved that the old standalone Horse runtime summary JSONs are no longer
available. Therefore I5.1B does not claim content-hash equality between the
fresh replay file and the historical Horse file. It records a fresh SHA256 and
keeps the comparison explicitly at scenario-continuity level.

## Authority

Always preserved:

- PyCOLMAP is the sole global camera / scene-frame authority;
- extra localized frames are observations, not a second map;
- SfM gauge remains arbitrary;
- no metric-scale, measurement, certified-metrology, or mesh authority;
- no provider ranking or promotion;
- no Auto policy;
- no Hybrid authorization;
- no production-default modification;
- no dense-consistency threshold relaxation.

## Next boundary

After fresh replay evidence is complete, only **I5.1C — Difficult-Handheld
Robustness Qualification and Closure** is authorized. I5.2 remains blocked until
I5.1C formally closes I5.1.

## Host-memory OOM correction

The first real Horse replay attempt on the 16 GiB RTX 2060 development host
reached I4.1.6B extra alignment/projection candidate 147/176 before Linux killed
the replay process at roughly 10.4 GiB resident host memory. I1 had completed
and I4.1.6A had localized all 176/176 candidates, so the failure was isolated to
the I4.1.6B in-memory record strategy rather than SfM, localization, CUDA, or a
quality-gate decision.

The correction is semantics-preserving: every full-resolution canonical cloud
continues to be persisted exactly as before, but the runner drops bulk
`points`/`colors`/`aligned_depth` arrays after persistence and reloads only the
two records needed for the current pair diagnostic. This changes memory
lifetime, not reconstruction inputs, thresholds, candidate plans, camera
authority, or dense-quality semantics.

The failed replay directory must remain preserved as failed/incomplete evidence;
the corrected replay must use a new timestamped output directory.
