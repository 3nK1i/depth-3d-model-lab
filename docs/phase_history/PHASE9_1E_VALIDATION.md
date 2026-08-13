# Phase 9.1E Validation — Metric Qualification Evidence Foundation

## Scope

Phase 9.1E adds an evidence/qualification layer above the existing Phase 8
`ScaleReferenceProfile` authority. It does not replace scale-reference,
known-size validation, measurement uncertainty, or canonical-geometry contracts.

Changed production scope:

- `core/reconstruction/metric_qualification.py`
- additive public exports in `core/reconstruction/__init__.py`

## Authority rules

- Known-size and known-distance references may establish **provisional metric
  coordinates** only, with explicit reference and measurement uncertainty.
- A validated reference target may establish **calibrated metric scale**.
- A validated metric-depth/external metric prior may establish calibrated metric
  scale only when an explicit reconstruction-unit-to-metre mapping, uncertainty,
  provenance, and validation record are supplied.
- Rejected evidence emits no `ScaleReferenceProfile`.
- The existing Phase 7.5 known-size benchmark is reused as **supporting-only
  measurement-system evidence** and cannot establish scene scale by itself.
- Stereo and RGB-D evidence classes are reserved for future qualified providers
  and remain unsupported/fail-closed in Phase 9.1E.
- `certified_metrology` remains `false`.

## Compatibility

The existing Phase 8 functions and schema remain unchanged:

- `known_dimension_scale_reference()`
- `reference_target_scale_reference()`
- `external_metric_prior_scale_reference()`
- `ScaleReferenceProfile`
- `core.validation.known_size`

The new Phase 9 path requires explicit uncertainty even though the legacy
Phase 8 convenience functions retain their backward-compatible defaults.

## Automated validation

Local qualification on the reconstructed v0.8.0 + committed 9.1A–9.1D tree:

```text
./scripts/run_phase9_1e_tests.sh quick
59 passed

./scripts/run_phase9_1e_tests.sh integration
7 passed

python scripts/run_phase9_1e_metric_qualification_smoke.py
PASS

Phase 9.1A–9.1D + persistent-scene backward-compatibility sanity
44 passed
```

The smoke verifies:

- known-size evidence -> provisional metric scale;
- validated reference target -> qualified scale;
- validated metric-depth evidence -> qualified scale with explicit uncertainty;
- known-size benchmark -> supporting-only, no scene-scale authority;
- future RGB-D -> unsupported, no authority;
- calibrated camera + validated target -> calibrated metric geometry;
- certified metrology remains false.

The user's production `depth` Conda environment remains the authoritative
pre-commit validation environment.
