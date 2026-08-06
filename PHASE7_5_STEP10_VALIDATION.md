# Phase 7.5 Step 10 Validation

Baseline commit: `d3959f90f002f8371a52257d5433ce738e8dc690`

## Scope delivered

- Known-size benchmark manifest and capture protocol
- Deterministic validation of Step 9 inspection exports
- Per-dimension signed, absolute, and relative errors
- MAE, RMSE, maximum error, and uncertainty coverage
- Export-manifest SHA-256 verification
- JSON, CSV, Markdown, and HTML validation reports
- Explicit provisional-threshold guard
- Phase 7.5 release-candidate gate
- Tracked checkpoint/generated-artifact release check
- Bounding-box volume wording in human-facing reports
- Local checkpoint and validation-output patterns added to `.gitignore`
- Safe untracking script for accidentally committed runtime artifacts

## Validation completed in the delivery environment

- Input archive checksum: passed
- Python compilation: passed
- Step 10 quick gate: 11 passed
- Step 10 integration gate: 1 passed
- Known-size validation smoke: passed
  - 1 synthetic case
  - 3 dimensions
  - MAE 1.333 mm
  - uncertainty coverage 100%
- Phase 7.5 dependency-light release-gate probe: passed
- Step 1-9 focused quick regressions: 230 passed
- Focused Phase 7.5 total: 242 passed
- Fresh-baseline overlay: passed
- Overlay Step 8 quick regression: 21 passed
- Overlay Step 9 quick regression: 3 passed
- ZIP checksum and integrity: passed

The complete `tests/unit` and UI groups cannot be claimed in this delivery container because it uses Python 3.13 and does not contain PyQt6 or VisPy. The user's configured Python 3.11 `depth` environment is the authoritative target for the complete split-process regression.

## Baseline hygiene finding

The committed Step 9 archive contained:

- `yolo11n-seg.pt` in the repository root
- Three generated Snapshot packages under `outputs/snapshots/`

Step 10 adds ignore rules and `scripts/untrack_generated_phase7_5_artifacts.sh`. The script uses `git rm --cached`, so local copies remain on disk while the files are removed from Git.

## Release honesty

This package intentionally does **not** change the application version or create a release tag. Real known-size captures and independently measured dimensions are required before claiming a stable Phase 7.5 metric release.

After the measured benchmark and target-hardware release gate pass, release metadata can be finalized as a small reviewed patch and pushed with an annotated tag.
