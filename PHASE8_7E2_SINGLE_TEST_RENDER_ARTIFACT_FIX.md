# Phase 8.7E2 — Single-test Render Artifact Fix

## Scope

Compatibility correction for Nerfstudio v1.1.5 dataset rendering during the bounded real-backend qualification.

## Root cause

E2 deliberately exports one test camera. Nerfstudio v1.1.5 derives each dataset-render output path relative to `os.path.commonpath(image_filenames)`. With exactly one test image, that common path can be the image file itself, so the valid RGB artifact is written as `rendered/test/rgb.png` rather than under `rendered/test/rgb/<image>.png`.

The EIA worker only searched the nested layout and therefore rejected a valid completed render.

## Correction

- Accept the v1.1.5 flat single-image layout: `test/rgb.png`.
- Preserve support for the normal nested layout: `test/rgb/**/*.png`.
- Require the selected PNG to exist and be non-empty.
- If no valid RGB artifact exists, report the actual produced-file inventory for diagnosis.
- No Nerfstudio, CUDA, profile, geometry, metrology, or production-environment contract is weakened.

## Validation

Focused unit coverage verifies both valid Nerfstudio output layouts. The real RTX 2060 qualification remains the authoritative E2 gate.
