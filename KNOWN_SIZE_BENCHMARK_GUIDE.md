# Phase 7.5 Known-Size Benchmark Guide

This benchmark validates the **complete Measure & Inspect pipeline** against physical dimensions recorded independently with a tape measure or caliper.

The result is an engineering validation. It is not a certified metrology record, and the application must continue to display:

> Estimated Uncertainty - Not Calibrated

## Minimum benchmark set

Use at least three rigid cuboids with visibly different sizes:

| Reference object | Recommended size range | Instrument |
|---|---:|---|
| Small calibration box | 100-250 mm per side | Caliper or rigid rule |
| Medium cardboard/plastic box | 250-600 mm per side | Tape measure or rule |
| Large cabinet/crate | 600-1200 mm per side | Tape measure |

Record width, height, and depth before capture. Use the same physical edge convention throughout the benchmark.

## Capture protocol

For every object:

1. Keep the complete object visible with margin around all sides.
2. Avoid touching an image border.
3. Use matte, rigid objects where possible.
4. Avoid transparent, reflective, deformable, or articulated objects for the release gate.
5. Use even lighting and avoid strong motion blur.
6. Capture front-oblique views so three faces are visible.
7. Test at more than one distance when practical.
8. Do not use digital zoom.
9. Record the camera/source, resolution, approximate distance, lighting, and notes.
10. Use the same selected box policy that will be reported in the export.

For the RTX 2060, begin with:

```text
Depth model: Depth Pro or MoGe-2
Perception: Ultralytics YOLO Instance Segmentation
Profile: Efficient
Maximum objects: 3
```

## Produce the inspection exports

For each capture:

1. Run Measure & Inspect.
2. Select the known-size object.
3. Inspect the segmentation mask.
4. Correct the mask only when necessary.
5. Choose the most defensible box method.
6. Lock the approved result.
7. Export the selected object.
8. Preserve the generated export directory unchanged.

Do not edit `measurements.json` or the export manifest manually.

## Create the benchmark manifest

```bash
python scripts/create_known_size_benchmark_template.py \
  --output outputs/phase7_5/known_size_benchmark.json
```

Edit the generated JSON:

- Add one case per exported capture.
- Use the exact export directory.
- Use `observation_id` from the exported `measurements.json`.
- Enter independently measured dimensions in millimetres.
- Check `measurement_mapping` carefully. This handles a deliberate width/depth swap when the selected local object frame uses a different edge convention.
- Review and approve the acceptance thresholds.
- Set `thresholds_confirmed` to `true` only after the acceptance criteria are agreed.

The template thresholds are **provisional** and intentionally block release until confirmed.

## Run the validator

```bash
python scripts/run_known_size_validation.py \
  --manifest outputs/phase7_5/known_size_benchmark.json \
  --output-dir outputs/phase7_5/known_size_validation
```

Generated files:

```text
known_size_validation.json
known_size_measurements.csv
known_size_validation.md
known_size_validation.html
manifest.json
```

Open the HTML report:

```bash
xdg-open outputs/phase7_5/known_size_validation/known_size_validation.html
```

## Metrics

The validator reports:

- Signed error per dimension
- Absolute error in millimetres
- Relative error in percent
- Mean absolute error
- RMSE
- Maximum absolute error
- Expanded-uncertainty coverage
- Availability/rejection status
- Reliability
- Mask source and revision
- Preferred 3D box provenance
- Export integrity using SHA-256

A dimension is covered by uncertainty when:

```text
absolute error <= reported expanded uncertainty
```

## Interpretation

A green numerical result is not sufficient by itself. Review:

- Whether the mask follows the intended physical object.
- Whether the chosen box represents the intended dimensions.
- Whether width, height, and depth mappings are consistent.
- Whether low-reliability measurements are accepted too often.
- Whether rejected values remain rejected.
- Whether uncertainty covers the observed error at an acceptable rate.

Do not relax thresholds merely to make the report pass. Threshold changes must be documented with a technical reason.

## Final release gate

After the benchmark passes:

```bash
./scripts/run_phase7_5_release.sh full \
  --known-size-report outputs/phase7_5/known_size_validation \
  --require-clean-git \
  --skip-static-dependencies
```

Remove `--skip-static-dependencies` on the complete production environment when all optional model packages are installed.

The release gate also rejects model checkpoints or generated runtime files that are accidentally tracked in Git.
