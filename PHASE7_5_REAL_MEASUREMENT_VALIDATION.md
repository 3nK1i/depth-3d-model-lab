# Phase 7.5 Real End-to-End Measurement Validation

This validation-only runner connects the existing Step 2–6 components without
modifying the Qt UI, production runtime, workers, or export system.

It deliberately loads perception first, unloads it, and then loads the depth
model. This sequential policy is safer on the RTX 2060 6 GB.

## Run

```bash
python scripts/run_real_measurement_validation.py \
  --image ~/Downloads/depth-test.png \
  --depth-model depth-pro \
  --perception-model ultralytics-yolo-seg \
  --depth-profile efficient \
  --perception-profile efficient \
  --output-dir /tmp/phase7_5_real_measurement
```

Alternative metric depth models registered by the application include
`moge-2`, `unidepth-v2`, and `metric3d-v2`. Their dependencies/checkpoints must
already be installed.

## Outputs

- `overview.png`
- `rgb_detections_masks.png`
- `depth_preview.png`
- `object_XXX_geometry_measurements.png`
- `measurements.json`

All results remain **Estimated Uncertainty — Not Calibrated**. This is a
validation artifact only; it is not the Step 9 production export writer.
