# Phase 6 Step 1 Validation

## Scope

Registry hardening only. No new depth model is integrated in this step.

## Automated validation

```bash
./scripts/run_phase6_step1_tests.sh quick
./scripts/run_phase6_step1_tests.sh full
```

The full mode reruns all Phase 5 unit, UI, integration, and stress tests together with the new Phase 6 tests.

## Manual validation

1. Start the application and confirm MoGe-2 remains the default model.
2. Confirm the detected RTX 2060 or RTX 5090 profile is unchanged.
3. Load MoGe-2, run an image, video, USB camera, and RTSP source, then stop and unload it.
4. Switch to Development Mock and confirm the GPU Profile control is disabled.
5. Switch back to MoGe-2 and confirm the GPU Profile control is enabled.
6. Verify model badges, load/warm-up stages, depth, point cloud, navigation, and performance telemetry.
7. Repeat model switching and source switching, then close the application and confirm clean shutdown.

## Pass condition

Do not commit until compile validation, the full automated suite, manual source validation, and lifecycle stress validation all pass on the project Python 3.11 environment.
