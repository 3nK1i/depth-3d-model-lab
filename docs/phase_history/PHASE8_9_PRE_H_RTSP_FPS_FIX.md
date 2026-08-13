# Phase 8.9 pre-H correction — RTSP FPS propagation + shutdown hardening

## Trigger

A real Android IP Webcam RTSP stream opened successfully as H.264 1920x1080 at 30 FPS, but Reconstruction -> Live source initially rejected stella_vslam startup because no explicit resolved frame rate reached the live-SLAM request.

After correcting FPS propagation, the same real stream reached stella_vslam tracking and produced live poses, keyframes, trajectory, and sparse landmarks. Closing the application during RTSP capture then exposed a second production defect: Qt aborted with `QThread: Destroyed while thread is still running`.

## Root causes

1. `RtspSource` opened the FFmpeg/OpenCV decoder without retaining `CAP_PROP_FPS`. The live SLAM bridge therefore created RTSP `InputSourceMetadata` with `expected_frame_rate_hz=None`, while the stella configuration correctly requires explicit source timing.
2. RTSP `VideoCapture.open/read` can block longer than the generic two-second `BaseSource.stop()` wait. The generic stop then discarded its QThread reference even when the acquisition method had not returned yet; subsequent QObject/QThread destruction could abort the application.

## Correction

- Resolve and retain decoder-reported RTSP FPS only when finite and strictly between 0 and 1000 Hz.
- Clear cached FPS before every open/reopen so stale timing cannot survive a failed reconnect.
- Expose the value as read-only `RtspSource.native_fps`.
- Pass `source.native_fps` through the existing RTSP metadata factory into `expected_frame_rate_hz`.
- Add explicit FFmpeg/OpenCV open and read timeout properties to bound native RTSP calls.
- Override only the RTSP stop path with a wait budget covering both native timeouts plus margin, while avoiding any self-wait if a status callback requests stop from the capture thread itself.
- If a backend unexpectedly ignores those timeouts, retain ownership and wait for the QThread to finish rather than destroying a running thread. Never use `QThread.terminate()`.
- Keep `CAP_PROP_BUFFERSIZE` post-open because this workstation's FFmpeg backend rejects it in the constructor parameter list.
- Do not hardcode 30 FPS and do not weaken stella_vslam frame-rate validation.
- Do not alter credentials handling, SLAM authority, scale semantics, calibration policy, or external-backend isolation.

## Validation

From the clean post-G3 baseline `77bf7f3fc05e80e5f632f1438fc044602663ed7e`, with this complete six-file pre-H overlay applied:

```bash
./scripts/run_phase8_9_pre_h_rtsp_fps_fix_tests.sh quick
./scripts/run_phase8_9_pre_h_rtsp_fps_fix_tests.sh affected
```

Then repeat the real phone RTSP application test. Acceptance requires:

- the old `explicit resolved frame rate` failure remains absent;
- the stream reaches real stella_vslam tracking;
- normal Stop/Disconnect/application close does not emit `QThread: Destroyed while thread is still running` and does not abort/core-dump;
- no `eia_stella_live_worker` remains after normal shutdown;
- Git scope remains exactly the six correction files;
- no release/tag action occurs until final regression requalification is green.
