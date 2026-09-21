# I6.7F — Full admitted dense GPU viewing and separate dense-job progress

## Current boundary

- One Start sparse progress remains its own 0–100% bar.
- Dense execution has a distinct animated indeterminate bar. Its stage and elapsed time are presented separately; no invented total completion percentage is displayed. Completion reaches 100% **only after** the independent E2B1 verifier has admitted the complete artifact. Failure/cancellation never shows completion.
- A **View complete dense points** button is enabled only for the current session's independently admitted dense result. It loads the full binary RGB PLY in a background QThread, independently rechecks source/frame/run/manifest and full-artifact hash, and stages **every** vertex in two contiguous float32 arrays.
- A distinct VisPy marker visual is submitted on the Qt/GL owner thread, with one geometry submission per load. The I6.7C 4,096-point provisional preview remains frozen and separate. The complete view never changes canonical geometry, persisted scene, PyCOLMAP camera authority, or metric scale.
- Fit uses robust bounds **for camera framing only**; it never crops, samples or filters the displayed cloud. An explicit memory ceiling refuses an unrenderable complete artifact rather than passing off a sample as a full result.
- Closing/unloading clears GPU marker buffers. New sparse sessions or persistent canonical scene changes also invalidate the full view. A stale background preparation cannot bind to a different session or dense run.

## Qualification

Run `bash scripts/run_i6_7f_tests.sh focused`, then visually replay TempleRing's **existing** admitted file via `scripts/run_i6_7f_full_dense_viewer_demo.py` (no SfM/MVS rerun). Verify all 163,827 points are visible and orbit/pan/zoom/fit are responsive on the RTX 2060. Viewer-apply time excludes asynchronous GPU drawing and is not an FPS measurement. Record VRAM and frame timing separately before I6.7 as a whole is declared closed.

After focused and manual GPU checks pass, run `bash scripts/run_i6_7f_tests.sh closure`. Do not freeze I6.7F or close I6.7 from dependency-light tests alone. South Building remains a separate, later real-run qualification.
