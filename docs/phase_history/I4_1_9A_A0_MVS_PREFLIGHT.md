# I4.1.9A A0 — MVS Preflight

Scope: read-only qualification of the existing isolated PyCOLMAP runtime for the House MVS positive control.

A0 does **not** run image undistortion, PatchMatch Stereo, Stereo Fusion, SfM, bundle adjustment, camera refinement, or scene persistence.

The probe requires the selected external PyCOLMAP interpreter to expose:

- `undistort_images`
- `PatchMatchOptions`
- `patch_match_stereo`
- `StereoFusionOptions`
- `stereo_fusion`
- CUDA device discovery

It also records free disk, available RAM, and NVIDIA VRAM headroom without importing Torch or PyCOLMAP into the main application process.

The default disk/RAM/VRAM thresholds are configurable **I4.1.9A House experiment launch guardrails**, not production view/resource limits.

Run:

```bash
python scripts/run_i4_1_9a_mvs_preflight.py
```

Expected report:

`/tmp/eia_i4_1_9a_mvs_preflight.json`

A PASS authorizes only the next bounded A1 authority-preservation/68-view bridge step. It does not authorize a heavy MVS job by itself.
