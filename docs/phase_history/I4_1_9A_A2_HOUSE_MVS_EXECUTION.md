# I4.1.9A A2 — House MVS Positive-Control Execution

## Scope

A2 changes only the dense backend in the controlled House experiment. It consumes the A1 PASS workspace and the exact 68-view authoritative PyCOLMAP camera bridge.

Allowed operations:

- COLMAP/PyCOLMAP image undistortion
- PatchMatch Stereo
- Stereo Fusion to a diagnostic fused PLY

Forbidden operations:

- new SfM / mapping
- bundle adjustment or pose refinement
- House `K.txt` injection
- metric-scale promotion
- measurement/metrology authority
- mesh generation

## First-run policy

All 68 registered House views remain reference candidates. The initial RTX 2060 configuration uses 10 automatically selected overlap source views per reference via PyCOLMAP's `num_patch_match_src_images=10`. This is a local PatchMatch working-set bound, not a global reconstruction view cap.

The initial run keeps native 1020×764 input resolution (`max_image_size=-1`), geometric consistency enabled, default PatchMatch quality iterations/samples, a 2 GiB PatchMatch cache, and a 2 GiB Stereo Fusion cache. These are configurable experiment resource controls, not production limits.

Each A2 attempt is immutable: the output directory must be absent/empty. A failed/OOM attempt is preserved as evidence and any changed configuration must use a new attempt directory.

## Qualification

A2 PASS requires:

- A1 PASS and all 68 image checksums intact
- qualified PyCOLMAP 4.1.1 CUDA12 runtime
- launch disk/RAM/VRAM guardrails satisfied
- undistorted dense workspace still contains all 68 cameras with the same poses and sparse 3D positions
- PatchMatch config contains all 68 reference views and respects the configured per-reference source-neighbor bound
- geometric depth and normal maps for all 68 references
- Stereo Fusion produces a non-empty PLY
- A1 sparse input hashes unchanged after execution
- no metric, measurement, metrology, mesh, or camera-authority promotion

A2 PASS does not by itself qualify visual reconstruction quality. The fused MVS PLY proceeds to the controlled DA3-vs-MVS visual A/B verdict.
