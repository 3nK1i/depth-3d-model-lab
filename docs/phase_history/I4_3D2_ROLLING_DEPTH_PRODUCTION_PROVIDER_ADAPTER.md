# I4.3D2 — RollingDepth Production Temporal Provider Adapter

## Scope

I4.3D2 adds RollingDepth as the second canonical `temporal_depth` implementation behind the I4.3A provider contract. RollingDepth remains an isolated external runtime and a **candidate** provider. This step does not implement Auto selection, provider promotion, a universal provider ranking, metric scale, metrology, or mesh authority.

## Runtime isolation

RollingDepth has its own official source checkout and Conda environment (`eia-rollingdepth` by default). `scripts/setup_rolling_depth_backend.sh` requires an exact 40-character source revision and records a runtime lock. The setup also verifies the RollingDepth-modified `diffusers` source marker. Model weights are downloaded only after explicit `--accept-model-license` acknowledgement.

Production defaults for the RTX 2060 development machine follow the official low-memory guidance: 512 processing resolution, VAE batch size 1, snippet unloading enabled, fp16, fast-style dilations `[1, 25]`, and no refinement. These are provider runtime defaults, not a permanent frame/view cap.

## Canonical frame and temporal alignment

RollingDepth's native output is temporally co-aligned **relative depth evidence**. The provider preserves the model's whole-sequence co-alignment. It explicitly forbids independent per-frame reconstruction-scale fitting.

One robust affine mapping is estimated from sparse PyCOLMAP observations pooled across the complete registered sequence:

`z_sfm ~= a * d_rolling + b`

The same `a` and `b` are applied to every RollingDepth frame. This mapping only expresses the temporal depth sequence in the existing arbitrary SfM reconstruction gauge. It is not physical metric-scale evidence.

Canonical projection and fusion use only bundle-adjusted PyCOLMAP poses. RollingDepth never supplies or replaces canonical camera poses.

## View policy

I4.3D2 consumes all registered authoritative SfM views in source temporal order. `permanent_global_view_cap=None`. Adaptive selection remains I4.3F.

## Authority locks

- camera pose authority: `pycolmap_sfm`
- RollingDepth pose authority promoted: `False`
- per-frame depth/SfM alignment: `False`
- sequence affine establishes physical metric scale: `False`
- metric scale claimed: `False`
- measurement authoritative: `False`
- certified metrology: `False`
- surface mesh generated: `False`
- provider promotion performed: `False`

## Qualification boundary

I4.3D2 validates the production binding and execution contract only. RollingDepth vs Video Depth Anything temporal quality, flicker, drift, dynamic-scene behavior, long-sequence behavior, and operating-envelope comparison remain I4.3G/I5. I4.3D3 closes the temporal-provider architecture after both bindings are available.

## Setup dependency-name correction

The pinned upstream requirements use `pyav>=13.1.0`, while the installable PyPI distribution for PyAV is named `av`. The setup helper therefore normalizes only the temporary requirements file used for installation (`pyav` -> `av`) and leaves the pinned RollingDepth checkout byte-for-byte untouched. All isolated-runtime pip/install/probe commands run with `PYTHONNOUSERSITE=1` so packages from the developer account's user site cannot satisfy or shadow the dedicated `eia-rollingdepth` environment.

## Runtime executable resolution and resumable setup correction

The isolated environment Python is resolved through `sys.executable` executed inside the named Conda environment rather than through `conda run ... which python`. This avoids empty `which` output observed on the development host while still proving that Conda can execute the environment interpreter. The resulting absolute executable must exist and be executable before any runtime lock is written.

The setup is also resumable after a successful large checkpoint extraction. If `checkpoint/rollingdepth-v1-0/model_index.json` already exists, `--download-weights` verifies and reuses that checkpoint instead of downloading the approximately 13.7 GB archive again.

`--finalize-existing` is provided for interrupted setup recovery after the environment, dependencies, and checkpoint already exist. It performs no remote fetch and no dependency reinstall; it verifies the pinned local checkout, resolves the environment interpreter through `sys.executable`, verifies the existing checkpoint, performs the runtime probe, and writes the runtime lock. This avoids repeating the multi-gigabyte model download after a late setup failure.

## Runtime-probe JSON capture correction

Some RollingDepth/diffusers imports can emit informational text to standard output before the setup helper serializes its probe. Capturing that entire stdout stream directly into `PROBE_JSON` can therefore produce a non-JSON string and fail later with `json.JSONDecodeError`. The setup helper now writes the structured probe payload directly to a temporary JSON file from inside the isolated interpreter and reads only that file back for runtime-lock creation. Import-time stdout is no longer treated as structured data. This is a setup-helper robustness correction only; it does not alter the pinned RollingDepth checkout, model weights, provider semantics, or authority boundaries.
