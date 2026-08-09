# Phase 8.7F2 — Depth Anything 3 qualification + isolated provider

## Scope

Phase 8.7F2 adds a dependency-light Depth Anything 3 provider boundary for the
Phase 8.7F1 reconstruction orchestrator. DA3 remains an **external supporting
geometry provider** for camera pose and relative multi-view depth.

This step does not bind DA3 output into canonical dense geometry, semantics or
inspection/QC; those remain Phase 8.7F3 responsibilities. Gaussian/NeRF binding
remains F4, PersistentReconstructionScene remains F5, and reconstruction UI
remains disabled until Phase 8.8.

## Isolation

- Production environment `(depth)` receives no DA3 / torch / xformers dependency.
- Runtime lives in `/home/eiatech/anaconda3/envs/eia-da3`.
- Upstream checkout lives in `/home/eiatech/Desktop/eia-da3-backends/depth-anything-3`.
- The setup requires an explicit full upstream Git revision and records it in
  `.eia_da3_runtime_lock.json`.
- Model loading resolves an immutable Hugging Face model revision before loading
  when no explicit model revision is supplied; the exact model revision is
  persisted in qualification evidence.
- Core files do not import `torch`, `xformers` or `depth_anything_3`.
- A bounded Turing compatibility patch uses native-bf16 capability detection on
  upstream revisions that still use emulated bf16 detection.

## F2 provider contract

Provider: `depth-anything-3-external`

Roles:

- `camera_pose`
- `depth`

Authority:

- supporting geometry only
- relative / arbitrary scale
- `canonical_geometry_authoritative=False`
- `certified_metrology=False`

The provider executor caches one isolated DA3 inference per orchestration request
and reuses that result for both camera-pose and depth stages.

Qualification profiles in F2:

- efficient / RTX 2060 gate: `depth-anything/DA3-SMALL`, process resolution 392
- balanced candidate: `depth-anything/DA3-BASE`

Larger/nested/metric variants are deliberately outside this F2 provider allowlist.

## Coordinate normalization

DA3 worker output is persisted as:

- depth `(N,H,W)`
- confidence `(N,H,W)`
- camera extrinsics `(N,3,4)` or `(N,4,4)` in OpenCV/COLMAP world-to-camera form
- intrinsics `(N,3,3)`

Core validation converts each world-to-camera extrinsic by rigid inversion into
EIA's internal `T_reconstruction_from_camera` pose convention. F2 does not infer
or certify metric scale from these outputs.

## Automated gates

Expected focused results for this delivery:

```text
./scripts/run_phase8_7f2_tests.sh quick        -> 36 passed
./scripts/run_phase8_7f2_tests.sh integration  -> 1 passed
./scripts/run_phase8_7f2_tests.sh stress       -> 2 passed
python scripts/run_da3_provider_contracts_smoke.py -> PASS
```

The contract smoke uses a deterministic fake external worker. It proves that the
same DA3 result feeds both pose and depth stages while canonical geometry remains
owned by later pipeline stages.

## Required real qualification before commit

1. Resolve the current upstream DA3 `main` commit and pass the full 40-character
   revision explicitly to the isolated setup script.
2. Run the environment probe and confirm the runtime lock, CUDA visibility,
   exact upstream revision and Turing compatibility guard.
3. Run real multi-view DA3-SMALL inference on the existing poster scene images.
4. Validate depth/confidence/intrinsics and conversion to
   `T_reconstruction_from_camera`.
5. Record real RTX 2060 inference duration and peak allocated VRAM.
6. Inspect the generated depth contact sheet visually.
7. Re-run the focused backward-compatibility gates.

No F2 commit should be made until the real RTX 2060 qualification and visual
inspection are green.

## Deferred ownership

- F3: canonical geometry + semantics + QC integration
- F4: Gaussian + NeRF appearance integration
- F5: PersistentReconstructionScene + final F validation
- G: whole-engine cancellation/recovery/checkpoint/endurance hardening
- 8.8: reconstruction application/UI
- 8.9D: whole-application RTX 2060 / 4090 / 5090 production profiling
