# Phase 8.7E4 — NeRF Performance + Artifact Hardening

## Baseline

- Branch: `phase8-reconstruction`
- Required user baseline: `75d7457a5cdff78d672efcf17200facc753ed8b8`
- Previous gate: Phase 8.7E3 real-scene Nerfacto validation complete.

## Scope

E4 is NeRF-backend-specific hardening only. It adds:

- real Nerfacto training duration measurement;
- process-tree CUDA VRAM sampling during training;
- fresh checkpoint reload timing;
- warm native full-image render timing (file I/O excluded);
- render mean / p50 / p95 latency and FPS;
- process-tree CUDA VRAM sampling during rendering;
- checkpoint/config integrity evidence and SHA-256 fingerprints;
- missing-artifact and malformed-checkpoint rejection;
- backend-process cancellation, timeout, crash and partial-artifact scenarios;
- process-group / child-process cleanup;
- a real backend recovery execution after every injected fault;
- an explicit RTX 5090 readiness contract with no fabricated performance numbers.

E4 does **not** implement whole reconstruction-engine recovery (Phase 8.7G), whole-application GPU profiling (Phase 8.9D), UI integration, canonical-geometry authority or certified metrology.

## Runtime isolation

Real Nerfacto execution remains in the previously qualified external environment:

`/home/eiatech/anaconda3/envs/eia-nerf-nerfstudio`

Production `(depth)` remains orchestration-only. No Nerfstudio, tiny-cuda-nn, Torch, LPIPS or Gaussian runtime dependency is added to `core/reconstruction` or `pyproject.toml` by E4.

## Performance definitions

- **Training duration:** wall-clock duration of the actual bounded Nerfacto training call.
- **Training peak VRAM:** maximum process-tree GPU memory observed with `nvidia-smi` while the external worker trains.
- **Checkpoint reload:** fresh Nerfstudio `eval_setup(..., test_mode="test")` reload of the produced config/checkpoint.
- **Render latency:** native `model.get_outputs_for_camera_ray_bundle(...)` CUDA execution for a full held-out image; one warm-up render is excluded and file writing is not timed.
- **Render FPS:** `1000 / mean_render_latency_ms` for the measured native full-image render loop.
- **Render peak VRAM:** maximum process-tree GPU memory observed during the measured render loop.

These are backend-level E4 measurements, not whole-application production profiling.

## RTX 5090 readiness policy

E4 defines the RTX 5090 hardware/readiness target but deliberately does not claim that the E2 CUDA 11.8 / SM75 isolated environment is Blackwell-qualified. No synthetic RTX 5090 timing, VRAM or FPS values are generated. A separate Blackwell-capable isolated runtime and real hardware qualification are required before real RTX 5090 execution; whole-application RTX 5090 profiling remains Phase 8.9D.

## Artifact hardening

The worker validates that produced config/checkpoint artifacts:

- exist and are non-empty;
- use expected YAML / checkpoint suffixes;
- remain inside the controlled artifact root;
- have recorded SHA-256 digests;
- contain a loadable Nerfstudio checkpoint mapping with `pipeline` and `step` keys;
- can be reloaded into a fresh pipeline and render finite RGB output.

The valid checkpoint is never mutated by rejection tests. Missing and deliberately malformed copies must be rejected.

## Backend robustness

Test-only fault injection is gated by `EIA_NERF_E4_TEST_FAULTS=1` and never enabled in normal execution. E4 exercises:

1. controlled cancellation;
2. timeout;
3. worker crash;
4. invalid partial-result artifact.

Each fault attempt spawns a child process so the supervisor proves process-group/child cleanup. After cleanup, the same external Nerfacto backend must successfully complete a real bounded recovery training/reload/render run.

This proves only the NeRF backend boundary. Whole-engine cancellation/recovery/resource cleanup remains Phase 8.7G.

## Automated validation

Development validation before packaging:

- E4 quick: 77 passed
- E4 integration: 8 passed
- E4 stress: 6 passed
- E4 contracts smoke: PASS
- E3 backward quick: 60 passed
- E2 backward quick: 44 passed
- E1 backward quick: 21 passed
- D3 backward quick: 55 passed
- D2 backward quick: 31 passed
- Python compile: PASS
- shell syntax: PASS
- `git diff --check`: PASS
- production-core heavy-import isolation: PASS
- `pyproject.toml` unchanged: PASS

## Required real RTX 2060 validation before commit

Using the already-prepared E3 real-scene manifests:

```bash
rm -rf /tmp/phase8_7e4_nerf_performance
python scripts/run_nerf_performance_hardening.py \
  --dataset-manifest /tmp/phase8_7e3_poster_prepared/dataset.json \
  --split-manifest /tmp/phase8_7e3_poster_prepared/split.json \
  --source-label nerfstudio-official-poster \
  --profile efficient \
  --training-steps 1000 \
  --render-repeats 8 \
  --hardware-profile rtx_2060_6gb
```

Then:

```bash
rm -rf /tmp/phase8_7e4_nerf_robustness
python scripts/run_nerf_backend_robustness.py \
  --dataset-manifest /tmp/phase8_7e3_poster_prepared/dataset.json \
  --split-manifest /tmp/phase8_7e3_poster_prepared/split.json \
  --profile efficient \
  --recovery-training-steps 16
```

Do not commit E4 until both real-hardware commands pass and their evidence is inspected.
