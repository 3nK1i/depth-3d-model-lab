# Phase 8.9D — Whole-Application GPU Production Profiling

Status: implementation/protocol prepared; production RTX 2060 measurement pending workstation execution.

## Scope

Phase 8.9D qualifies the existing hardware-adaptive application without introducing GPU-name-specific production branching. Real measurements are accepted only from physically present hardware. Synthetic RTX 2060 / RTX 4090 / RTX 5090 entries are policy-readiness contracts only and must never be reported as measured performance.

Current operator hardware available for real qualification: NVIDIA GeForce RTX 2060 6 GB.

RTX 4090 / RTX 5090 handling remains automatic through the existing VRAM/capability-driven compute and reconstruction profile resolvers. Their deterministic readiness checks do not substitute for future real-machine qualification.

## Profiling contract

`scripts/run_phase8_9d_gpu_production_profile.py` records:

- detected physical GPU and CUDA/PyTorch compatibility;
- startup/free VRAM and `nvidia-smi` device telemetry;
- existing automatic model compute profile;
- existing automatic reconstruction profile and memory budget;
- optional actual `MainWindow` construction at 1366×768;
- application shell startup/shutdown time;
- process RSS samples;
- process GPU-memory samples when exposed by `nvidia-smi`;
- loaded heavy-runtime module check.

The UI probe uses an isolated Qt settings directory so it does not overwrite the operator's normal saved layout.

## Safety / authority

- No new GPU-selection architecture.
- No source/config edit is required when moving to a supported GPU.
- No RTX 4090/5090 performance result may be claimed until measured on that hardware.
- No DA3, Gaussian, NeRF, PyCOLMAP, or SLAM heavy backend is intentionally started by 8.9D.
- Heavy external backends remain isolated.
- Geometry/QC/metrology authority is unchanged.
- Real DA3/Gaussian/NeRF campaign remains reserved for Phase 8.9F.

## Gates

1. `./scripts/run_phase8_9d_tests.sh quick`
2. `./scripts/run_phase8_9d_tests.sh affected`
3. policy-readiness CLI must mark every synthetic target as `measured=false`;
4. real physically-present-GPU profile with `--with-ui` must PASS;
5. `application_shell.heavy_runtime_leak_detected=false`;
6. run the existing isolated cross-model production benchmark with `--profile auto --smoke-test --require-all` to collect real model timing/VRAM evidence on the physically present GPU;
7. no fabricated RTX 4090/5090 measurement claims;
8. `git diff --check` and exact-scope staging before commit.

The cross-model benchmark is existing qualified infrastructure and remains process-isolated per model. Its deterministic assets measure runtime/output contracts, not ground-truth depth accuracy. It is intentionally reused rather than duplicated by 8.9D.
