# Phase 9.5H — Real-Scene Qualification & Phase 9.5 Closure

Baseline: clean pushed `phase9-digital-twin` at `8a2b97e85ed6de584029d1cf8841eac8afd0a0de` (Phase 9.5G closed).

## Purpose

Phase 9.5H closes the complete Interactive NeRF / Nerfacto phase. It adds no new NeRF product feature, persistence schema, geometry authority, measurement authority, or UI architecture.

The closure gate proves the already-implemented Phase 9.5A→9.5G chain on the real RTX 2060 target workstation, characterizes the current South Building visual quality honestly, and then runs the complete current-tree automated regression.

## Real closure campaign

The real campaign starts from the already-qualified Phase 9.3J South Building canonical scene. DA3 is not rerun. It performs:

1. **9.5A** — fresh read-only qualification of the isolated `eia-nerf-nerfstudio` runtime and clean pinned Nerfstudio/tiny-cuda-nn checkouts;
2. **9.5B** — fresh exact source-image/canonical-camera NeRF input bridge with retained intrinsics;
3. **9.5C** — fresh real CUDA Nerfacto optimization, checkpoint creation and preview proof;
4. **9.5D** — fresh immutable NeRF appearance persistence and restart/reload verification;
5. **9.5E** — real isolated interactive render from the persisted checkpoint;
6. **9.5F** — render-detail/opacity/auto-refresh and NeRF scene-layer contract audit;
7. **9.5G** — repeated worker start/render/release cycles with bounded queue/frame residency, torch CUDA telemetry and `nvidia-smi` PID-release proof;
8. **9.5H quality characterization** — render the single held-out validation camera and measure PSNR/SSIM/LPIPS through the already-qualified isolated backend-neutral Phase 8.7D3 metrics worker.

The closure result is written to:

`/tmp/phase9_5h_real_scene/phase9_5h_qualification_summary.json`

The newly-qualified persistent NeRF scene is written under:

`/tmp/phase9_5h_real_scene/persistent/`

## Visual-quality truthfulness

The current South Building Phase 9.5 dataset has only **3 source frames: 2 training + 1 held-out validation**. This is deliberately below the established Phase 8.7E3 sanity-policy requirement of at least **8 training + 3 held-out** views.

Therefore Phase 9.5H:

- measures the real single held-out view with PSNR/SSIM/LPIPS;
- requires finite, structurally valid metric evidence and clean worker lifecycle;
- labels this evidence `single_view_heldout_observational_quality`;
- sets `quality_characterized=true` and `visual_quality_evaluated=true`;
- keeps `quality_certified=false` and `production_quality_certified=false`;
- explicitly records `insufficient_views_for_phase8_e3_sanity_policy`;
- does **not** invent a new quality threshold to make the 3-frame demo pass or fail.

A denser real dataset and stronger training campaign may be used later when production appearance quality is required, without changing the Phase 9 authority model.

## Authority rules

The closure must preserve:

- persistent canonical geometry remains authoritative;
- NeRF remains `appearance_only`;
- NeRF canonical-geometry authority is false;
- NeRF metric-scale authority is false;
- NeRF measurement authority is false;
- NeRF QC/reference authority is false;
- `certified_metrology=false`;
- South Building remains arbitrary scale / `unit_to_metre=null` unless separate scale evidence is supplied;
- ruler/picking remain canonical-only and no metric value is fabricated;
- NeRF display controls never modify the persistent checkpoint/config/dataset;
- no Nerfstudio or tiny-cuda-nn module is imported into the production `depth` process;
- no Gaussian↔NeRF fallback substitution occurs.

## Automated gates

```bash
./scripts/run_phase9_5h_tests.sh quick
./scripts/run_phase9_5h_tests.sh integration
./scripts/run_phase9_5h_tests.sh stress
./scripts/run_phase9_5h_tests.sh backward
```

The integration gate also runs the deterministic backend-free closure smoke.

## Real target-machine campaign

```bash
rm -rf /tmp/phase9_5h_real_scene
python scripts/run_phase9_5h_real_scene_qualification.py \
  --phase93j-summary /tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json \
  --output-dir /tmp/phase9_5h_real_scene \
  --profile efficient \
  --training-steps 64 \
  --resource-cycles 4
```

The real campaign must report `status=PASS`, exact bridge/fingerprint lineage, clean pinned external revisions, OOM-free real CUDA optimization, durable restart-safe persistence, real interactive rendering, canonical-only picking, bounded lifecycle/resource evidence, worker PID release, finite single-held-out-view quality metrics, and no production-runtime heavy-module contamination.

The campaign records actual RTX 2060 evidence only. It does not extrapolate performance to RTX 4090/5090 hardware.

## Manual closure validation

Load:

`/tmp/phase9_5h_real_scene/persistent/persistent_reconstruction_scene.json`

Verify:

- canonical surface and camera context load normally;
- NeRF Interactive / Preview / Hide switch correctly;
- Orbit / Pan / Zoom / Fit / Reset request fresh views correctly;
- Fast / Balanced / Detail, opacity and auto-refresh behave as presentation-only controls;
- NeRF Scene Layer OFF releases appearance while canonical geometry remains;
- NeRF Scene Layer ON reloads the isolated renderer on demand;
- ruler pauses NeRF appearance, picks canonical geometry only, and reports metric estimate unavailable on the arbitrary-scale scene;
- Resources / Performance / Lifecycle diagnostics remain truthful;
- normal application shutdown leaves no `nerfstudio_interactive_worker.py` process.

The quality contact sheet is written under:

`/tmp/phase9_5h_real_scene/quality/heldout_contact_sheet.png`

It is evidence for the current 3-frame experiment, not production-quality certification.

## Full Phase 9.5 regression

Only after focused, real and manual closure gates are green:

```bash
./scripts/run_phase9_5h_tests.sh full
```

`full` runs the complete current-tree unit, UI, integration and stress suites plus the deterministic 9.5H closure smoke. Explicitly optional external-hardware tests may remain skipped when their documented optional dependency is unavailable; unexpected failures are blockers.

## Closure rule

Phase 9.5 may be committed/closed only when:

- quick/integration/stress/backward gates are green;
- the real Phase 9.5H campaign reports `PASS`;
- held-out quality is characterized without overclaiming production certification;
- the newly-created persistent closure scene manually loads and behaves correctly;
- worker/process/GPU lifecycle cleanup is clean;
- full current-tree regression is green apart from explicitly expected optional skips;
- Git scope is clean and no datasets, model weights, generated renders/contact sheets, temporary outputs, caches, external environments or core dumps are staged.

No `v0.9.0` tag, branch merge, or unified Digital Twin implementation belongs to Phase 9.5H. The next locked roadmap item remains **Phase 9.6 — Unified Digital Twin Workspace**.
