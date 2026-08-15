# Phase 9.4H — Real-Scene Qualification & Phase 9.4 Closure

Baseline: clean pushed `phase9-digital-twin` at `029643aa43cfbbcf714e99a1de91d32d5767b3cc` (Phase 9.4G closed).

## Purpose

Phase 9.4H closes the complete Interactive Gaussian Splatting phase. It does not add a new Gaussian feature, change persistence, redesign the viewer, or grant Gaussian output any geometry/metric/measurement/QC authority.

The closure gate proves the already-implemented Phase 9.4A→9.4G chain on the real target workstation and then runs the complete current-tree automated regression.

## Real closure campaign

The real campaign starts from the already-qualified Phase 9.3J South Building canonical scene. It deliberately does **not** rerun DA3. It then performs a fresh Gaussian phase chain:

1. **9.4A** — fresh read-only qualification of the isolated `eia-gaussian-gsplat` runtime and clean pinned checkout;
2. **9.4B** — fresh exact source-image/canonical-camera Gaussian input bridge;
3. **9.4C** — fresh real CUDA gsplat optimization using the qualified external runtime;
4. **9.4D** — fresh immutable Gaussian appearance persistence plus fresh-process reload verification;
5. **9.4E** — backend-free interactive persisted-Gaussian load with canonical picking retained;
6. **9.4F** — density/size/opacity presentation controls and scene-layer behavior;
7. **9.4G** — repeated bounded load/control/release cycles with CPU staging collection and advisory UI-side timing evidence.

The closure result is written to:

`/tmp/phase9_4h_real_scene/phase9_4h_qualification_summary.json`

The newly qualified persistent Gaussian scene is written under:

`/tmp/phase9_4h_real_scene/persistent/`

## Authority rules

The closure must preserve all existing authority boundaries:

- persistent canonical geometry remains authoritative;
- Gaussian remains `appearance_only`;
- Gaussian canonical-geometry authority is false;
- Gaussian metric-scale authority is false;
- Gaussian measurement authority is false;
- Gaussian QC/reference authority is false;
- `certified_metrology=false`;
- the South Building canonical scene remains arbitrary scale / `unit_to_metre=null` unless separate real scale evidence is supplied;
- the interactive ruler therefore exposes native reconstruction units but no fabricated metric value on this scene;
- Gaussian display controls and lifecycle diagnostics never modify `scene.pt` or persistent authority;
- no heavy Gaussian runtime is imported into the production `depth` process;
- no Graphdeco or Nerfstudio substitution/fallback occurs.

## Automated gates

Focused closure checks:

```bash
./scripts/run_phase9_4h_tests.sh quick
./scripts/run_phase9_4h_tests.sh integration
./scripts/run_phase9_4h_tests.sh stress
./scripts/run_phase9_4h_tests.sh backward
```

The integration gate also runs the deterministic closure smoke. These checks are backend-free.

## Real target-machine campaign

```bash
rm -rf /tmp/phase9_4h_real_scene
python scripts/run_phase9_4h_real_scene_qualification.py \
  --phase93j-summary /tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json \
  --output-dir /tmp/phase9_4h_real_scene \
  --profile efficient \
  --max-gaussians 2048 \
  --resource-cycles 12
```

A successful campaign must report `status=PASS` and prove real CUDA optimization, non-empty persisted/reloadable Gaussian appearance, exact bridge/optimization/persistence fingerprint lineage, clean runtime revision, positive real optimization VRAM evidence, interactive appearance readiness, canonical-only picking, display-control non-mutation, all requested release checks passing, and no production-runtime Gaussian contamination.

The real campaign records actual RTX-2060 evidence only. It must not extrapolate performance to RTX 4090/5090 hardware.

## Manual closure validation

Load the new closure scene in the application:

`/tmp/phase9_4h_real_scene/persistent/persistent_reconstruction_scene.json`

Verify:

- canonical surface and cameras load normally;
- Interactive / Preview / Hide switch correctly;
- Orbit / Pan / Zoom remain responsive;
- Density / Splat size / Opacity controls remain presentation-only;
- Gaussian Scene Layer OFF releases appearance while canonical geometry remains;
- Gaussian Scene Layer ON reloads appearance on demand;
- the ruler continues to pick canonical geometry and reports metric estimate unavailable for the arbitrary-scale South Building scene;
- application shutdown exits cleanly.

For an explicit repeated real Qt/VisPy lifecycle check against the newly-created closure scene:

```bash
python scripts/run_phase9_4g_gaussian_lifecycle_preview.py \
  --scene-manifest /tmp/phase9_4h_real_scene/persistent/persistent_reconstruction_scene.json \
  --cycles 12 \
  --show
```

## Full Phase 9.4 regression

Only after the focused and real closure gates are green:

```bash
./scripts/run_phase9_4h_tests.sh full
```

`full` runs the complete current-tree unit, UI, integration and stress suites plus the deterministic 9.4H closure smoke. Optional target-hardware tests may remain explicitly skipped when their optional external dependency is unavailable; unexpected failures are blockers.

## Closure rule

Phase 9.4 may be committed/closed only when:

- focused quick/integration/stress/backward gates are green;
- real Phase 9.4H Gaussian campaign reports `PASS`;
- the newly-created persistent closure scene manually loads and behaves correctly;
- the real Qt/VisPy lifecycle check is clean;
- full current-tree regression is green apart from explicitly expected optional skips;
- Git diff/scope is clean and no generated datasets, model weights, temporary outputs, screenshots, caches, external environments or core dumps are staged.

No `v0.9.0` tag, branch merge, NeRF implementation, or unified Digital Twin release belongs to Phase 9.4H. The next locked roadmap item remains Phase 9.5 Interactive NeRF.
