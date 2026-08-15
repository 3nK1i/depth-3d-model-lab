# Phase 9.4D — Persistent Gaussian Representation

Baseline: `74d79e79b5b30594fa3e4c77809b9667c643760c` on `phase9-digital-twin`.

## Scope

Phase 9.4D persists an already-qualified Phase 9.4C Gaussian appearance artifact by reusing the existing Phase 8.7F5 persistent reconstruction-scene schema. It does not introduce a second scene format, does not change the persistent-scene schema version, and does not add interactive rendering.

The adapter:

- verifies the parent persistent scene and every existing file reference before attachment;
- requires an optimized, render-capable, appearance-only Gaussian artifact;
- requires exact 9.4B dataset fingerprint compatibility, canonical reconstruction-frame compatibility, and canonical scale-state compatibility;
- requires the 9.4C source artifact manifest and real-CUDA qualification metrics to agree with the supplied artifact;
- materializes durable scene-owned Gaussian files under `appearance/gaussian/`;
- writes the existing `GaussianSceneArtifact` manifest with durable paths;
- adds one `gaussian_appearance` artifact to the existing persistent scene contract;
- preserves the parent canonical geometry artifact record exactly;
- records parent scene ID/fingerprint lineage;
- creates a new immutable persistent scene revision/fingerprint;
- verifies immediate round-trip reload and file-reference integrity;
- supports a fresh-process restart/reload validation with no Gaussian backend execution.

## Authority boundary

- Persistent canonical geometry remains authoritative.
- Gaussian is `appearance_only`.
- Gaussian is not canonical geometry authority.
- Gaussian is not metric-scale authority.
- Gaussian is not measurement authority.
- Gaussian is not QC/reference authority.
- `certified_metrology` remains `false`.
- Arbitrary scene scale remains arbitrary; no `unit_to_metre` is invented.

## Runtime boundary

Phase 9.4D never imports or starts gsplat/Graphdeco/Nerfstudio. It persists and reloads the artifact produced by 9.4C only. Interactive Gaussian rendering remains Phase 9.4E.

## Focused validation

```bash
./scripts/run_phase9_4d_tests.sh quick
./scripts/run_phase9_4d_tests.sh integration
./scripts/run_phase9_4d_tests.sh backward
python scripts/run_phase9_4d_gaussian_persistence_smoke.py
```

Expected dependency-light results before target-machine validation:

- quick: 10 passed
- integration: 2 passed + smoke PASS
- backward: 63 passed

## Real RTX-2060 persistence validation

Requires the already-PASS real 9.4B and 9.4C temporary evidence:

```bash
rm -rf /tmp/phase9_4d_persistent_gaussian_scene
python scripts/run_phase9_4d_real_gaussian_persistence.py \
  --optimization-summary /tmp/phase9_4c_real_gaussian_optimization/phase9_4c_real_optimization_summary.json \
  --bridge-summary /tmp/phase9_4b_real_gaussian_input/phase9_4b_real_bridge_summary.json \
  --output-dir /tmp/phase9_4d_persistent_gaussian_scene
```

Then prove restart/reload from a fresh Python process:

```bash
python scripts/run_phase9_4d_gaussian_reload_check.py \
  --summary /tmp/phase9_4d_persistent_gaussian_scene/phase9_4d_persistence_summary.json
```

The reload must not start Gaussian/DA3/NeRF work. The existing static Gaussian preview presentation may resolve the persisted `preview.png`; interactive splat rendering is intentionally deferred to 9.4E.
