# Phase 8.8A4 — Scene / Checkpoint / Resume Bridge

## Scope

A4 exposes existing Phase 8.7 persistence/recovery contracts to the application controller. It does not add a Reconstruction page, enable navigation, change reconstruction algorithms, or introduce a new persistence schema.

Application ownership after A4:

- A3 sparse offline output remains a partial result unless an integrated worker explicitly emits a valid persistent-scene manifest.
- `PersistentReconstructionScene` remains the durable completed/degraded integrated scene contract.
- `ReconstructionEngineCheckpoint` remains the whole-engine resumable-prefix contract.
- resume compatibility continues to be decided by the existing G2 request/plan/provider/payload identity checks.

## Safety locks

- Sparse SfM is never promoted to canonical geometry.
- Persistent scenes still require the complete F5 core stage set through semantics and inspection/QC.
- Gaussian/NeRF authority remains appearance-only through the existing F5 schema.
- `certified_metrology` remains false.
- persisted `reconstruction_ui_enabled` remains false; UI activation belongs to later Phase 8.8 work.
- no PyCOLMAP, Stella, gsplat, Nerfstudio, or DA3 import is introduced in application persistence code.
- no `tests` fixture is imported by production application code or the A4 smoke script.

## Focused validation

```bash
./scripts/run_phase8_8a4_tests.sh quick
./scripts/run_phase8_8a4_tests.sh integration
./scripts/run_phase8_8a4_tests.sh backward
```

Expected focused counts at delivery: `15 passed` for quick and `1 passed` for integration. The backward gate must preserve the A3 `20 passed` quick group and `1 passed` integration group.

## Controller-level persistence smoke

The smoke starts no heavy backend. It exercises the application bridge over the existing F5/G2 contracts and can additionally consume the real A3 result to prove it stays partial:

```bash
rm -rf /tmp/phase8_8a4_scene_checkpoint_bridge
python scripts/run_phase8_8a4_scene_checkpoint_bridge_smoke.py \
  --offline-result /tmp/phase8_8a3_offline_controller_pycolmap/offline_result.json
```

Expected result:

```text
Phase 8.8A4 scene/checkpoint/resume bridge smoke: PASS
```

The summary must show:

- real A3 result preserved as partial;
- persistent integrated scene canonical geometry authoritative;
- certified metrology false;
- reconstruction UI false in the persisted engine contract;
- checkpoint compatibility true;
- three committed prefix stages reused;
- resume begins at dense geometry;
- ingestion/pose/depth are not re-executed during the resumed deterministic validation.

## Commit gate

Do not commit until focused tests, backward tests, `git diff --check`, and the A4 smoke are green in the production `depth` environment.
