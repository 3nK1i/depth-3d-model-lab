# Phase 8.9C — Persistence / Reload / Reproducibility Qualification

Phase 8.9C qualifies the **existing** Phase 8.7F5 persistent-scene and Phase 8.7G2 checkpoint/resume contracts for production release. Source inspection found no reason to introduce a new persistence format, controller, checkpoint system, or reconstruction lifecycle.

## Production-code decision

**No production code is changed by 8.9C.**

The existing implementation already provides:

- stable `PersistentReconstructionScene` identity and normalized SHA-256 scene fingerprint;
- manifest SHA-256 sidecars and semantic fingerprint validation;
- persistent provider provenance and fallback/degradation evidence;
- file-backed artifact path/size/SHA-256 references with explicit verification;
- same-session application reload with referenced-file verification;
- whole-engine checkpoint manifest/checksum and semantic fingerprint validation;
- exact request/plan/provider/payload resume compatibility;
- canonical geometry authority preservation;
- Gaussian/NeRF `appearance_only` persistence;
- `certified_metrology=false`.

`Save Scene` remains manifest persistence only. It does not copy external artifact payloads and is not a product-export bundle; Phase 8.9A owns product export.

## Added qualification evidence

8.9C adds tests and a dependency-light smoke that verify:

- byte-identical manifests/checksum sidecars when the exact same scene is saved to different directories;
- stable scene ID/fingerprint through save → load → re-save → reload;
- manifest checksum tamper rejection;
- normalized semantic tamper rejection even when a manifest checksum is recomputed;
- missing referenced artifact rejection;
- referenced artifact size-drift rejection;
- same-size referenced artifact content/SHA drift rejection;
- application reload preserves the bound scene when verification fails;
- canonical geometry remains authoritative after reload;
- Gaussian and NeRF remain appearance-only after reload;
- no certified-metrology claim appears;
- checkpoint reload preserves semantic fingerprint/stage prefix;
- provider-version drift rejects resume;
- missing payload references reject resume;
- repeated persistence/reload cycles retain exact scene identity.

The qualification smoke uses deterministic core executors and real temporary files only. It starts no PyCOLMAP, DA3, Gaussian, Nerfstudio/NeRF, or other heavy backend. Real DA3/Gaussian/NeRF artifact qualification remains reserved for the later Phase 8.9 real-scene campaign.

## Validation

Run in the production `depth` environment:

```bash
./scripts/run_phase8_9c_tests.sh quick
./scripts/run_phase8_9c_tests.sh affected

rm -rf /tmp/phase8_9c_persistence_reproducibility
python scripts/run_phase8_9c_persistence_reproducibility_smoke.py \
  --output-dir /tmp/phase8_9c_persistence_reproducibility
cat /tmp/phase8_9c_persistence_reproducibility/summary.json
```

Expected smoke invariants include:

- `pass=true`;
- `production_code_changed_by_8_9c=false`;
- `new_persistence_format_created=false`;
- `heavy_backends_started=false`;
- scene byte/checksum/identity reproducibility true;
- at least three referenced appearance files verified;
- manifest and referenced-artifact tamper rejection true;
- canonical geometry authoritative;
- Gaussian/NeRF authority `appearance_only`;
- certified metrology false;
- checkpoint compatibility true for exact inputs;
- provider drift and missing-payload compatibility both rejected.

Manual validation for 8.9C is limited to the existing Reconstruction `SESSION & PERSISTENCE` flow: Load Scene → Save Scene → Reload on a qualified scene, plus a small backward sanity that Product Export remains separate. Full real-scene/backends validation belongs to Phase 8.9F and full regression belongs to Phase 8.9G.
