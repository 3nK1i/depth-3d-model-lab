# Phase 8.9F5 — Persistent Multi-Artifact Scene Qualification

Status: workstation real-evidence qualification gate.

## Purpose

Qualify the existing `PersistentReconstructionScene` contract with the already-qualified real Phase 8.9F evidence:

- F2 real South Building canonical surface,
- F3 real isolated gsplat scene,
- F4 real isolated Nerfstudio/Nerfacto checkpoint + config.

F5 starts no heavy backend and creates no new persistence format or reconstruction algorithm.

## Locked authority

- canonical geometry: authoritative,
- Gaussian: `appearance_only`,
- NeRF/Nerfacto: `appearance_only`,
- certified metrology: `false`.

Gaussian or NeRF must never become canonical, metric, measurement, QC or reference authority.

## Truthful semantic/QC boundary

The existing persistent-scene schema requires semantic-geometry and inspection/QC stage records. F2-F4 did not re-run real South Building semantics or inspection. F5 therefore binds those two required records as **explicit contract-only context**. This is a persistence/schema qualification only and must not be reported as new real South Building semantic or QC evidence.

## Real artifact persistence

The runner copies the required real artifacts into the F5 output root before persistence so the scene owns bounded self-contained references:

- canonical OBJ,
- gsplat `scene.pt`,
- backend-authored Gaussian preview,
- Nerfacto checkpoint,
- Nerfacto config,
- backend-authored NeRF preview.

The existing persistence layer records absolute paths, size and SHA-256 for these files.

## Required gates

- exact clean F4 baseline before overlay,
- source F2/F3/F4 summaries all PASS,
- 9-stage persistent integrated scene record,
- deterministic scene identity/fingerprint,
- byte/checksum reproducible manifest writes,
- all file references verify,
- referenced-file tamper fails closed,
- referenced-file missing fails closed,
- manifest tamper fails closed,
- persisted Gaussian and NeRF previews resolve through the existing app presentation contracts without starting a backend,
- production `depth` cannot import DA3/PyCOLMAP/gsplat/Nerfstudio/tiny-cuda-nn,
- zero production code changes.

## Expected workload

No PyCOLMAP, DA3, Gaussian training or Nerfacto training is started. F5 performs local file copies/hashes and persistence/reload checks only. The Nerfacto checkpoint is large, so several SHA-256 passes are expected.

## Output

Default output root:

`/tmp/phase8_9f5_persistent_scene`

Primary persistent scene:

`/tmp/phase8_9f5_persistent_scene/scene-a/persistent_reconstruction_scene.json`

Qualification summary:

`/tmp/phase8_9f5_persistent_scene/summary.json`

## Next gate

A PASS advances to **8.9F6 — Actual Application End-to-End Acceptance**, where the existing PyQt Reconstruction UI must load this real persistent scene and qualify Canonical/Gaussian/NeRF presentation switching, fallback, cleanup and 1366×768 behavior without loading heavy runtimes into the application process.
