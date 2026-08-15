# Phase 9.5D — Persistent NeRF Representation

Phase 9.5D persists an already-qualified Phase 9.5C Nerfstudio/Nerfacto result into the existing `PersistentReconstructionScene` schema. It does not add a new scene format and does not start Nerfstudio, tiny-cuda-nn, training, or rendering.

## Locked behavior

- Canonical geometry remains authoritative.
- NeRF remains `appearance_only` and never gains geometry, metric-scale, measurement, QC, or certified-metrology authority.
- The parent persistent scene and canonical geometry artifact remain unchanged; persistence creates a new immutable scene revision.
- The NeRF checkpoint, immutable training config, backend preview, runtime evidence, 9.5C provenance, and a self-owned copy of the NeRF input dataset are integrity-bound.
- Dataset files are recursively hashed through `dataset_integrity.json`; restart reload verifies those hashes in addition to the existing persistent-scene file references.
- `runtime_load_spec.json` records durable checkpoint/dataset locations for the later isolated interactive renderer. The persisted training config is not silently rewritten; 9.5E must stage/rebind it explicitly if needed.
- Backend-free reload preserves the existing static NeRF preview presentation contract.

## Scope

Phase 9.5D does not implement interactive NeRF rendering, representation controls, resource lifecycle, quality qualification, or final Phase 9.5 closure. Those remain 9.5E–9.5H.
