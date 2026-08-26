# E1 — Qualified Ten-View Voxel Fusion Validation

E1 is the first real multi-view fusion step after the D4.3 ten-view chain is fully qualified.

## Scope

E1 consumes only the selected passing D4.3 chain and reuses the existing `DenseVoxelFusion` production kernel. It does **not** rerun PyCOLMAP or DA3. The saved aligned depth maps are already expressed in the arbitrary offline-SfM reconstruction gauge and are therefore bound as reconstruction-unit depth with identity alignment.

The initial fusion policy is intentionally conservative and deterministic:

- voxel edge length: robust sparse-SfM scene diagonal × `0.004` by default;
- full source pixel sampling by default (`pixel_stride=1`);
- raw fused voxel map retained for diagnostics;
- primary E1 output keeps voxels supported by at least **2 qualified views**;
- a secondary high-support cloud keeps voxels supported by at least **3 views**;
- no capacity truncation is accepted;
- uniform per-pixel fusion confidence is used in E1 so no new heuristic weighting is hidden in the first real fusion result.

## Authority boundary

E1 applies real spatial voxel merging and multi-view support filtering, but remains validation-only geometry:

- SfM scale remains arbitrary;
- no metres or physical metric scale are claimed;
- no canonical persistence authority is promoted;
- no measurement authority is promoted;
- no surface mesh is generated;
- no certified metrology is claimed.

## Acceptance

E1 passes only when the source D4.3 chain is qualified and fusion-authorized, all ten frames integrate, the voxel accumulator does not hit its capacity bound, fusion reduces the raw point count, and at least the configured minimum number of multi-view-supported voxels remain.

Visual validation should inspect both RGB appearance and per-voxel view support before any later refinement or surface reconstruction step.
