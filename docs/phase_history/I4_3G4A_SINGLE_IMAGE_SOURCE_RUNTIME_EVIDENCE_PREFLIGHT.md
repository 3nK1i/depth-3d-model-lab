# I4.3G4A — Single-Image Source / Runtime / Evidence Preflight

## Purpose

I4.3G4A is the execution-free entry gate for the single-image qualification family frozen by I4.3G1. It does not rerun PyCOLMAP and does not start Metric3D v2, UniDepthV2, MoGe-2, Depth Pro, Depth Anything V2, or Marigold inference.

The preflight binds the exact three G4 scenarios to the corresponding already-qualified G2 source/SfM contexts:

- `g4_single_image_complete_room` → `g2_complete_room_mvs_da3`
- `g4_single_image_low_texture` → `g2_low_texture_mvs_da3`
- `g4_single_image_weak_overlap` → `g2_weak_overlap_mvs_da3`

This preserves one authoritative PyCOLMAP frame per scenario and avoids introducing unnecessary new SfM variability before cross-provider comparison.

## Predecessor gates

G4A requires:

- the exact passing frozen I4.3G1 qualification framework;
- the formally closed passing I4.3G3C closure with `i4_3g4_authorized=true`;
- complete G3 production/manual-review accounting and preserved quality failures;
- the three matching passing I4.3G2B production-native snapshots;
- byte-valid source manifests, source image sets, and authoritative SfM snapshots from those G2 contexts.

The runner verifies the frozen G1 and G3C SHA256 identities by default. It also re-hashes the current source image directories and refuses source-byte drift.

## Six-provider runtime gate

All six single-image production adapters must be ready on the current workstation without starting inference. Provider-specific probes preserve the already-closed E1–E6 provenance contracts and additionally require CUDA availability for G4B production execution.

G4A deliberately distinguishes **installed-runtime identity/integrity** from **fresh-install reproducibility metadata**. The former is a readiness gate; the latter is recorded truthfully but does not retroactively strengthen the closed E-series contracts. In particular, Depth Pro's requirement-line commit pin, Depth Anything V2's installer/requested checkpoint ref pins, and Marigold's exact Diffusers requirement pin are not required to be true when the installed official runtime/source/checkpoint identity is independently resolved and byte-bound as established by E4/E5/E6. Source-checkout agreement, installed-runtime origin, resolved revisions, local checkpoint readiness, and checkpoint/marker hashes remain mandatory.

The preflight records the complete probe payload for:

1. Metric3D v2
2. UniDepthV2
3. MoGe-2
4. Depth Pro
5. Depth Anything V2
6. Marigold

No provider is selected or promoted by readiness.

## Run-status contract refinement

The frozen G1 v1 run-evidence schema remains unchanged and still records:

`pass`, `degraded`, `failed`, `cancelled`, `unavailable`.

G3B/G3C subsequently established and regression-protected a distinct terminal provider-quality outcome, `rejected`. G4A therefore freezes the effective G4 execution semantics without rewriting historical G1 evidence:

- provider-quality terminal outcomes: `pass`, `degraded`, `rejected`;
- infrastructure/execution terminal outcomes: `failed`, `cancelled`, `unavailable`.

A legitimate quality rejection is evidence, not automatically a software failure. G4B must not weaken thresholds merely to convert a rejection into a pass/degraded result.

## Manual-review contract

G1 requires controlled manual review for all three G4 scenarios. G4A therefore freezes the seven required dimensions for G4C:

- global completeness;
- local shape plausibility;
- edge definition;
- small-detail preservation;
- texture sensitivity;
- cross-view stability after SfM alignment;
- false geometry and floaters.

`not_assessable` remains valid only with an explicit reason. Manual review cannot override a production-native rejection, and independent provider-specific visual auto-fit remains forbidden.

## Storage gate

G4A records free space on the selected evidence/storage filesystem and applies a configurable preflight safety floor. The default runner floor is 32 GiB. This is an execution-safety floor for preserving new evidence; it is not a reconstruction quality threshold and it does not authorize deletion or rewriting of frozen evidence.

## Authority preservation

G4A preserves:

- PyCOLMAP as camera/global scene-frame authority;
- no provider selection or promotion;
- no Auto policy;
- no Hybrid authorization;
- no metric-scale claim;
- no measurement/metrology authority;
- no surface-mesh authority;
- `permanent_global_view_cap=None`.

## Closure condition

G4A closes only when all three G2 source/SfM contexts are byte-bound, all six production runtimes are ready, and the storage gate passes. Only then is I4.3G4B real single-image production-native execution authorized.
