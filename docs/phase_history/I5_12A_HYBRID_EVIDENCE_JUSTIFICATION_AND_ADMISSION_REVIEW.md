# I5.12A — Hybrid Evidence-Justification and Admission Review

## Purpose

I5.12A decides whether the accumulated frozen evidence actually justifies the optional I5.12 hybrid reconstruction experiment. It is an evidence-only decision boundary. It does not run SfM, MVS, DA3, fusion, matched-view rendering, provider inference, fault injection, or Auto policy work.

## Historical admission rule

I4.3A reserved the `hybrid` family only as a future hypothesis and explicitly required cross-provider evidence showing that **MVS is cleaner while AI depth meaningfully recovers legitimate MVS holes**. I5.12A treats that wording as a scientific admission condition, not a suggestion.

Global completeness alone is not enough. The direct matched-view `legitimate_hole_recovery` dimension must prefer the AI provider while the frozen cleanup dimensions (`false_geometry_and_floaters`, `surface_noise_and_roughness`, and `false_bridging_or_thickening`) show the MVS cleanliness advantage.

## Evidence chain

The formal review binds:

- I5.11D bounded failure/recovery closure, which authorizes only this admission review;
- I5.6C MVS robustness closure, confirming the production COLMAP/PyCOLMAP MVS path only within its admitted envelope;
- I5.2A low-overlap admission, used as the already-frozen binding to the historical I4.3G2 final closure;
- the independently rehashed I4.3G2 final MVS-vs-DA3 closure;
- the committed I4.3A phase-history rule;
- an I5.12A regression receipt.

## Only plausible direct candidate

The only candidate considered is the frozen G2 weak-overlap MVS-vs-DA3 scenario. It is direct same-family static multiview evidence using one PyCOLMAP-authoritative registered scope. Cross-family temporal or single-image preferences are not converted into hybrid evidence because their input semantics and scenario sets differ.

If admitted, the bounded hypothesis is:

> Preserve COLMAP MVS as the primary dense geometry and allow DA3 supplementation only at predeclared legitimate MVS holes, in the same PyCOLMAP frame, with provider-native outputs immutable.

I5.12A itself does **not** authorize provider reruns, SfM reruns, fusion execution, or hybrid execution. A justified result authorizes only I5.12B runtime/integration/evidence preflight.

## Two legitimate outcomes

### `hybrid_justified`

Requires both:

1. MVS is preferred on all frozen cleanup dimensions; and
2. DA3 is preferred specifically on `legitimate_hole_recovery`.

Then I5.12B is authorized, while I5.12C remains unauthorized until B freezes the executable contract.

### `hybrid_not_justified`

If the direct legitimate-hole-recovery condition is absent, I5.12 closes without experiment. A DA3 preference on `global_completeness` alone is deliberately insufficient. I5.13 may then proceed to the first evidence-driven Auto-policy boundary.

## Authority and claim boundaries

Always preserved:

- PyCOLMAP remains sole camera/global-frame authority;
- no provider is selected, promoted, or universally ranked;
- no global scalar score is created;
- no metric scale, measurement authority, certified metrology, or mesh authority is created;
- production defaults and thresholds remain unchanged;
- Auto remains unimplemented;
- `hybrid_authorized=false` and `hybrid_execution_authorized=false` in I5.12A itself.
