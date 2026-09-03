# I5.3A — Blur / Exposure / Clutter / Symmetry Scenario Admission

## Purpose

I5.3A starts **I5.3 — Blur / exposure / clutter / symmetry** as an execution-free admission boundary. It binds the frozen I5.2C closure, rebinds the exact ordered Castle-P19 control source by content identity, and freezes deterministic source-derived stress recipes for the later I5.3B real replay.

I5.3A does **not** materialize transformed images and does not run PyCOLMAP, MVS, DA3, any depth provider, fusion, filtering, MP-SfM, Auto, or Hybrid.

## Frozen control source

The I5.3 control is the exact source already bound through I5.2. The aggregate ordered asset-set SHA256 must match the source identity carried by the frozen I5.2C closure. Directory names or image counts alone are not sufficient.

All stress cases preserve the exact same ordered source membership. This keeps I5.3B like-for-like and prevents a dataset change from being mistaken for a robustness effect.

## Frozen stress recipes

I5.3A freezes five result-independent recipes:

1. **Blur proxy** — Gaussian blur, kernel `13`, sigma `3.0`.
2. **Underexposure proxy** — linear intensity gain `0.35`.
3. **Overexposure proxy** — linear intensity gain `1.55` plus bias `32` with uint8 clipping.
4. **Clutter proxy** — a bottom distractor band occupying `18%` of the frame, sourced deterministically from an image seven positions away in the same frozen capture. The band is intentionally inconsistent with the current camera view.
5. **Symmetry proxy** — repeated left/right side bands derived from a central vertical source region, with the right repeated band mirrored.

The recipes are frozen before real replay. I5.3B may not tune them based on reconstruction outcomes.

## Semantic guards

The five stress recipes are controlled **proxies** at I5.3A. Their stress labels are not authoritative until I5.3B materializes them and revalidates the intended image-domain effect.

I5.3B must demonstrate, at minimum:

- blur: lower median Laplacian variance than control;
- underexposure: lower median luma and no artificial improvement in underexposure/clipping burden;
- overexposure: higher median luma and no artificial improvement in overexposure/clipping burden;
- clutter: increased feature burden in the injected distractor region, while preserving that this is a synthetic controlled clutter proxy rather than natural-scene clutter ground truth;
- symmetry: increased repeated-region self-similarity or descriptor ambiguity, while preserving that this is a synthetic controlled symmetry proxy rather than natural-scene symmetry ground truth.

After image-domain revalidation, I5.3B observes the unchanged production-native PyCOLMAP sparse-SfM outcome for each case. A failed semantic revalidation or degraded/rejected SfM result is legitimate evidence; it does not authorize recipe tuning or threshold relaxation.

## Authority and roadmap guard

- PyCOLMAP remains the sole camera/scene-frame authority.
- `permanent_global_view_cap=None` remains unchanged.
- No provider is selected, promoted, ranked, or made a universal winner.
- No metric-scale, measurement-authority, certified-metrology, or surface-mesh claim is introduced.
- MP-SfM remains deferred to I5.10.
- Hybrid remains deferred to I5.12 if justified.
- Auto remains deferred to I5.13.

A passing I5.3A admission authorizes only **I5.3B — current frozen-foundation blur / exposure / clutter / symmetry replay**.
