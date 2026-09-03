# I5.2C — Low-Overlap / Weak-Parallax Robustness Qualification + Closure

## Purpose

I5.2C is the evidence-only closure boundary for **I5.2 — Low overlap / weak parallax**. It consumes the frozen I5.2A scenario admission, the evidence-complete I5.2B real Castle-P19 replay, and a narrow closure regression receipt.

It does **not** rerun PyCOLMAP, execute a dense provider, retune the admitted subsets, relax thresholds, change production defaults, start MP-SfM, authorize Hybrid, or implement Auto.

## Qualified evidence

The frozen I5.2B replay established three like-for-like production-native PyCOLMAP cases using exhaustive matching:

- full-source control: 19/19 registered;
- admitted low-overlap stress: 6/7 registered, with partial registration preserved;
- admitted weak-parallax stress: 7/7 registered.

The low-overlap label is authoritative because median adjacent sparse co-visible landmark count decreases relative to control. The weak-parallax label is authoritative because both the scale-invariant maximum-supporting-pair parallax median and baseline/range median decrease relative to control.

I5.2C preserves these as directional sparse-SfM robustness observations. It does not convert them into absolute metric geometry or a universal robustness threshold.

## Generalization boundary

A `PASS` closure means the current frozen PyCOLMAP sparse-SfM foundation remained operational on the **specific admitted Castle-P19 control/low-overlap/weak-parallax cases** and the intended stress directions were independently revalidated from reconstruction geometry.

It does **not** claim:

- universal low-overlap robustness;
- universal weak-parallax robustness;
- dense-reconstruction robustness;
- provider robustness or provider ranking;
- metric scale or certified metrology;
- MP-SfM qualification.

Partial registration remains partial registration. In particular, 6/7 low-overlap registration is not rewritten as full-input coverage.

## Evidence semantics

I5.2C can close structurally valid `PASS`, `DEGRADED`, or `REJECTED` I5.2B quality evidence without retry-until-PASS behavior. The current frozen replay is `PASS`, but the closure contract intentionally preserves legitimate non-PASS evidence if a future admitted replay reaches I5.2C with that result.

The closure independently SHA-binds I5.2A and I5.2B and verifies that I5.2B's predecessor binding points to the exact I5.2A artifact supplied to the closure.

## Authority and guardrails

I5.2C preserves the frozen architecture:

- PyCOLMAP remains the sole global camera / scene-frame authority;
- each SfM scenario remains arbitrary-scale;
- `permanent_global_view_cap=None` remains unchanged;
- no threshold relaxation;
- no production-default modification;
- no provider selection, promotion, ranking, winner, or scalar score;
- no dense-provider execution;
- no MP-SfM before I5.10;
- no Hybrid before I5.12 and only if justified;
- no Auto before I5.13;
- no metric-scale, authoritative-measurement, certified-metrology, or surface-mesh authority.

## Closure decision

A passing I5.2C evidence record means:

```text
i5_2a_complete = true
i5_2b_complete = true
i5_2c_complete = true
i5_2_complete = true
i5_2_quality_status = <pass|degraded|rejected>
i5_3_authorized = true
i5_3_started = false
```

## Next

**I5.3 — Blur / exposure / clutter / symmetry**
