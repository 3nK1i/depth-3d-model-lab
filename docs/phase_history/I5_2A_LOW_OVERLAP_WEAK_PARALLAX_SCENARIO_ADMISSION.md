# I5.2A — Low-Overlap / Weak-Parallax Scenario Admission

## Purpose

I5.2A starts **I5.2 — Low overlap / weak parallax** without running a new reconstruction. It binds the frozen I5.1C handoff, admits the already-qualified I4.3 weak-overlap history, content-addresses one ordered static multiview source, and freezes two deterministic controlled stress slices for I5.2B.

I5.2A is deliberately a pre-execution boundary. It does not run PyCOLMAP, MVS, DA3, any depth provider, fusion, filtering, Auto, Hybrid, or MP-SfM.

## Historical weak-overlap baseline

The I4.3G2 weak-overlap scenario is admitted only at the level already supported by frozen evidence:

- `g2_weak_overlap_mvs_da3` completed controlled matched-view review;
- partial registration was explicitly preserved;
- the authoritative review scope was the registered-view scope, not a full-scene claim;
- committed G2C phase history records a connected five-view authoritative registered scope;
- G2B required authoritative PyCOLMAP co-visibility connectivity before dense execution;
- connectivity failure was a legitimate scenario outcome and thresholds were not relaxed to erase it.

Provider preferences from the old scenario are not converted into rankings, scores, winners, or production defaults by I5.2A.

## Fresh source binding

The I5.2B source is bound as an exact naturally ordered image set:

- exact directory;
- exact ordered image names;
- per-file size and SHA256;
- aggregate ordered asset-set SHA256;
- explicit engineer confirmation that the source is an ordered static multiview capture.

The directory name is never used as semantic proof.

## Frozen stress plan

I5.2A derives three like-for-like slices from the same source:

1. **control** — the full ordered image set;
2. **low-overlap proxy** — an increased-spacing, full-span subset with at least six images;
3. **weak-parallax proxy** — a centered contiguous narrow-span subset with the same bounded target size.

These are **source-order stress proxies only**. Increased index spacing is not authoritative overlap measurement, and a narrow contiguous source window is not authoritative parallax measurement.

I5.2B must revalidate:

- low-overlap behavior from authoritative PyCOLMAP registration / co-visibility evidence;
- weak-parallax behavior from authoritative PyCOLMAP camera geometry using scale-invariant geometry diagnostics;
- all control and stress slices under the same frozen like-for-like matching strategy (`exhaustive`).

Failure to confirm the intended stress geometry is valid evidence. I5.2B must not tune the slices, matching thresholds, or production defaults after seeing quality results merely to obtain a PASS.

## Authority locks

I5.2A preserves:

- PyCOLMAP as sole global camera / scene-frame authority;
- arbitrary reconstruction scale;
- `permanent_global_view_cap=None`;
- no provider selection, promotion, ranking, winner, or scalar score;
- no Auto policy;
- no Hybrid authorization;
- no metric-scale or metrology claim;
- no surface-mesh authority;
- no production sampling-policy modification.

## Decision

A passing I5.2A artifact means:

```text
i5_1_complete = true
i5_2_started = true
i5_2a_complete = true
i5_2b_real_replay_authorized = true
i5_2_complete = false
i5_3_authorized = false
```

## Next

**I5.2B — Current frozen-foundation low-overlap / weak-parallax replay**
