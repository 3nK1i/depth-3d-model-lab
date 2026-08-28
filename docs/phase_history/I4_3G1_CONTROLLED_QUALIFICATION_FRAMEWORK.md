# I4.3G1 — Controlled Cross-Provider Qualification Framework

## Purpose

G1 freezes the methodology for I4.3G before any new comparative provider result is consumed. It defines the scenario matrix, common metric vocabulary, run-evidence schema, fairness rules and fail-closed authority gates used by G2–G5.

G1 performs no provider execution and produces no ranking. It does not implement Auto, select/promote a provider, authorize hybrid fusion, claim metric scale, authorize measurement/metrology, generate a mesh, or modify the authoritative PyCOLMAP reconstruction.

## Comparison modes

Two controlled comparison modes are frozen for new executions:

- **Production-native** — primary operating-envelope evidence. Each provider uses its own I4.3F adaptive plan generated from the same source/SfM/resource state. Provider-specific planning differences are preserved rather than hidden.
- **Matched-view diagnostic** — secondary causal diagnostic when technically valid. Providers consume a common frozen view subset to separate provider behavior from adaptive-selection effects. Temporal providers must use a contiguous source-ordered subsequence.

Historical House A/B evidence is a third read-only context mode. It must never be represented as a new G2 execution.

## Scenario matrix

### I4.3G2 — MVS vs DA3

1. complete-room real production comparison — mandatory primary scenario;
2. TempleRing detail-preservation transfer;
3. frozen House MVS-vs-DA3 positive-control context;
4. low-texture static comparison;
5. weak-overlap static comparison.

### I4.3G3 — temporal providers

1. ordered predominantly-static camera sweep;
2. ordered motion/dynamics sequence;
3. ordered low-texture sequence.

Both RollingDepth and Video Depth Anything preserve source order and their native sequence semantics.

### I4.3G4 — single-image providers

The same complete-room, low-texture and weak-overlap operating envelopes are exercised for Metric3D v2, UniDepthV2, MoGe-2, Depth Pro, Depth Anything V2 and Marigold after each provider's already-authorized alignment into the PyCOLMAP SfM gauge.

## Frozen evidence rules

Within a static scenario, providers must share the same source asset identity and authoritative SfM evidence. Every run persists its provider/checkpoint identity, effective profile, parameters, adaptive-plan identity, selected views, output hashes, wall time and measured resource scope.

Mandatory authority gates require the source identity and PyCOLMAP frame to remain unchanged. No metric-scale, measurement, metrology, mesh, Auto, hybrid or provider-promotion claim may appear in a G2–G4 run record.

Infrastructure failures may be rerun only with attempt provenance preserved. A quality failure cannot be silently erased by untracked parameter retuning.

## Metrics and manual review

G1 freezes a multi-axis metric registry rather than a universal weighted score. It includes runtime/resources, view/alignment coverage, common-view consistency, completeness/support proxies and artifact proxies. Optional metrics require an explicit missing-data reason.

Controlled visual review uses the same camera/framing and forbids independent provider auto-fit. Static comparisons review completeness, legitimate hole recovery, false geometry/floaters, noise/roughness, thin structures, edges, small detail and false bridging/thickening. Temporal review adds flicker, drift, static consistency, motion boundaries and transient artifacts.

## Decision policy

There is **no global scalar leaderboard** and no assumed universal winner. G5 may synthesize only scenario-specific operating envelopes from frozen G2–G4 evidence. Hybrid experimentation remains unauthorized unless the collected evidence later justifies a separate explicit experiment. The first Auto policy remains later work.

## Next

A passing G1 binding closes the framework and authorizes **I4.3G2 — controlled MVS vs DA3 production comparison**.
