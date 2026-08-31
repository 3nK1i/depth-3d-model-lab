# I4.3G5A — Cross-Provider Evidence Synthesis

## Scope

I4.3G5A is evidence-only. It binds the frozen I4.3G1 methodology and the formally closed I4.3G2, I4.3G3C and I4.3G4C qualification families into one cross-provider operating-envelope synthesis. It performs no provider execution, no PyCOLMAP rerun, no new matched-view rendering and no new manual review.

G5A does **not** formally close I4.3G. That remains I4.3G5B.

## Family boundaries

The synthesis preserves three distinct provider families rather than forcing them onto one leaderboard:

- **MVS / DA3** — static multiview dense-reconstruction evidence from I4.3G2;
- **temporal depth** — RollingDepth and Video Depth Anything evidence from I4.3G3;
- **single-image depth** — Metric3D v2, UniDepthV2, MoGe-2, Depth Pro, Depth Anything V2 and Marigold evidence from I4.3G4.

Their input semantics and scenario sets differ. Direct cross-family provider ranking is therefore invalid.

## Evidence policy

G5A requires SHA256-bound G1/G2/G3/G4 closure artifacts. Production-native outcomes remain primary; matched-view/manual evidence remains secondary. Scenario-specific preferences are preserved as diagnostics but are never accumulated into a global preference count, weighted score or universal winner.

The G2 final closure is retained as the authoritative G2 family synthesis. If that closure exposes scenario-level detail, G5A carries it forward without reinterpretation; otherwise G5A records the formally closed family-level synthesis rather than inventing missing G2 fields.

G3/G4 terminal `pass`, `degraded` and `rejected` outcomes are preserved exactly. Terminal-outcome counts are informational operating-envelope summaries only and are explicitly not scores.

## Authority locks

Throughout G5A:

- PyCOLMAP remains the sole global camera/scene-frame authority;
- no provider is selected, promoted or ranked;
- no universal winner or global scalar score is created;
- Auto remains unimplemented;
- Hybrid remains unauthorized;
- no physical metric scale is claimed;
- no measurement/certified-metrology authority is created;
- no surface mesh is generated;
- `permanent_global_view_cap` remains `None`.

## Closure boundary

A passing G5A synthesis records `g5a_cross_provider_synthesis_complete=true`, authorizes I4.3G5B, and deliberately keeps `i4_3g_closed=false`. I4.3G5B is the only next step authorized to perform formal I4.3G closure.
