# I4.3G3C — Temporal Matched-View Manual Review & G3 Closure

## Scope

I4.3G3C consumes the three frozen I4.3G3B production-native scenarios and closes temporal-provider qualification. It does **not** rerun PyCOLMAP, RollingDepth, or Video Depth Anything.

G3C remains backward-compatible with clean G3B snapshots produced before the terminal-provider-rejection correction: when `provider_terminal_outcomes_complete` is absent, G3C accepts only `production_native_complete` snapshots whose independently validated provider evidence has non-rejection terminal statuses. Explicit `false`, or any rejection-bearing snapshot without the post-correction terminal-outcome assertion, fails closed.

The G3B production outcome remains primary evidence. The G3C matched-view diagnostic is secondary and cannot erase a `degraded` or `rejected` production result.

## Frozen scenarios

- `g3_temporal_static_sweep`
- `g3_temporal_motion`
- `g3_temporal_low_texture`

Providers remain candidates:

- `rolling_depth`
- `video_depth_anything`

## Matched temporal review contract

For each scenario G3C:

1. verifies the G1 manual-review/matched-view authorization;
2. verifies the exact G3B production snapshot and provider evidence hashes;
3. requires both providers to share the same source identity, authoritative PyCOLMAP SfM identity, ordered keyframe ids, source sequence indices, and timestamps;
4. reads the already-produced provider-native temporal depth stacks;
5. selects one deterministic centered contiguous provider-output window (32 views by default);
6. uses the same authoritative SfM workspace RGB frames for both blind panels;
7. applies RollingDepth's already-qualified native-to-canonical polarity multiplier exactly once;
8. computes one fixed sequence-global display normalization per provider over the full frozen sequence;
9. forbids per-frame display auto-fit and forbids interpreting A/B color values as a cross-provider metric scale;
10. creates a blinded RGB/A/B temporal review video and contact sheet.

The six G1 temporal manual-review dimensions remain unchanged:

- temporal flicker
- temporal drift
- static geometry consistency
- motion boundary stability
- transient artifacts
- textureless-region stability

Allowed blinded ratings are `A`, `B`, `tie`, and `not_assessable`.

## Blinding and reveal

The prepare step writes only a commitment to the A/B assignment. The mapping is revealed only after all six ratings are complete and `reviewer_confirmation=true`.

The controlled reveal maps A/B preferences back to providers while preserving the frozen production status. A production `rejected` result cannot be overridden by manual preference.

## Final G3 closure

The final closure requires controlled reveals for all three temporal scenarios. It records scenario-specific production statuses and manual preferences, declares no universal provider winner, performs no provider promotion/selection, and authorizes I4.3G4.

## Authority locks

Throughout G3C:

- PyCOLMAP remains authoritative global camera/scene-frame owner.
- No provider is selected or promoted.
- Auto remains unimplemented.
- Hybrid remains unauthorized.
- No physical metric scale is claimed.
- No measurement or certified-metrology authority is created.
- No surface mesh is generated.
