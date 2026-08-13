# Phase 8.9 pre-H final post-fix whole-app requalification

## Purpose

This gate requalifies the final post-RTSP/post-UI application tree before Phase 8.9H release preparation. It is qualification-only: no production application file is changed by this substep and no release tag is created.

Baseline: `phase8-reconstruction` at `d8882391df8f3d976223a08893f66fd6d2ba14f9`.

## Frozen scope

The gate intentionally replays:

- the complete G2 current-tree regression matrix using the existing G2 runner;
- the G3 backward-compatibility/final-regression matrix using the existing G3 runner;
- the bounded post-fix RTSP affected gate;
- the combined UI-2/UI-3/UI-4 affected gate.

The historical G2/G3 runners embed the HEAD values that were correct when those gates were originally created. They are therefore called with their package-test `--non-strict-git` option. This final wrapper independently enforces the **current** branch, current HEAD/origin, exact overlay, clean `git diff --check`, version `0.7.5`, and absence of `v0.8.0`.

## Heavy campaign policy

Do not replay expensive F2/F3/F4/F5 real reconstruction/training campaigns. Their authority contracts were not modified by the RTSP/UI changes. The existing G2 quick-chain contract gates are replayed, but no real SfM + DA3 campaign, Gaussian optimization, Nerfacto training, or persistent real-scene rebuild is started.

## Pass criteria

A PASS requires all four groups to finish with return code 0 and repository postflight to remain identical to preflight:

1. G2 complete current-tree regression.
2. G3 backward compatibility.
3. RTSP post-fix affected qualification.
4. UI-4 combined affected qualification.

On failure, stop and diagnose. Do not weaken a test or mutate production code merely to obtain green.

If PASS, the next step is Phase 8.9H documentation/release preparation. The application remains `v0.7.5`; `v0.8.0` is still forbidden until Phase 8.9I final release approval.
