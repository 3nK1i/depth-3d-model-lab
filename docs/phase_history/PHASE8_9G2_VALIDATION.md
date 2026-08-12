# Phase 8.9G2 — Complete Current-Tree Regression

G2 executes the broad regression matrix frozen by 8.9G1. It is qualification-only: no production source, application behavior, reconstruction algorithm, dependency, external backend checkout, release metadata, or tag is changed.

## Strict baseline

The workstation must be on `phase8-reconstruction` at commit `c8ef0fcbc9ab1cb25f4f7606b8aca71596bba85f`, synchronized with `origin/phase8-reconstruction`, with only the five G2 validation files present as worktree changes.

Version remains `0.7.5`; `v0.8.0` must not exist yet.

## Frozen regression matrix

G2 runs these seven groups and stops at the first failure:

1. `compileall` over `app`, `core`, `tests`, `scripts`, and `main.py`.
2. Complete `tests/unit` in its own process.
3. Complete `tests/ui` in its own process.
4. Complete `tests/integration` in its own process.
5. Complete `tests/stress` in its own process.
6. The inherited Phase 8.8 protocol chain: `scripts/run_phase8_8h3_tests.sh protocol`.
7. Every Phase 8.9 A1–F6 `quick` runner, individually and in order.

The runner records a per-group log plus `/tmp/phase8_9g2_full_regression/summary.json`. The four primary pytest suite counts are measured from the actual workstation run; they are not fabricated or hard-coded as acceptance numbers.

## Deliberate exclusions

G2 does **not** rerun the expensive real 8.9F campaigns. It does not start real PyCOLMAP, DA3, gsplat optimization, or Nerfstudio/Nerfacto training. Those real-scene/hardware results were already qualified in 8.9F1–F6 and remain release evidence.

G2 also does not duplicate the explicit backward-compatibility release gates. Phase 5/6/7/7.5, the inherited 8.8 backward chain, and bounded application startup/shutdown are owned by 8.9G3.

## Failure policy

Any failing G2 group is a release blocker until diagnosed. The runner stops at the first failing group so the failure remains local and actionable. Tests, thresholds, authority rules, or isolation requirements must never be weakened simply to obtain a green report.

## Locked authority/isolation rules

- Canonical geometry remains authoritative.
- Gaussian and NeRF remain appearance-only.
- `certified_metrology=false` without separately qualified calibration.
- Heavy/native reconstruction runtimes remain outside production `depth` and outside direct `core/reconstruction` imports.
- No branch switch, destructive Git operation, history rewrite, or release tag is allowed.
