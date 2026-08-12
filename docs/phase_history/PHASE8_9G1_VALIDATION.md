# Phase 8.9G1 — Regression Inventory & Preflight

G1 freezes the final Phase 8 regression plan before any broad release-gate execution. It is qualification-only: no production source, application behavior, reconstruction algorithm, dependency, external backend checkout, release metadata, or tag is changed.

## Purpose

G1 inventories the current automated test surface after 8.9F6 and verifies that the inherited release gates needed by G2/G3 still exist. The preflight deliberately does not execute PyCOLMAP, DA3, Gaussian, Nerfstudio/Nerfacto, or any other heavy reconstruction backend.

The strict workstation baseline for G1 is `phase8-reconstruction` at `0a987bc392bd03a8b56bdc62884a8e624ebd764c`. Only the five G1 validation files may be present as worktree changes while the strict inventory is run.

## G2 — complete current-tree regression

G2 will run, in separate processes where appropriate:

1. `compileall` over application, core, tests, scripts, and `main.py`.
2. Complete `tests/unit`.
3. Complete `tests/ui`.
4. Complete `tests/integration`.
5. Complete `tests/stress`.
6. The inherited Phase 8.8 protocol chain through `run_phase8_8h3_tests.sh protocol`.
7. Every Phase 8.9 A1–F6 `quick` runner, so the release-specific runner contracts themselves are exercised after the broad suites.

G2 will not re-run the expensive real SfM/DA3/Gaussian/Nerfacto campaigns. Those real hardware results are already qualified by 8.9F and are inputs to release evidence, not something the complete regression should repeatedly retrain.

Any G2 failure must be diagnosed. Tests or acceptance criteria must not be weakened simply to obtain a green report.

## G3 — explicit backward compatibility + final regression gate

G3 will not duplicate all four complete pytest suites. It will run targeted inherited release gates for stable pre-Phase-8 product behavior:

- Phase 5 smoke/core-runtime compatibility;
- Phase 6 release/model/benchmark quick gate;
- Phase 7 operational-mode release quick gate;
- Phase 7.5 measurement/inspection release quick gate;
- the inherited Phase 8.8 backward chain;
- a bounded real application startup/shutdown sanity check.

G3 will then consolidate G2 + G3 evidence into the final 8.9G closure report.

## Locked safety rules

- Version remains `0.7.5`; `v0.8.0` must not exist yet.
- Canonical geometry remains authoritative.
- Gaussian and NeRF remain appearance-only.
- `certified_metrology=false` without qualified calibration.
- Heavy/native reconstruction runtimes remain outside the production `depth` environment and outside direct `core/reconstruction` imports.
- No branch switch, destructive Git operation, history rewrite, or release tag is allowed in G1/G2/G3.
- G1 itself starts no heavy training/inference and adds no new regression framework; it only inventories the existing test/release surface and freezes the execution matrix.
