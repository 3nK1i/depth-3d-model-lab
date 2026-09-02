# I4.4A — Focused Offline Dense Foundation Regression

## Scope

I4.4A is the first boundary of I4.4 — Offline Dense Foundation Freeze. It does not develop or compare providers. It consumes the formal I4.3H2 closure, re-audits the current offline-dense contract surfaces, and binds a focused automated-regression receipt.

The boundary is intentionally execution-light: no provider backend execution, provider runtime probe, PyCOLMAP/SfM rerun, matched-view rerun or manual cross-provider visual verification is started. I4.4B visual verification, I4.4C final provenance/authority audit and I4.4D final I4 closure remain separate later boundaries.

## Focused foundation coverage

The regression slice covers the existing implementation rather than duplicating it:

- canonical dense-provider family/implementation contracts and import surfaces;
- PyCOLMAP/SfM camera-frame authority and arbitrary-scale truthfulness;
- dense/SfM alignment and projection contracts;
- adaptive view selection and production planning, including `permanent_global_view_cap=None`;
- dense fusion and reconstruction-unit semantics;
- persistent dense RGB scene round-trip surfaces;
- persistent canonical reconstruction-scene compatibility;
- I4.3H2 formal closure/authorization compatibility;
- a small Phase 9 canonical-scene compatibility sanity check.

The affected regression is deliberately narrower than the H1 668-test predecessor sweep. H1 already proved the complete I4.3 predecessor suite; I4.4A instead tests the cross-layer offline-dense foundation most likely to expose freeze-boundary regressions.

## Evidence and authority guards

I4.4A requires a SHA256-bound passing I4.3H2 closure and a SHA256-bound passing I4.4A regression receipt. It fails closed if H2 no longer proves the frozen 5-family / 4-executable-family / 10-provider architecture, the 11 controlled qualification scenarios, or the five preserved terminal quality events.

I4.4A preserves:

- PyCOLMAP as sole authoritative offline camera/scene-frame owner;
- no provider selected, promoted or ranked;
- no universal provider winner or global scalar score;
- Auto unimplemented;
- Hybrid unauthorized;
- no metric-scale, authoritative-measurement or certified-metrology claim;
- no surface-mesh authority;
- `permanent_global_view_cap=None`.

The five existing degraded/rejected terminal quality events remain scenario-specific operating-envelope evidence and are not converted into provider scores.

## Validation

`scripts/run_i4_4a_tests.sh all <receipt.json>` runs:

1. focused I4.4A unit/integration/stress tests;
2. execution-free import and runner-help probes;
3. affected unit regression;
4. affected integration regression;
5. affected stress regression;
6. optional regression-receipt generation after all preceding gates pass.

`scripts/run_i4_4a_focused_regression.py` then SHA256-verifies the frozen I4.3H2 closure and regression receipt and writes `i4_4a_focused_regression.json` to a new empty evidence directory.

## Decision boundary

A passing I4.4A result records `i4_4a_complete=true` and authorizes **I4.4B — Cross-Provider Visual Verification**. It does not start I4.4B, I4.4C or I4.4D.
