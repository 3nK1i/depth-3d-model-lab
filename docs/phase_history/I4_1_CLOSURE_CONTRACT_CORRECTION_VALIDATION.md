# I4.1 Closure-Contract Correction Validation

## Purpose

Correct the I4.1 consumer of the persisted I3.2 closure record. I3.2 stores
phase authorization under the nested `decision` object and authority boundaries
under `authority`; the initial I4.1 runner incorrectly looked for authorization
fields at the JSON root.

This overlay changes only the I3→I4 preflight contract and its validation
coverage. It does not change SfM, DA3, dense alignment, fusion, filtering,
persistence, orbit thresholds, metric authority, measurement authority, or mesh
authorization.

## Correct contract

I4.1 now requires a passing I3 closure with:

- `phase == "I3"`, `status == "pass"`, `pass == true`
- all three final I3 checks equal to `pass`
- promoted canonical geometry and persistence authority
- no metric-scale, measurement, or mesh authority
- `decision.i3_closure_authorized == true`
- `decision.i4_connector_orbit_authorized == true`
- `decision.mesh_authorization == false`

Flattened root-level authorization fields are intentionally rejected.

## Regression strategy update

The original I4.1 package had focused/quick/affected suites plus a real-scene
campaign, but no explicit cross-phase integration or contract stress modes. The
closure bug demonstrated that unit coverage of orbit mathematics was not enough
to protect the producer/consumer JSON boundary.

I4.1 now has five explicit gates:

1. `focused` — I4 unit coverage plus the exact I3→I4 producer/consumer contract
2. `quick` — quick historical regression plus the cross-phase contract
3. `integration` — exact I3.2 closure producer to I4.1 consumer tests
4. `stress` — repeated valid closures, mutation matrix, malformed input, and
   authorization-bypass attempts
5. `affected` — prior affected suite plus the new integration and stress tests

This policy should be retained for I4+ cross-phase boundaries. I5 remains the
dedicated full performance/robustness campaign; per-step stress tests are a
separate fast fail-closed regression layer and do not replace I5.

## Expected local counts

- focused: 19
- quick: 274
- integration: 5
- stress: 5
- affected: 341 on the complete `depth` environment, based on the previously
  verified 331 I4.1 affected tests plus the 10 new integration/stress tests.

## Authority

- SfM scale remains arbitrary.
- Metric scale is not claimed.
- Measurement authority is not promoted.
- Surface mesh generation/authorization remains false.
