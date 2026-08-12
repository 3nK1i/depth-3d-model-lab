# Phase 8.9E — Endurance / Resource / Recovery Validation

Status: qualification protocol prepared; production-machine bounded campaign pending.

## Scope

Phase 8.9E qualifies the existing reconstruction/application recovery and resource-ownership architecture under repeated operation. It composes the already-qualified Phase 8.7G3 whole-engine endurance, Phase 8.8H2 application resource coordination, Phase 8.8H3 failure/cancellation recovery, Phase 8.9C persistence/reload, and Phase 8.9A/B export paths.

No new recovery engine, persistence format, checkpoint system, backend algorithm, GPU policy, or authority model is introduced by 8.9E. Production code is intentionally unchanged unless the production campaign exposes a concrete defect.

## Bounded production campaign

`scripts/run_phase8_9e_endurance_resource_recovery.py` runs the actual PyQt `MainWindow` at 1366×768 with an isolated Qt settings directory and a lightweight external reconstruction fixture worker. It performs:

- at least 3 repeated external-worker start/cancel/finalize cycles;
- explicit child-process cleanup verification after every cycle;
- application-wide navigation/resource release verification after every cycle;
- a terminal worker failure after the repeated cancellation campaign;
- repeated application-controller persistent-scene save/reload;
- repeated verified product export;
- final application shutdown and controller shutdown verification;
- process RSS and process GPU-memory before/after measurement;
- heavy-runtime module leak inspection.

Default production gate: 5 cycles. RSS/GPU growth limits are deliberately bounded but tolerant of Python/Qt allocator caching; raw measurements are always emitted for review. A threshold failure is evidence to inspect, not permission to weaken the test or rewrite production architecture blindly.

## Backend isolation

8.9E intentionally does **not** start real PyCOLMAP, DA3, Gaussian, NeRF/Nerfstudio, or Stella backends. Their real-scene/reconstruction qualification remains owned by Phase 8.9F. The endurance probe verifies that those heavy runtime modules do not leak into the main application process.

The carry-forward authority model remains unchanged:

- canonical geometry = authoritative geometry;
- DA3 = external dense/relative supporting geometry until correctly canonicalized; never invented metric scale;
- Gaussian = appearance only;
- NeRF/Nerfacto = appearance only;
- certified metrology remains false unless separately qualified.

## Validation gates

1. `./scripts/run_phase8_9e_tests.sh quick`
2. `./scripts/run_phase8_9e_tests.sh affected`
3. bounded production campaign with actual `MainWindow` at 1366×768;
4. every cancellation/failure returns Reconstruction ownership to idle;
5. no orphan child process after any terminal cycle;
6. repeated persistence identity/fingerprint remains stable;
7. repeated export verifies successfully and remains authority-truthful;
8. final shutdown leaves Reconstruction inactive/shut down;
9. heavy runtime leak detection remains false;
10. RSS/GPU-memory growth remains within the declared bounded thresholds;
11. `git diff --check` and exact-scope staging before commit.

Full project regression remains Phase 8.9G. Real DA3/Gaussian/NeRF execution remains Phase 8.9F.
