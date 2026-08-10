# Phase 8.7D3 Final Validation — Gaussian Real-Scene Closure

This delivery closes **8.7D3 Real-scene validation + benchmarking**. It does not introduce a new Gaussian algorithm, backend, production dependency, UI feature, or reconstruction-engine recovery mechanism.

## Closure scope

D3 finalization consolidates and revalidates:

1. D2 real Graphdeco + gsplat qualification and pinned external-runtime provenance;
2. D3 Step 2 held-out PSNR / SSIM / LPIPS evidence;
3. D3 Step 3 training/render VRAM, timing, FPS and measured hardware evidence;
4. D3 Step 4 cancel/timeout/crash/partial-artifact cleanup + real backend recovery;
5. production isolation, canonical-authority and UI-disabled safeguards;
6. backward compatibility and full automated regression.

Gaussian remains an **appearance-only** representation:

- `canonical_geometry_authoritative=False`
- `certified_metrology=False`
- whole reconstruction-engine recovery remains Phase 8.7G
- full application RTX 2060 / 4090 / 5090 production profiling remains Phase 8.9D

## Delivery behavior

`scripts/run_gaussian_d3_finalization.py` runs fresh real closure evidence using the already-qualified isolated environments. It then writes:

```text
/tmp/phase8_7d3_finalization/d3_final_summary.json
/tmp/phase8_7d3_finalization/PHASE8_7D3_FINAL_REPORT.md
```

The finalizer fails closed if:

- either real backend is no longer qualified;
- either backend is missing from quality/performance evidence;
- PSNR/SSIM/LPIPS are incomplete;
- detailed training/render performance or hardware evidence is incomplete;
- the measured hardware profile is mislabeled;
- any Step 4 scenario lacks process cleanup or successful real recovery;
- partial artifacts are not detected and cleaned;
- a forbidden heavy backend import appears in `core/reconstruction`;
- a Gaussian heavy dependency is added to `pyproject.toml`;
- Reconstruction UI is enabled;
- Gaussian becomes canonical geometry or certified metrology.

## Dependency-light gates

```bash
./scripts/run_phase8_7d3_tests.sh quick
./scripts/run_phase8_7d3_tests.sh integration
./scripts/run_phase8_7d3_tests.sh stress
./scripts/run_phase8_7d2_tests.sh quick
./scripts/run_phase8_7d1_tests.sh quick
```

## Fresh real D3 closure — RTX 2060

```bash
rm -rf /tmp/phase8_7d3_finalization

python scripts/run_gaussian_d3_finalization.py \
  --hardware-profile rtx_2060_6gb
```

Required ending:

```text
Phase 8.7D3 Finalization: PASS
D3_CLOSED=True next_phase=8.7E1
canonical_geometry_authoritative=False certified_metrology=False
gaussian_scene_role=appearance_only
engine_level_recovery_deferred_to_phase8_7g=True
```

The generated comparison values are bounded deterministic D3 smoke evidence, not production-scale backend ranking claims.

## RTX 5090 rule

The RTX 5090 readiness contract remains available, but this RTX 2060 finalization must not claim real RTX 5090 measurements. A real RTX 5090 D3 performance rerun can be executed later on the actual target hardware. Whole-application multi-GPU production profiling remains Phase 8.9D.

## Full closure regression

After the fresh real finalizer passes:

```bash
./scripts/run_phase8_7d3_tests.sh full
python main.py
```

Only the established optional perception integration skips are acceptable. `main.py` must start/exit normally and Reconstruction UI must remain disabled.

## Git gate

Do not commit until:

- focused D1/D2/D3 gates pass;
- fresh real D3 finalization passes;
- full unit/UI/integration/stress regression passes;
- `python main.py` sanity passes;
- `git diff --check` is clean;
- no cache/build artifacts are staged;
- branch remains `phase8-reconstruction`;
- baseline before this delivery is `62b49144f69543e9675781b2a69eb2b31a693d4a`.

No tag is created for D3 finalization. After commit/push synchronization, the next locked roadmap item is **8.7E1 — NeRF contracts + external isolation**.
