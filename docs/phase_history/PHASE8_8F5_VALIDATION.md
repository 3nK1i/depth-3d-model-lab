# Phase 8.8F5 — Resource / Fallback UX

## Scope

F5 closes the 8.8F appearance UI by making resource state, degraded appearance
evidence, and fail-closed fallback behavior explicit.

F5 consumes the application's existing startup hardware snapshot and the already
persisted F1/F2/F3/F4 appearance state. It does **not** install, import, probe,
or execute an external Gaussian/NeRF runtime and does not render or retrain a
neural representation.

Canonical geometry remains authoritative at all times.

## Application behavior

- Adds one compact `RESOURCE & FALLBACK` group to the existing scrollable
  Reconstruction settings sidebar.
- Reports the current application device and startup CUDA/VRAM snapshot.
- Explicitly states that existing backend-authored static previews do not require
  launching the external neural authoring runtime.
- Existing static previews remain displayable even when the application runtime
  is CPU-only or CUDA is unavailable.
- Missing/unavailable/undecodable selected previews fail closed to canonical
  geometry and expose the reason.
- F5 never silently substitutes Gaussian for NeRF or NeRF for Gaussian.
- Persisted appearance-stage fallback/degradation evidence is surfaced without
  rerunning the reconstruction pipeline.
- F1/F2/F3/F4 authority remains unchanged: appearance is never canonical geometry,
  measurement, inspection/QC, reference, or certified-metrology authority.

## Files

Added:

- `app/reconstruction_appearance_resources.py`
- `docs/phase_history/PHASE8_8F5_VALIDATION.md`
- `scripts/run_phase8_8f5_resource_fallback_preview.py`
- `scripts/run_phase8_8f5_resource_fallback_smoke.py`
- `scripts/run_phase8_8f5_tests.sh`
- `tests/ui/test_phase8_8f5_resource_fallback_ux.py`
- `tests/unit/test_phase8_8f5_contract.py`
- `tests/unit/test_phase8_8f5_reconstruction_appearance_resources.py`

Modified only where F5 ownership makes earlier deferred wording stale or where
F5 needs to observe existing appearance state:

- `app/reconstruction_appearance_presentation.py`
- `app/reconstruction_gaussian_visualization.py`
- `app/reconstruction_nerf_visualization.py`
- `app/reconstruction_representation_switching.py`
- `app/widgets/reconstruction_page.py`
- F1-F4 contract/presentation tests that asserted the previous deferred wording.

## Validation

```bash
./scripts/run_phase8_8f5_tests.sh quick
./scripts/run_phase8_8f5_tests.sh protocol
./scripts/run_phase8_8f5_tests.sh backward
```

### Decisive smoke

```bash
rm -rf /tmp/phase8_8f5_resource_fallback
python scripts/run_phase8_8f5_resource_fallback_smoke.py
cat /tmp/phase8_8f5_resource_fallback/summary.json
```

The smoke proves:

- a qualified static appearance preview is still displayable with a CPU-only app
  resource snapshot;
- a missing selected preview retains canonical geometry;
- an undecodable preview fails closed and reports the presentation failure;
- recovery clears stale fallback state;
- no cross-representation fallback occurs;
- persisted appearance-stage degradation is visible;
- no external neural backend or backend resource probe is executed by F5;
- canonical geometry authority and non-certified-metrology status remain unchanged.

### Interactive preview

```bash
python scripts/run_phase8_8f5_resource_fallback_preview.py
```

Manual checks:

1. Confirm `RESOURCE & FALLBACK` is compact and readable in the existing sidebar.
2. Confirm `Device`/`Resources` identify the application startup hardware snapshot.
3. Confirm `Presentation` says the selected static preview is backend-free.
4. Activate Gaussian and NeRF and confirm F4 switching still works.
5. Hide an appearance view and confirm canonical geometry returns.
6. Confirm `Fallback` always says `Canonical geometry • no neural cross-fallback`.
7. Confirm authority text remains fully readable at 1366×768.

## Phase boundary

F5 is UX/presentation only. It does not turn the production `depth` environment
into a Gaussian or NeRF training environment, and it does not change any
reconstruction authority chain.
