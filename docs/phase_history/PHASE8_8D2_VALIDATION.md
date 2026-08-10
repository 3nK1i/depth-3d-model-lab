# Phase 8.8D2 Validation — Offline Camera Structure / Inspection

## Scope

D2 is a presentation-only extension of the D1 bounded offline sparse-SfM viewer.
It adds registered-camera structure inspection without changing the authoritative
PyCOLMAP result, scene authority, metric scale, or metrology status.

## User-facing behavior

- The completed offline SfM result exposes an **OFFLINE INSPECTION** group.
- A registered-camera selector highlights one camera with a distinct yellow frustum/center.
- Previous/Next navigation steps through the bounded D1 camera samples.
- Focus centers the viewer on the selected registered camera.
- Clear removes only the inspection selection.
- A subtle camera-center path helps interpret the registered camera structure.
- Position and previous-camera displacement are labeled explicitly in **arbitrary gauge**.
- The panel reports bounded UI sample counts separately from authoritative full counts.

## Safety / authority invariants

- `Scene = Partial` remains unchanged for sparse SfM.
- `scale_status = arbitrary` remains unchanged.
- D2 never labels gauge coordinates as metres.
- `canonical_geometry_authoritative = false` remains unchanged.
- `certified_metrology = false` remains unchanged.
- PyCOLMAP remains an isolated external process.
- D2 does not modify the D1 worker artifact or the Stella live protocol/worker.
- Starting a new reconstruction clears stale camera inspection selection.

## Validation gates

1. `./scripts/run_phase8_8d2_tests.sh quick`
2. `./scripts/run_phase8_8d2_tests.sh protocol`
3. `./scripts/run_phase8_8d2_tests.sh backward`
4. `python scripts/run_phase8_8d2_offline_inspection_preview.py`
5. Real South Building smoke via `run_phase8_8d2_offline_inspection_smoke.py`
6. Normal `python main.py` sanity.
7. `git diff --check` clean before commit.
