# Phase 9.6D — Unified Inspection / Measurement / Semantic Interaction

Phase 9.6D keeps every established authority boundary intact while allowing the
existing semantic-inspection and canonical-ruler workflows to coexist safely
with interactive neural appearance.

## Scope

- Add a dependency-light Digital Twin interaction policy/presentation.
- Keep interactive Gaussian co-rendered with canonical semantic/measurement overlays.
- Keep interactive NeRF mouse-transparent and transiently dim/freeze its rendered image
  during canonical ruler use; the user opacity setting and persisted NeRF are unchanged.
- Dim interactive NeRF while a semantic object is selected so canonical membership,
  measurement and QC overlays remain legible.
- Refresh interactive NeRF after semantic Focus from the canonical viewport camera.
- Treat static Gaussian/NeRF preview images as unsynchronised presentation artifacts:
  ruler/semantic inspection fail closed to the canonical viewport instead of picking the image.
- Add one compact `Interaction` row to the existing Digital Twin workspace summary.

## Authority invariants

- Canonical geometry is the only picking and measurement authority.
- Semantic boxes/membership remain context and never create metric scale or QC decisions.
- Gaussian and NeRF remain appearance only.
- Static previews are never picking surfaces.
- Existing uncertainty, rejection, arbitrary-scale behavior, calibration compatibility and
  `certified_metrology=false` remain unchanged.
- No `core/reconstruction/*`, persistence schema, controller, VisPy picking implementation,
  measurement algorithm, Gaussian runtime or NeRF worker protocol is changed.

## Automated gates

- `./scripts/run_phase9_6d_tests.sh quick` — 13 focused tests.
- `./scripts/run_phase9_6d_tests.sh integration` — 2 composition/interaction integration tests.
- `./scripts/run_phase9_6d_tests.sh backward` — 82 affected backward-compatibility tests in
  the production `depth` environment, including the 9.3I scene-bind, 9.3D/E semantic/inspection,
  pre-9.4E ruler, 9.4 Gaussian and 9.5 NeRF contracts.

Manual validation is limited to semantic selection/focus and one ruler interaction under
interactive Gaussian/NeRF, plus a static-preview fail-closed sanity check.
