# Phase 8.8F4 — Representation Switching

## Scope

F4 coordinates the already-qualified F2 Gaussian and F3 NeRF appearance previews
into one explicit, user-driven Reconstruction viewport switching workflow.

It does **not** train or render either neural representation, launch isolated
workers, parse heavy scene/checkpoint payloads, probe GPU resources, select a
resource fallback, or change geometry/measurement/QC/reference authority.

Canonical geometry remains authoritative in every viewport mode.

## Application behavior

- The existing `APPEARANCE REPRESENTATIONS` selector remains the single generic
  representation selector; F4 does not add a duplicate SaaS-style view control.
- `currentIndexChanged` continues to update metadata and clear stale imagery.
- `QComboBox.activated` is the explicit user action that requests the selected
  qualified appearance view. Programmatic restoration does not switch the view.
- Re-activating the already-selected Gaussian/NeRF item can enter its appearance
  view from canonical geometry.
- Switching Gaussian → NeRF or NeRF → Gaussian replaces the shared preview
  atomically through the existing F2/F3 display paths.
- `Hide` returns to canonical geometry without changing the selected metadata.
- If the selected persisted preview is unavailable, F4 retains canonical geometry
  and does not silently choose the other representation. F5 owns resource/fallback UX.
- The new `Viewport` row reports `Canonical geometry`, `Gaussian appearance`, or
  `NeRF appearance` while the existing authority text remains unchanged.

## Files

Added:

- `app/reconstruction_representation_switching.py`
- `scripts/run_phase8_8f4_representation_switching_preview.py`
- `scripts/run_phase8_8f4_representation_switching_smoke.py`
- `scripts/run_phase8_8f4_tests.sh`
- `tests/ui/test_phase8_8f4_representation_switching.py`
- `tests/unit/test_phase8_8f4_contract.py`
- `tests/unit/test_phase8_8f4_reconstruction_representation_switching.py`

Modified only where F4 ownership makes prior deferred wording stale:

- `app/reconstruction_appearance_presentation.py`
- `app/reconstruction_gaussian_visualization.py`
- `app/reconstruction_nerf_visualization.py`
- `app/widgets/reconstruction_page.py`
- F1/F2/F3 contract/presentation tests that asserted the previous deferred F4 wording.

## Validation

```bash
./scripts/run_phase8_8f4_tests.sh quick
./scripts/run_phase8_8f4_tests.sh protocol
./scripts/run_phase8_8f4_tests.sh backward
```

### Decisive smoke

```bash
rm -rf /tmp/phase8_8f4_representation_switching
python scripts/run_phase8_8f4_representation_switching_smoke.py
cat /tmp/phase8_8f4_representation_switching/summary.json
```

The smoke proves:

- initial view remains canonical;
- user activation can show the currently selected Gaussian representation;
- Gaussian ↔ NeRF switching replaces the shared appearance view correctly;
- Hide returns to canonical geometry without changing metadata selection;
- unavailable selected preview retains canonical geometry with no cross-representation fallback;
- no Gaussian/NeRF backend, neural renderer, resource probe, or F5 fallback policy is executed;
- canonical geometry authority and non-certified-metrology status remain unchanged.

### Interactive preview

```bash
python scripts/run_phase8_8f4_representation_switching_preview.py
```

Manual checks:

1. Activate `Gaussian` in the existing representation selector and confirm the Gaussian preview appears.
2. Activate `NeRF` and confirm the same viewport switches directly to the NeRF preview.
3. Activate `Gaussian` again and confirm no stale NeRF image remains.
4. Press `Hide` on the active representation and confirm the normal Reconstruction viewport returns.
5. Confirm the `Viewport` row follows the active view and authority text remains fully readable.
6. Confirm no new resource/fallback controls are present; those remain F5.

## Deferred ownership

- F5: resource probing, fallback policy, and degraded-resource UX.
