# Phase 8.8F3 — NeRF Visualization

## Scope

F3 adds Reconstruction UI visualization of an **already-produced backend-authored NeRF render** from the persisted Phase 8.7 NeRF artifact.

It does not train, render, parse NeRF checkpoints/configuration for rendering, probe resources, or implement whole-representation switching.

Authority remains unchanged:

- Canonical geometry remains geometry authority.
- NeRF remains appearance-only.
- Measurement / inspection / reference authority is unchanged.
- Certified metrology remains false.

F3 reuses the generic F2 appearance-preview overlay rather than introducing a second viewport implementation.

## Production changes

- `app/reconstruction_nerf_visualization.py`
  - resolves only existing artifact-authored NeRF images;
  - accepts the explicit E2 runtime-evidence preview when present;
  - accepts an existing E3 held-out rendered view when a held-out render manifest is present;
  - falls back to the conventional artifact-local `preview.png`;
  - rejects preview paths that escape the NeRF artifact directory;
  - never imports or executes a NeRF runtime.
- `app/widgets/reconstruction_page.py`
  - adds `NERF VISUALIZATION` status/actions;
  - `Show preview` displays the already-authored NeRF image in the reusable appearance overlay;
  - selecting another F1 representation hides stale NeRF preview;
  - Gaussian and NeRF preview ownership is mutually safe on the shared overlay;
  - F4 still owns whole-representation switching.
- `app/reconstruction_appearance_presentation.py`
  - updates the F1 metadata state from deferred F2/F3 visualization wording to the now-current `visualization available in F2/F3` wording.

## Validation

### Quick

```bash
./scripts/run_phase8_8f3_tests.sh quick
```

### Protocol / authority

```bash
./scripts/run_phase8_8f3_tests.sh protocol
```

### Backward

```bash
./scripts/run_phase8_8f3_tests.sh backward
```

### Interactive preview

```bash
python scripts/run_phase8_8f3_nerf_visualization_preview.py
```

Select `NeRF` in `APPEARANCE REPRESENTATIONS`, press `Show preview`, and confirm:

- an existing backend-authored NeRF image appears inside the Reconstruction viewport;
- the viewport says `NeRF preview • appearance only`;
- the authority row says canonical geometry remains authoritative;
- `Hide` restores the normal Reconstruction viewport and the F3 ready state;
- selecting `Gaussian` hides a visible NeRF preview rather than leaving stale imagery;
- returning to Gaussian preserves the completed F2 preview behavior.

### Decisive smoke

```bash
rm -rf /tmp/phase8_8f3_nerf_visualization
python scripts/run_phase8_8f3_nerf_visualization_smoke.py
cat /tmp/phase8_8f3_nerf_visualization/summary.json
```

The smoke proves:

- the persisted 8.7F5 NeRF artifact is the source;
- the exact existing NeRF preview image is displayed;
- F3 does not execute an external NeRF backend or compute a render;
- F3 does not parse a NeRF checkpoint for rendering;
- stale preview is hidden when F1 selection moves to Gaussian;
- missing preview is explicit;
- canonical geometry authority and non-certified-metrology status are preserved.

## Deferred ownership

- F4: whole-representation switching.
- F5: resource probing, fallback policy, and degraded-resource UX.
