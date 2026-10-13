# Phase 8.8F2 — Gaussian Splatting Visualization

## Scope

F2 adds Reconstruction UI visualization of an **already-produced backend-authored Gaussian preview render** from the persisted Phase 8.7 Gaussian artifact.

It does not train, render, parse the heavy Gaussian scene payload, probe resources, or switch the whole Reconstruction representation.

Authority remains unchanged:

- Canonical geometry remains geometry authority.
- Gaussian remains appearance-only.
- Measurement / inspection / reference authority is unchanged.
- Certified metrology remains false.

The generic preview overlay is representation-agnostic and can be reused by F3 without object-specific logic.

## Production changes

- `app/reconstruction_gaussian_visualization.py`
  - resolves only artifact-local backend preview renders;
  - honors `qualification_metrics.json` when present;
  - refuses qualification preview paths that escape the Gaussian artifact directory;
  - never imports or executes Gaussian runtimes.
- `app/widgets/reconstruction_appearance_preview.py`
  - generic aspect-preserving static appearance preview overlay.
- `app/widgets/reconstruction_page.py`
  - adds `GAUSSIAN SPLATTING VISUALIZATION` status/actions;
  - `Show preview` displays the already-authored preview image;
  - selecting another F1 representation hides stale Gaussian preview;
  - F4 still owns whole-representation switching.
- `app/theme.py`
  - adds native Depth & 3D Model Lab styling for the appearance preview overlay.

## Validation

### Quick

```bash
./scripts/run_phase8_8f2_tests.sh quick
```

### Protocol / authority

```bash
./scripts/run_phase8_8f2_tests.sh protocol
```

### Backward

```bash
./scripts/run_phase8_8f2_tests.sh backward
```

### Interactive preview

```bash
python scripts/run_phase8_8f2_gaussian_visualization_preview.py
```

Select `Gaussian` in `APPEARANCE REPRESENTATIONS`, press `Show preview`, and confirm:

- a Gaussian backend preview appears in the Reconstruction viewport;
- the viewport says `Gaussian Splatting preview • appearance only`;
- the authority row says canonical geometry remains authoritative;
- `Hide` restores the normal Reconstruction viewport;
- selecting `NeRF` hides a visible Gaussian preview rather than leaving stale imagery.

### Decisive smoke

```bash
rm -rf /tmp/phase8_8f2_gaussian_visualization
python scripts/run_phase8_8f2_gaussian_visualization_smoke.py
cat /tmp/phase8_8f2_gaussian_visualization/summary.json
```

The smoke proves:

- the persisted 8.7F5 Gaussian artifact is the source;
- the exact backend preview artifact is displayed;
- F2 does not execute a Gaussian backend or compute a render;
- F2 does not parse `scene.pt` / Graphdeco point-cloud payloads;
- stale preview is hidden when F1 selection moves to NeRF;
- missing preview is explicit;
- canonical geometry authority and non-certified-metrology status are preserved.

## Deferred ownership

- F3: NeRF visualization.
- F4: whole-representation switching.
- F5: resource probing, fallback policy, and degraded-resource UX.
