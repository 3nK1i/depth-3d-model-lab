# Phase 9.5E — Interactive NeRF Viewer

Phase 9.5E adds an explicit interactive Nerfacto appearance mode over the persisted 9.5D representation.

## Authority and isolation

- Canonical geometry remains the only geometry, picking and measurement authority.
- NeRF remains `appearance_only`; it cannot claim metric scale, measurement, QC or certified-metrology authority.
- `depth` never imports Nerfstudio, tiny-cuda-nn or the heavy NeRF checkpoint runtime.
- Interactive rendering runs in the pinned external `eia-nerf-nerfstudio` interpreter and process group.
- There is no silent Gaussian↔NeRF fallback.

## Viewer behavior

The existing Reconstruction turntable controls remain the navigation source. During a drag, the NeRF image is hidden so the canonical guide moves immediately. On release, the current canonical-view camera is converted to a presentation-only OpenGL c2w request and rendered by the isolated worker. The returned frame is a mouse-transparent appearance overlay.

The ruler deliberately pauses/hides NeRF appearance and operates on the canonical surface underneath. `Preview` preserves the existing static backend-authored image without starting Nerfstudio.

## Scope

9.5E does not qualify NeRF quality or rendering performance, add user quality controls, or harden long-session GPU lifecycle. Those remain 9.5F–9.5H.
