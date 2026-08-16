# Phase 9.6E4 — Professional Unified Workspace UX Consolidation

## Scope

Phase 9.6E4 is presentation-only. It does not remove or duplicate qualified Phase 8/9 controls and does not change reconstruction, persistence, measurement, QC, scale, Gaussian, DA3, or NeRF authority.

The Reconstruction settings sidebar now promotes the unified Digital Twin context to the first visible stack:

1. Digital Twin Workspace
2. Build & Processing
3. Canonical Scene
4. Ruler Metric Estimate

The Digital Twin Workspace card keeps Workspace, Active view, Composition, Scale and Authority visible at a glance. Navigation, Interaction, Resources and Safety remain live but are nested behind a small **Show details** disclosure to reduce density.

Existing Workflow, Session & Persistence, inspection, appearance, calibration, advanced reconstruction and diagnostic disclosure groups remain available below with their established object identities, actions and default expanded/collapsed state.

A newly observed E1 heavyweight job expands and scrolls the existing **Build & Processing** group into view once. Subsequent status refreshes for that same job do not override the user's disclosure choice. A different new job may surface the group once again.

## Authority / compatibility

- Canonical geometry remains the only spatial, picking and measurement authority.
- DA3 remains supporting relative geometry only.
- Gaussian and NeRF remain appearance only.
- Metric scale and certified metrology are not invented or promoted.
- E1 remains the only heavyweight build-job lifecycle supervisor.
- No backend command, persistence schema, scene artifact or measurement logic is changed.
- Existing Phase 8 disclosure defaults are preserved.
- Horizontal scrolling remains disabled at the qualified 1366×768 workspace size.
