# Phase 8.8H3 — Failure / Cancellation / Recovery Integration

H3 validates the existing G4 recovery UX against real application lifecycle boundaries: offline failure/cancellation, retained scene/checkpoint evidence, Prepare Resume, fresh retry separation, shutdown, and post-terminal resource release.

One concrete integration defect is corrected in `core/reconstruction/live_runtime.py`: a terminal live backend failure previously published `FAILED` before backend/shared-memory cleanup completed. Because `FAILED` is non-active, application resource guards could release ownership while teardown was still running. H3 now publishes `STOPPING` during terminal teardown and emits `FAILED` only after cleanup is complete.

H3 adds no recovery engine, persistence format, backend algorithm, authority model, device-specific VRAM threshold, or Gaussian/NeRF runtime. Existing bounded live worker recovery remains authoritative; canonical geometry/inspection authority and `certified_metrology=False` remain unchanged.

Validation: focused H3 tests, inherited H2 quick/protocol/backward gates, dedicated H3 smoke, actual app sanity, and exact Git/source inspection. Full Phase 8.8 regression remains H4.
