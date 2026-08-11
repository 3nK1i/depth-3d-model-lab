# Phase 8.8H2 — Application Resource Coordination

H2 closes the reciprocal application admission boundary around the existing Phase 8.8A5 Reconstruction resource guards. Reconstruction already refused to start over Benchmark, Snapshot, Stable Video, Measure & Inspect, or inspection export; H2 makes those application jobs also refuse to start while Reconstruction is active instead of relying only on navigation locking.

The production change is intentionally confined to `app/main_window.py`. It adds no resource manager, scheduler, GPU-name branch, VRAM threshold, backend runtime, reconstruction algorithm, persistence format, checkpoint format, or recovery engine.

Resource policy remains: live Reconstruction may share the existing capture/inference path; offline Reconstruction requires idle inference and reuses `InferenceController.release_model_for_external_job()` before the isolated PyCOLMAP job. Persistent scenes, static Gaussian/NeRF previews, and prepared resume metadata are not active resource owners. Heavy Gaussian/NeRF runtimes remain externally isolated and are never launched by H2.

Hardware-specific whole-application profiling remains Phase 8.9D. H2 validates ownership/admission semantics on the current machine without inventing RTX 2060/4090/5090 thresholds or claiming unmeasured hardware results.
