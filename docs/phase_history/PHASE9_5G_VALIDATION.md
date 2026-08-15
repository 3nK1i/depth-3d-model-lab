# Phase 9.5G — Performance & GPU Resource Lifecycle

Phase 9.5G hardens the already-qualified 9.5E/F interactive Nerfacto viewer without changing the persisted NeRF representation or any scene authority.

## Delivered

- one external Nerfstudio render may be in flight at a time;
- one newest navigation request may wait; older pending views are coalesced;
- the isolated worker reports real torch CUDA allocated/reserved/peak-allocation telemetry;
- the worker retains at most two rendered PNG frames per interactive session;
- the Reconstruction panel exposes resource, observed render-performance, and lifecycle diagnostics;
- Hide, Preview, appearance switch, scene clear/switch, and application shutdown release the isolated worker and reload it only on demand;
- process-group cleanup remains SIGTERM then SIGKILL fallback;
- canonical geometry remains the only picking/measurement authority;
- NeRF remains `appearance_only`; metric scale, measurement, QC, and certified-metrology authority remain false.

## Truthfulness boundaries

Torch CUDA values describe the isolated Nerfstudio worker only; they are not total application VRAM. The real target qualification also queries `nvidia-smi` by worker PID and proves that each released worker disappears from both the OS process table and NVIDIA compute-process list.

Render timings are measured observations, not visual-quality certification and not a cross-GPU performance promise. NeRF quality remains deferred to the real-scene Phase 9.5H closure gate.

## Validation

Dependency-light development gates:

- quick unit/UI-source contracts;
- integration/source-isolation contracts + resource smoke;
- queue stress test;
- affected non-PyQt backward compatibility.

Target GPU gate:

`run_phase9_5g_real_nerf_resource_check.py` repeatedly starts the persisted 9.5D Nerfacto representation, renders Fast/Balanced/Detail profiles, records worker torch CUDA telemetry and `nvidia-smi` process memory, verifies the two-frame retention bound, shuts the worker down, and verifies every PID is released. It never retrains NeRF and never modifies persistence.
