# Phase 8.7E2 RTX 2060 CUDA-visible VRAM preflight correction

The real RTX 2060 environment probe reported 5744.5625 MiB from
`torch.cuda.get_device_properties().total_memory`. The original preflight
incorrectly treated the nominal 6 GiB board capacity as an exact CUDA-visible
minimum and rejected the valid target before real Nerfacto training could run.

The preflight now:

- keeps the measured CUDA-visible VRAM value unchanged in runtime evidence;
- still requires RTX 2060 identity and compute capability 7.5;
- uses a conservative 5500 MiB CUDA-visible floor to reject materially smaller
  devices; and
- leaves the real E2 Nerfacto training run as the authoritative fit/OOM gate.

No production dependency or authority boundary changes are introduced.
