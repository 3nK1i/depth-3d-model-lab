#!/usr/bin/env python3
"""Inspect or simulate the Phase 5 automatic hardware tuning decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware import (  # noqa: E402
    AcceleratorBackend,
    ComputeProfile,
    HardwareInfo,
    detect_hardware,
    resolve_profile,
)

_GIB = 1024**3


def _simulated_hardware(name: str) -> HardwareInfo:
    normalized = name.strip().lower()
    if normalized == "rtx2060":
        return HardwareInfo(
            backend=AcceleratorBackend.CUDA,
            device="cuda:0",
            device_index=0,
            name="NVIDIA GeForce RTX 2060",
            total_memory_bytes=6 * _GIB,
            free_memory_bytes=int(5.2 * _GIB),
            compute_capability=(7, 5),
            supports_fp16=True,
            supports_bf16=False,
            torch_available=True,
            torch_cuda_available=True,
            torch_version="simulated",
            torch_cuda_version="simulated",
            compiled_architectures=("sm_75",),
            torch_build_compatible=True,
        )
    if normalized == "rtx5090":
        return HardwareInfo(
            backend=AcceleratorBackend.CUDA,
            device="cuda:0",
            device_index=0,
            name="NVIDIA GeForce RTX 5090",
            total_memory_bytes=32 * _GIB,
            free_memory_bytes=int(30 * _GIB),
            compute_capability=(12, 0),
            supports_fp16=True,
            supports_bf16=True,
            torch_available=True,
            torch_cuda_available=True,
            torch_version="simulated",
            torch_cuda_version="simulated",
            compiled_architectures=("sm_120",),
            torch_build_compatible=True,
        )
    if normalized == "rtx5090-pressure":
        hardware = _simulated_hardware("rtx5090")
        return HardwareInfo(
            backend=hardware.backend,
            device=hardware.device,
            device_index=hardware.device_index,
            name=hardware.name,
            total_memory_bytes=hardware.total_memory_bytes,
            free_memory_bytes=10 * _GIB,
            compute_capability=hardware.compute_capability,
            supports_fp16=hardware.supports_fp16,
            supports_bf16=hardware.supports_bf16,
            torch_available=True,
            torch_cuda_available=True,
            torch_version="simulated",
            torch_cuda_version="simulated",
            compiled_architectures=("sm_120",),
            torch_build_compatible=True,
        )
    if normalized == "cpu":
        return HardwareInfo(
            backend=AcceleratorBackend.CPU,
            device="cpu",
            device_index=None,
            name="CPU",
            total_memory_bytes=None,
            free_memory_bytes=None,
            compute_capability=None,
            supports_fp16=False,
            supports_bf16=False,
            torch_available=True,
            torch_cuda_available=False,
            torch_version="simulated",
            torch_cuda_version=None,
            compiled_architectures=(),
            torch_build_compatible=None,
        )
    raise ValueError(f"Unknown simulation: {name!r}")


def _report(hardware: HardwareInfo) -> dict[str, object]:
    automatic = resolve_profile(ComputeProfile.AUTO, hardware)
    return {
        "hardware": {
            "name": hardware.name,
            "backend": hardware.backend.value,
            "device": hardware.device,
            "total_vram_gib": hardware.total_memory_gib,
            "free_vram_gib": hardware.free_memory_gib,
            "compute_capability": hardware.compute_capability,
            "supports_fp16": hardware.supports_fp16,
            "supports_bf16": hardware.supports_bf16,
            "torch_build_compatible": hardware.torch_build_compatible,
            "warning": hardware.warning,
        },
        "automatic_profile": {
            "requested": automatic.requested.value,
            "effective": automatic.effective.value,
            "device": automatic.device,
            "precision": automatic.precision.value,
            "model_repo": automatic.spec.model_repo,
            "model_label": automatic.spec.model_label,
            "resolution_level": automatic.spec.resolution_level,
            "num_tokens": automatic.spec.num_tokens,
            "static_point_budget": automatic.spec.point_budget,
            "live_point_budget": automatic.spec.live_point_budget,
            "warmup_size": automatic.spec.warmup_size,
            "warmup_runs": automatic.spec.warmup_runs,
            "allow_tf32": automatic.spec.allow_tf32,
            "matmul_precision": automatic.spec.matmul_precision,
            "native_sdpa": automatic.spec.enable_native_sdpa,
            "downgraded": automatic.downgraded,
            "memory_pressure": automatic.memory_pressure,
            "warning": automatic.warning,
        },
    }


def _print_report(report: dict[str, object]) -> None:
    hardware = report["hardware"]
    automatic = report["automatic_profile"]
    assert isinstance(hardware, dict)
    assert isinstance(automatic, dict)

    print("=== Hardware ===")
    print(f"Name             : {hardware['name']}")
    print(f"Backend          : {hardware['device']}")
    print(f"Total VRAM       : {_fmt_gib(hardware['total_vram_gib'])}")
    print(f"Startup free VRAM: {_fmt_gib(hardware['free_vram_gib'])}")
    print(f"Compute capability: {hardware['compute_capability']}")
    print(f"FP16 / BF16      : {hardware['supports_fp16']} / {hardware['supports_bf16']}")
    print(f"Torch compatible : {hardware['torch_build_compatible']}")
    if hardware.get("warning"):
        print(f"Warning          : {hardware['warning']}")

    print("\n=== Automatic profile ===")
    print(f"Effective        : {str(automatic['effective']).capitalize()}")
    print(f"Model            : {automatic['model_label']}")
    print(f"Checkpoint       : {automatic['model_repo']}")
    print(f"Precision        : {str(automatic['precision']).upper()}")
    print(f"Detail           : level {automatic['resolution_level']} / {automatic['num_tokens']} tokens")
    print(f"Point budgets    : {int(automatic['static_point_budget']):,} static / {int(automatic['live_point_budget']):,} live")
    print(f"Warm-up          : {automatic['warmup_runs']} × {automatic['warmup_size']} px")
    print(f"TF32 / SDPA      : {automatic['allow_tf32']} / {automatic['native_sdpa']}")
    if automatic.get("warning"):
        print(f"Warning          : {automatic['warning']}")



def _probe_cuda_execution(hardware: HardwareInfo) -> tuple[bool, str]:
    """Run a one-element CUDA kernel to verify the installed wheel can execute."""

    if not hardware.has_cuda_device:
        return True, "not required"
    try:
        import torch

        device = torch.device(hardware.device)
        with torch.cuda.device(device):
            probe = torch.ones(1, device=device)
            probe.add_(1)
            torch.cuda.synchronize(device)
            del probe
            torch.cuda.empty_cache()
        return True, "tiny CUDA tensor/kernel succeeded"
    except Exception as exc:  # noqa: BLE001 - diagnostic command must report exact runtime failure
        return False, f"{type(exc).__name__}: {exc}"

def _fmt_gib(value: object) -> str:
    return "unknown" if value is None else f"{float(value):.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate",
        choices=("rtx2060", "rtx5090", "rtx5090-pressure", "cpu"),
        help="Resolve a deterministic virtual device instead of local hardware.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--assert-ready",
        action="store_true",
        help="Exit non-zero when CUDA/PyTorch is not ready on a real GPU.",
    )
    args = parser.parse_args()

    hardware = _simulated_hardware(args.simulate) if args.simulate else detect_hardware()
    report = _report(hardware)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)

    if args.assert_ready and hardware.has_cuda_device:
        if not hardware.cuda_ready:
            print("CUDA execution probe: skipped because the environment is not CUDA-ready.")
            return 2
        if not args.simulate:
            ok, detail = _probe_cuda_execution(hardware)
            print(f"CUDA execution probe: {'PASS' if ok else 'FAIL'} — {detail}")
            if not ok:
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
