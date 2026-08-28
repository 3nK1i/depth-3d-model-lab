from __future__ import annotations

from pathlib import Path

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from tests.i4_3c_helpers import make_da3_sfm_result


def make_f1_sfm(tmp_path: Path, *, view_count: int = 12):
    return make_da3_sfm_result(tmp_path, view_count=view_count)


def resources_high() -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="Synthetic RTX 5090",
        total_vram_bytes=32 * 1024**3,
        free_vram_bytes=28 * 1024**3,
        available_ram_bytes=64 * 1024**3,
        cuda_ready=True,
    )


def resources_low() -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="Synthetic constrained GPU",
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=0,
        available_ram_bytes=1 * 1024**2,
        cuda_ready=True,
    )


def hardware_2060() -> HardwareInfo:
    return HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device="cuda:0",
        device_index=0,
        name="NVIDIA GeForce RTX 2060",
        total_memory_bytes=6 * 1024**3,
        free_memory_bytes=5 * 1024**3,
        compute_capability=(7, 5),
        supports_fp16=True,
        supports_bf16=False,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="2.5.1",
        torch_cuda_version="12.4",
        compiled_architectures=("sm_75",),
        torch_build_compatible=True,
    )


def inflate_resolution(sfm, *, width: int = 3840, height: int = 2160):
    from dataclasses import replace
    from core.contracts import ImageSize

    size = ImageSize(width, height)
    workspace = replace(
        sfm.workspace,
        images=tuple(replace(item, image_size=size) for item in sfm.workspace.images),
    )
    camera_models = tuple(replace(item, image_size=size) for item in sfm.camera_models)
    return replace(sfm, workspace=workspace, camera_models=camera_models)


def resources_mid() -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="Synthetic RTX 2060 under load",
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=2 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        cuda_ready=True,
    )
