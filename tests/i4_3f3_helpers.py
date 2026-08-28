from __future__ import annotations

from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot


def resources_constrained() -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="Synthetic constrained 6GB",
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=512 * 1024**2,
        available_ram_bytes=2 * 1024**3,
        cuda_ready=True,
    )


def resources_2060() -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="Synthetic RTX 2060",
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=5 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        cuda_ready=True,
    )


def resources_5090() -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="Synthetic RTX 5090",
        total_vram_bytes=32 * 1024**3,
        free_vram_bytes=28 * 1024**3,
        available_ram_bytes=64 * 1024**3,
        cuda_ready=True,
    )
