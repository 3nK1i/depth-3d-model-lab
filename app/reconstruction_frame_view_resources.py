"""I6.5D application-owned resource handoff for frame/view planning.

Hardware discovery already belongs to the application SettingsPanel.  This
module converts that existing startup snapshot plus Linux MemAvailable into the
frozen I4.3F/I6.5 resource contract.  ReconstructionController remains probe-free.
"""
from __future__ import annotations

from pathlib import Path

from core.hardware.gpu_info import HardwareInfo
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot


def available_ram_bytes_from_meminfo(path: Path = Path("/proc/meminfo")) -> int | None:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        token = rest.strip().split()
        if not token:
            continue
        try:
            values[key] = int(token[0]) * 1024
        except ValueError:
            continue
    return values.get("MemAvailable", values.get("MemFree"))


def build_startup_frame_view_resource_snapshot(
    hardware: HardwareInfo,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
) -> AdaptiveSelectionResourceSnapshot:
    if not isinstance(hardware, HardwareInfo):
        raise TypeError("hardware must be HardwareInfo.")
    return AdaptiveSelectionResourceSnapshot.from_hardware(
        hardware,
        available_ram_bytes=available_ram_bytes_from_meminfo(meminfo_path),
    )


__all__ = [
    "available_ram_bytes_from_meminfo",
    "build_startup_frame_view_resource_snapshot",
]
