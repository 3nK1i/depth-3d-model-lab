"""Safe, dependency-light compute hardware discovery.

The UI process uses this module only to choose a conservative inference
profile and describe the installed accelerator. It never allocates model
weights or tensors. NVIDIA information is queried through ``nvidia-smi``
first so startup does not create a CUDA context in the Qt process. PyTorch
is imported only for build/runtime metadata and as a fallback when
``nvidia-smi`` is unavailable.

Real, continuously changing utilization/VRAM metrics belong to the runtime
monitor added later in Phase 5; the values here are a startup snapshot.
"""

from __future__ import annotations

import csv
import importlib
import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from io import StringIO
from typing import Any

logger = logging.getLogger(__name__)

_GIB = 1024**3
_NVIDIA_SMI_TIMEOUT_SECONDS = 3.0


class AcceleratorBackend(str, Enum):
    """Compute backend selected for model inference."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Immutable startup snapshot of the selected compute device."""

    backend: AcceleratorBackend
    device: str
    device_index: int | None
    name: str
    total_memory_bytes: int | None
    free_memory_bytes: int | None
    compute_capability: tuple[int, int] | None
    supports_fp16: bool
    supports_bf16: bool
    torch_available: bool
    torch_cuda_available: bool
    torch_version: str | None
    torch_cuda_version: str | None
    compiled_architectures: tuple[str, ...]
    torch_build_compatible: bool | None
    warning: str | None = None

    @property
    def has_cuda_device(self) -> bool:
        return self.backend is AcceleratorBackend.CUDA

    @property
    def cuda_ready(self) -> bool:
        """Whether the current environment is suitable for CUDA inference.

        ``None`` compatibility means the build could not be proven either
        way; model loading still gets a chance and will report a precise
        worker-side error if the wheel cannot execute on the device.
        """

        return (
            self.has_cuda_device
            and self.torch_available
            and self.torch_cuda_available
            and self.torch_build_compatible is not False
        )

    @property
    def total_memory_gib(self) -> float | None:
        if self.total_memory_bytes is None:
            return None
        return self.total_memory_bytes / _GIB

    @property
    def free_memory_gib(self) -> float | None:
        if self.free_memory_bytes is None:
            return None
        return self.free_memory_bytes / _GIB

    @property
    def short_label(self) -> str:
        if not self.has_cuda_device:
            return "CPU"
        display_name = self.name
        for prefix in ("NVIDIA GeForce ", "NVIDIA "):
            if display_name.startswith(prefix):
                display_name = display_name[len(prefix):]
                break
        memory = self.total_memory_gib
        if memory is None:
            return display_name
        return f"{display_name} · {memory:.1f} GB"

    @property
    def details(self) -> str:
        lines = [f"Device: {self.name}", f"Backend: {self.device}"]
        if self.total_memory_gib is not None:
            lines.append(f"VRAM: {self.total_memory_gib:.1f} GB")
        if self.compute_capability is not None:
            major, minor = self.compute_capability
            lines.append(f"Compute capability: {major}.{minor}")
        if self.torch_version:
            lines.append(f"PyTorch: {self.torch_version}")
        if self.torch_cuda_version:
            lines.append(f"PyTorch CUDA build: {self.torch_cuda_version}")
        if self.warning:
            lines.append(f"Warning: {self.warning}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _NvidiaDevice:
    index: int
    name: str
    total_memory_bytes: int | None
    free_memory_bytes: int | None
    compute_capability: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class _TorchMetadata:
    available: bool
    cuda_available: bool
    version: str | None
    cuda_version: str | None
    compiled_architectures: tuple[str, ...]
    module: Any | None = None


@lru_cache(maxsize=8)
def detect_hardware(device_index: int = 0) -> HardwareInfo:
    """Detect the preferred NVIDIA GPU and current PyTorch capability.

    The result is cached because this is a startup capability snapshot, not
    a live monitor. A separate Phase 5 runtime monitor will refresh changing
    values such as memory usage and utilization.
    """

    torch_meta = _read_torch_metadata()
    nvidia_devices = _query_nvidia_smi()

    if nvidia_devices:
        selected = next((gpu for gpu in nvidia_devices if gpu.index == device_index), None)
        selection_warning: str | None = None
        if selected is None:
            selected = nvidia_devices[0]
            selection_warning = (
                f"CUDA device {device_index} was not reported; using device {selected.index}."
            )
        return _hardware_from_nvidia(selected, torch_meta, selection_warning)

    if torch_meta.cuda_available and torch_meta.module is not None:
        fallback = _query_torch_device(torch_meta.module, device_index)
        if fallback is not None:
            return _hardware_from_nvidia(fallback, torch_meta, None)

    warning = None
    if not torch_meta.available:
        warning = "PyTorch is not installed; real model inference is unavailable."
    elif torch_meta.cuda_version is None:
        warning = "The installed PyTorch build is CPU-only."
    elif not torch_meta.cuda_available:
        warning = "PyTorch cannot access a CUDA device in the current environment."

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
        torch_available=torch_meta.available,
        torch_cuda_available=torch_meta.cuda_available,
        torch_version=torch_meta.version,
        torch_cuda_version=torch_meta.cuda_version,
        compiled_architectures=torch_meta.compiled_architectures,
        torch_build_compatible=None,
        warning=warning,
    )


def _hardware_from_nvidia(
    gpu: _NvidiaDevice,
    torch_meta: _TorchMetadata,
    selection_warning: str | None,
) -> HardwareInfo:
    compatibility = _torch_build_compatibility(
        gpu.compute_capability,
        torch_meta.compiled_architectures,
        torch_meta.cuda_available,
    )

    warnings: list[str] = []
    if selection_warning:
        warnings.append(selection_warning)
    if not torch_meta.available:
        warnings.append("NVIDIA GPU detected, but PyTorch is not installed.")
    elif torch_meta.cuda_version is None:
        warnings.append("NVIDIA GPU detected, but the installed PyTorch build is CPU-only.")
    elif not torch_meta.cuda_available:
        warnings.append("PyTorch CUDA is installed but cannot access this GPU.")
    elif compatibility is False:
        capability = gpu.compute_capability
        cap_text = f"sm_{capability[0]}{capability[1]}" if capability else "this GPU"
        warnings.append(f"The installed PyTorch wheel does not include {cap_text} support.")
    elif compatibility is None and gpu.compute_capability is not None:
        warnings.append("PyTorch architecture compatibility could not be confirmed.")

    capability_major = gpu.compute_capability[0] if gpu.compute_capability else 0
    return HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device=f"cuda:{gpu.index}",
        device_index=gpu.index,
        name=gpu.name,
        total_memory_bytes=gpu.total_memory_bytes,
        free_memory_bytes=gpu.free_memory_bytes,
        compute_capability=gpu.compute_capability,
        supports_fp16=capability_major >= 5,
        supports_bf16=capability_major >= 8,
        torch_available=torch_meta.available,
        torch_cuda_available=torch_meta.cuda_available,
        torch_version=torch_meta.version,
        torch_cuda_version=torch_meta.cuda_version,
        compiled_architectures=torch_meta.compiled_architectures,
        torch_build_compatible=compatibility,
        warning=" ".join(warnings) or None,
    )


def _read_torch_metadata() -> _TorchMetadata:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - optional dependency discovery
        logger.info("PyTorch unavailable during hardware detection: %s", exc)
        return _TorchMetadata(False, False, None, None, (), None)

    version = str(getattr(torch, "__version__", "unknown"))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda = getattr(torch, "cuda", None)

    cuda_available = False
    architectures: tuple[str, ...] = ()
    if cuda is not None:
        try:
            cuda_available = bool(cuda.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.cuda.is_available() failed: %s", exc)
        try:
            architectures = tuple(str(item) for item in cuda.get_arch_list())
        except Exception:
            architectures = ()

    return _TorchMetadata(
        available=True,
        cuda_available=cuda_available,
        version=version,
        cuda_version=str(cuda_version) if cuda_version is not None else None,
        compiled_architectures=architectures,
        module=torch,
    )


def _query_nvidia_smi() -> tuple[_NvidiaDevice, ...]:
    queries = (
        "index,name,memory.total,memory.free,compute_cap",
        # Older NVIDIA tooling may not expose compute_cap as a query field.
        "index,name,memory.total,memory.free",
    )
    for fields in queries:
        command = [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
            logger.info("nvidia-smi unavailable: %s", exc)
            return ()

        if completed.returncode == 0:
            devices = _parse_nvidia_smi(completed.stdout)
            if devices:
                return devices
        else:
            logger.info(
                "nvidia-smi query %s returned %s: %s",
                fields,
                completed.returncode,
                completed.stderr.strip(),
            )
    return ()


def _parse_nvidia_smi(output: str) -> tuple[_NvidiaDevice, ...]:
    devices: list[_NvidiaDevice] = []
    reader = csv.reader(StringIO(output))
    for row in reader:
        if len(row) < 4:
            continue
        try:
            index = int(row[0].strip())
        except ValueError:
            continue
        name = row[1].strip() or f"NVIDIA GPU {index}"
        total_memory = _mib_to_bytes(row[2])
        free_memory = _mib_to_bytes(row[3])
        capability = _parse_compute_capability(row[4]) if len(row) >= 5 else None
        devices.append(
            _NvidiaDevice(
                index=index,
                name=name,
                total_memory_bytes=total_memory,
                free_memory_bytes=free_memory,
                compute_capability=capability,
            )
        )
    return tuple(devices)


def _query_torch_device(torch: Any, device_index: int) -> _NvidiaDevice | None:
    """Fallback when NVIDIA tooling is unavailable but torch sees CUDA."""

    try:
        count = int(torch.cuda.device_count())
        if count <= 0:
            return None
        index = device_index if 0 <= device_index < count else 0
        properties = torch.cuda.get_device_properties(index)
        capability_raw = torch.cuda.get_device_capability(index)
        capability = (int(capability_raw[0]), int(capability_raw[1]))
        total_memory = int(properties.total_memory)
        free_memory: int | None = None
        try:
            free_memory, _ = torch.cuda.mem_get_info(index)
            free_memory = int(free_memory)
        except Exception:
            pass
        return _NvidiaDevice(
            index=index,
            name=str(properties.name),
            total_memory_bytes=total_memory,
            free_memory_bytes=free_memory,
            compute_capability=capability,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyTorch CUDA device query failed: %s", exc)
        return None


def _torch_build_compatibility(
    capability: tuple[int, int] | None,
    architectures: tuple[str, ...],
    cuda_available: bool,
) -> bool | None:
    if not cuda_available:
        return False
    if capability is None or not architectures:
        return None

    major, minor = capability
    target = f"sm_{major}{minor}"
    if target in architectures:
        return True

    # A PTX target for the exact architecture may be JIT-compiled even if
    # no cubin target is listed. Do not make assumptions across different
    # architectures; report unknown rather than claiming compatibility.
    if f"compute_{major}{minor}" in architectures:
        return True
    return None


def _mib_to_bytes(value: str) -> int | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return int(float(cleaned) * 1024**2)
    except ValueError:
        return None


def _parse_compute_capability(value: str) -> tuple[int, int] | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        major_text, minor_text = cleaned.split(".", maxsplit=1)
        return int(major_text), int(minor_text)
    except (ValueError, TypeError):
        return None