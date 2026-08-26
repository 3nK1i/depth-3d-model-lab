"""Read-only PyCOLMAP MVS capability and host-resource preflight for I4.1.9A.

This module does not create a dense workspace or start image undistortion,
PatchMatch Stereo, Stereo Fusion, SfM, or bundle adjustment.  Heavy PyCOLMAP
imports remain isolated in the explicitly selected external interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
from typing import Any

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_environment import (
    PYCOLMAP_CUDA_PACKAGE,
    PYCOLMAP_RECOMMENDED_VERSION,
)
from core.reconstruction.validation import require_nonnegative_int

MVS_PREFLIGHT_SCHEMA_VERSION = 1
PYCOLMAP_MVS_REQUIRED_APIS = (
    "undistort_images",
    "PatchMatchOptions",
    "patch_match_stereo",
    "StereoFusionOptions",
    "stereo_fusion",
    "Reconstruction",
    "get_num_cuda_devices",
)

# These are bounded I4.1.9A House-experiment launch guardrails, not product
# reconstruction limits.  They are intentionally configurable by the caller.
MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB = 8.0
MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB = 4.0
MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB = 2.0

_MVS_PROBE_CODE = r'''
import importlib
import importlib.metadata as md
import json

required = %r
payload = {
    "available": False,
    "pycolmap_version": "",
    "distribution": "",
    "distribution_version": "",
    "cuda_devices": 0,
    "apis": {},
    "patch_match_options_constructible": False,
    "stereo_fusion_options_constructible": False,
    "patch_match_option_fields": [],
    "stereo_fusion_option_fields": [],
    "error": "",
}
try:
    pycolmap = importlib.import_module("pycolmap")
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
else:
    payload["available"] = True
    payload["pycolmap_version"] = str(getattr(pycolmap, "__version__", ""))
    for candidate in ("pycolmap-cuda12", "pycolmap"):
        try:
            payload["distribution_version"] = md.version(candidate)
        except md.PackageNotFoundError:
            continue
        else:
            payload["distribution"] = candidate
            break
    try:
        payload["cuda_devices"] = int(pycolmap.get_num_cuda_devices())
    except Exception:
        payload["cuda_devices"] = 0
    payload["apis"] = {name: bool(hasattr(pycolmap, name)) for name in required}
    try:
        options = pycolmap.PatchMatchOptions()
    except Exception as exc:
        payload["error"] = f"PatchMatchOptions: {type(exc).__name__}: {exc}"
    else:
        payload["patch_match_options_constructible"] = True
        payload["patch_match_option_fields"] = sorted(
            name for name in dir(options) if not name.startswith("_") and name not in {"summary", "todict"}
        )
    try:
        options = pycolmap.StereoFusionOptions()
    except Exception as exc:
        prior = payload["error"]
        detail = f"StereoFusionOptions: {type(exc).__name__}: {exc}"
        payload["error"] = f"{prior}; {detail}" if prior else detail
    else:
        payload["stereo_fusion_options_constructible"] = True
        payload["stereo_fusion_option_fields"] = sorted(
            name for name in dir(options) if not name.startswith("_") and name not in {"summary", "todict"}
        )
print(json.dumps(payload, sort_keys=True))
''' % (PYCOLMAP_MVS_REQUIRED_APIS,)


def _positive_finite(value: float, *, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
    return number


@dataclass(frozen=True, slots=True)
class PycolmapMvsEnvironmentProbe:
    python_executable: str
    available: bool
    pycolmap_version: str = ""
    distribution: str = ""
    distribution_version: str = ""
    cuda_devices: int = 0
    apis: tuple[tuple[str, bool], ...] = ()
    patch_match_options_constructible: bool = False
    stereo_fusion_options_constructible: bool = False
    patch_match_option_fields: tuple[str, ...] = ()
    stereo_fusion_option_fields: tuple[str, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "python_executable",
            require_nonempty(self.python_executable, field_name="python_executable"),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool.")
        for name in ("pycolmap_version", "distribution", "distribution_version", "error"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        object.__setattr__(
            self,
            "cuda_devices",
            require_nonnegative_int(self.cuda_devices, field_name="cuda_devices"),
        )
        apis = tuple(
            (require_nonempty(name, field_name="api_name"), bool(value))
            for name, value in self.apis
        )
        if len({name for name, _ in apis}) != len(apis):
            raise ReconstructionValidationError("PyCOLMAP MVS API probe names must be unique.")
        object.__setattr__(self, "apis", tuple(sorted(apis)))
        object.__setattr__(
            self,
            "patch_match_option_fields",
            tuple(sorted({str(value).strip() for value in self.patch_match_option_fields if str(value).strip()})),
        )
        object.__setattr__(
            self,
            "stereo_fusion_option_fields",
            tuple(sorted({str(value).strip() for value in self.stereo_fusion_option_fields if str(value).strip()})),
        )
        if self.available and not self.pycolmap_version:
            raise ReconstructionValidationError(
                "Available PyCOLMAP MVS probes require pycolmap_version."
            )

    @property
    def api_map(self) -> dict[str, bool]:
        return dict(self.apis)

    @property
    def required_apis_available(self) -> bool:
        mapping = self.api_map
        return all(mapping.get(name, False) for name in PYCOLMAP_MVS_REQUIRED_APIS)

    @property
    def recommended_version(self) -> bool:
        return self.pycolmap_version == PYCOLMAP_RECOMMENDED_VERSION

    @property
    def cuda_distribution(self) -> bool:
        return self.distribution == PYCOLMAP_CUDA_PACKAGE

    @property
    def mvs_api_ready(self) -> bool:
        return bool(
            self.available
            and self.required_apis_available
            and self.patch_match_options_constructible
            and self.stereo_fusion_options_constructible
        )

    @property
    def patchmatch_cuda_ready(self) -> bool:
        return self.mvs_api_ready and self.cuda_devices > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "available": self.available,
            "pycolmap_version": self.pycolmap_version,
            "distribution": self.distribution,
            "distribution_version": self.distribution_version,
            "cuda_devices": self.cuda_devices,
            "apis": self.api_map,
            "required_apis_available": self.required_apis_available,
            "recommended_version": self.recommended_version,
            "cuda_distribution": self.cuda_distribution,
            "patch_match_options_constructible": self.patch_match_options_constructible,
            "stereo_fusion_options_constructible": self.stereo_fusion_options_constructible,
            "patch_match_option_fields": list(self.patch_match_option_fields),
            "stereo_fusion_option_fields": list(self.stereo_fusion_option_fields),
            "mvs_api_ready": self.mvs_api_ready,
            "patchmatch_cuda_ready": self.patchmatch_cuda_ready,
            "error": self.error,
        }


def _environment_from_payload(
    python_executable: str,
    payload: dict[str, Any],
) -> PycolmapMvsEnvironmentProbe:
    apis_raw = payload.get("apis", {})
    if not isinstance(apis_raw, dict):
        raise ReconstructionValidationError("PyCOLMAP MVS probe APIs payload must be an object.")
    return PycolmapMvsEnvironmentProbe(
        python_executable=python_executable,
        available=bool(payload.get("available", False)),
        pycolmap_version=str(payload.get("pycolmap_version", "")),
        distribution=str(payload.get("distribution", "")),
        distribution_version=str(payload.get("distribution_version", "")),
        cuda_devices=int(payload.get("cuda_devices", 0)),
        apis=tuple((str(name), bool(value)) for name, value in apis_raw.items()),
        patch_match_options_constructible=bool(
            payload.get("patch_match_options_constructible", False)
        ),
        stereo_fusion_options_constructible=bool(
            payload.get("stereo_fusion_options_constructible", False)
        ),
        patch_match_option_fields=tuple(payload.get("patch_match_option_fields", ())),
        stereo_fusion_option_fields=tuple(payload.get("stereo_fusion_option_fields", ())),
        error=str(payload.get("error", "")),
    )


def probe_pycolmap_mvs_environment(
    python_executable: str,
    *,
    timeout_s: float = 20.0,
) -> PycolmapMvsEnvironmentProbe:
    """Probe MVS capabilities in the selected external PyCOLMAP interpreter."""

    executable = require_nonempty(python_executable, field_name="python_executable")
    timeout = _positive_finite(timeout_s, field_name="timeout_s")
    try:
        completed = subprocess.run(
            [executable, "-c", _MVS_PROBE_CODE],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PycolmapMvsEnvironmentProbe(
            python_executable=executable,
            available=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        detail = completed.stderr.strip() or f"probe exited with code {completed.returncode}"
        return PycolmapMvsEnvironmentProbe(
            python_executable=executable,
            available=False,
            error=detail,
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return PycolmapMvsEnvironmentProbe(
            python_executable=executable,
            available=False,
            error=f"Invalid MVS probe JSON: {exc}",
        )
    if not isinstance(payload, dict):
        return PycolmapMvsEnvironmentProbe(
            python_executable=executable,
            available=False,
            error="PyCOLMAP MVS probe returned a non-object JSON payload.",
        )
    return _environment_from_payload(executable, payload)


@dataclass(frozen=True, slots=True)
class MvsGpuResource:
    index: int
    name: str
    total_vram_gib: float
    free_vram_gib: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", require_nonnegative_int(self.index, field_name="gpu_index"))
        object.__setattr__(self, "name", require_nonempty(self.name, field_name="gpu_name"))
        for field_name in ("total_vram_gib", "free_vram_gib"):
            number = float(getattr(self, field_name))
            if not isfinite(number) or number < 0.0:
                raise ReconstructionValidationError(f"{field_name} must be finite and non-negative.")
            object.__setattr__(self, field_name, number)
        if self.free_vram_gib > self.total_vram_gib + 1e-6:
            raise ReconstructionValidationError("free_vram_gib cannot exceed total_vram_gib.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_vram_gib": self.total_vram_gib,
            "free_vram_gib": self.free_vram_gib,
        }


@dataclass(frozen=True, slots=True)
class MvsHostResourceProbe:
    workspace_path: str
    disk_free_gib: float
    ram_available_gib: float
    gpus: tuple[MvsGpuResource, ...] = ()
    nvidia_smi_error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_path", require_nonempty(self.workspace_path, field_name="workspace_path"))
        for field_name in ("disk_free_gib", "ram_available_gib"):
            number = float(getattr(self, field_name))
            if not isfinite(number) or number < 0.0:
                raise ReconstructionValidationError(f"{field_name} must be finite and non-negative.")
            object.__setattr__(self, field_name, number)
        object.__setattr__(self, "gpus", tuple(self.gpus))
        object.__setattr__(self, "nvidia_smi_error", str(self.nvidia_smi_error).strip())

    @property
    def max_free_vram_gib(self) -> float:
        return max((gpu.free_vram_gib for gpu in self.gpus), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace_path": self.workspace_path,
            "disk_free_gib": self.disk_free_gib,
            "ram_available_gib": self.ram_available_gib,
            "gpus": [gpu.as_dict() for gpu in self.gpus],
            "max_free_vram_gib": self.max_free_vram_gib,
            "nvidia_smi_error": self.nvidia_smi_error,
        }


def _existing_disk_target(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _available_ram_gib(meminfo_path: Path = Path("/proc/meminfo")) -> float:
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0
    values: dict[str, int] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue
    kib = values.get("MemAvailable", values.get("MemFree", 0))
    return float(kib) / (1024.0 * 1024.0)


def _probe_nvidia_smi(*, timeout_s: float = 10.0) -> tuple[tuple[MvsGpuResource, ...], str]:
    timeout = _positive_finite(timeout_s, field_name="nvidia_smi_timeout_s")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (), f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return (), completed.stderr.strip() or f"nvidia-smi exited with code {completed.returncode}"
    gpus: list[MvsGpuResource] = []
    try:
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != 4:
                raise ValueError(f"unexpected nvidia-smi row: {line!r}")
            total_mib = float(fields[2])
            free_mib = float(fields[3])
            gpus.append(
                MvsGpuResource(
                    index=int(fields[0]),
                    name=fields[1],
                    total_vram_gib=total_mib / 1024.0,
                    free_vram_gib=free_mib / 1024.0,
                )
            )
    except (TypeError, ValueError, ReconstructionValidationError) as exc:
        return (), f"Invalid nvidia-smi output: {exc}"
    return tuple(gpus), ""


def probe_mvs_host_resources(
    workspace_path: str | Path,
    *,
    nvidia_smi_timeout_s: float = 10.0,
) -> MvsHostResourceProbe:
    """Collect disk/RAM/VRAM headroom without importing Torch or PyCOLMAP."""

    requested = Path(workspace_path).expanduser()
    disk_target = _existing_disk_target(requested)
    usage = shutil.disk_usage(disk_target)
    gpus, error = _probe_nvidia_smi(timeout_s=nvidia_smi_timeout_s)
    return MvsHostResourceProbe(
        workspace_path=str(requested.resolve()),
        disk_free_gib=float(usage.free) / (1024.0 ** 3),
        ram_available_gib=_available_ram_gib(),
        gpus=gpus,
        nvidia_smi_error=error,
    )


@dataclass(frozen=True, slots=True)
class MvsPreflightReport:
    environment: PycolmapMvsEnvironmentProbe
    resources: MvsHostResourceProbe
    min_free_disk_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB
    min_available_ram_gib: float = MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB
    min_free_vram_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB
    schema_version: int = MVS_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("min_free_disk_gib", "min_available_ram_gib", "min_free_vram_gib"):
            object.__setattr__(self, field_name, _positive_finite(getattr(self, field_name), field_name=field_name))
        if int(self.schema_version) != MVS_PREFLIGHT_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported MVS preflight schema version.")
        object.__setattr__(self, "schema_version", MVS_PREFLIGHT_SCHEMA_VERSION)

    @property
    def disk_ready(self) -> bool:
        return self.resources.disk_free_gib >= self.min_free_disk_gib

    @property
    def ram_ready(self) -> bool:
        return self.resources.ram_available_gib >= self.min_available_ram_gib

    @property
    def vram_ready(self) -> bool:
        return self.resources.max_free_vram_gib >= self.min_free_vram_gib

    @property
    def runtime_ready(self) -> bool:
        return bool(
            self.environment.mvs_api_ready
            and self.environment.patchmatch_cuda_ready
            and self.environment.recommended_version
            and self.environment.cuda_distribution
        )

    @property
    def resource_ready(self) -> bool:
        return self.disk_ready and self.ram_ready and self.vram_ready

    @property
    def ready(self) -> bool:
        return self.runtime_ready and self.resource_ready

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": "I4.1.9A-A0",
            "status": "pass" if self.ready else "not_ready",
            "read_only_preflight": True,
            "heavy_mvs_started": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "camera_geometry_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "environment": self.environment.as_dict(),
            "resources": self.resources.as_dict(),
            "guardrails": {
                "scope": "I4.1.9A House positive-control launch guardrails; configurable; not production limits",
                "min_free_disk_gib": self.min_free_disk_gib,
                "min_available_ram_gib": self.min_available_ram_gib,
                "min_free_vram_gib": self.min_free_vram_gib,
            },
            "checks": {
                "runtime_ready": self.runtime_ready,
                "disk_ready": self.disk_ready,
                "ram_ready": self.ram_ready,
                "vram_ready": self.vram_ready,
                "resource_ready": self.resource_ready,
                "ready": self.ready,
            },
        }


def build_mvs_preflight_report(
    python_executable: str,
    workspace_path: str | Path,
    *,
    timeout_s: float = 20.0,
    min_free_disk_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB,
    min_available_ram_gib: float = MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB,
    min_free_vram_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB,
) -> MvsPreflightReport:
    return MvsPreflightReport(
        environment=probe_pycolmap_mvs_environment(python_executable, timeout_s=timeout_s),
        resources=probe_mvs_host_resources(workspace_path),
        min_free_disk_gib=min_free_disk_gib,
        min_available_ram_gib=min_available_ram_gib,
        min_free_vram_gib=min_free_vram_gib,
    )
