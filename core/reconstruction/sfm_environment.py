"""Out-of-process discovery for the optional PyCOLMAP reconstruction environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Any

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import require_nonnegative_int

PYCOLMAP_RECOMMENDED_VERSION = "4.1.1"
PYCOLMAP_CUDA_PACKAGE = "pycolmap-cuda12"
PYCOLMAP_REQUIRED_APIS = (
    "extract_features",
    "match_exhaustive",
    "match_sequential",
    "incremental_mapping",
    "Reconstruction",
    "Camera",
    "Rigid3d",
    "Rotation3d",
    "FeatureExtractionOptions",
    "FeatureMatchingOptions",
    "IncrementalPipelineOptions",
    "FeatureExtractor",
)

_PROBE_CODE = r'''
import importlib
import importlib.metadata as md
import json

payload = {
    "available": False,
    "pycolmap_version": "",
    "distribution": "",
    "distribution_version": "",
    "cuda_devices": 0,
    "apis": {},
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
    names = %r
    payload["apis"] = {name: bool(hasattr(pycolmap, name)) for name in names}
print(json.dumps(payload, sort_keys=True))
''' % (PYCOLMAP_REQUIRED_APIS,)


@dataclass(frozen=True, slots=True)
class PycolmapEnvironmentProbe:
    python_executable: str
    available: bool
    pycolmap_version: str = ""
    distribution: str = ""
    distribution_version: str = ""
    cuda_devices: int = 0
    apis: tuple[tuple[str, bool], ...] = ()
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
        apis = tuple((require_nonempty(name, field_name="api_name"), bool(value)) for name, value in self.apis)
        if len({name for name, _ in apis}) != len(apis):
            raise ReconstructionValidationError("PyCOLMAP API probe names must be unique.")
        object.__setattr__(self, "apis", tuple(sorted(apis)))
        if self.available and not self.pycolmap_version:
            raise ReconstructionValidationError(
                "Available PyCOLMAP probes require pycolmap_version."
            )

    @property
    def api_map(self) -> dict[str, bool]:
        return dict(self.apis)

    @property
    def required_apis_available(self) -> bool:
        mapping = self.api_map
        return all(mapping.get(name, False) for name in PYCOLMAP_REQUIRED_APIS)

    @property
    def recommended_version(self) -> bool:
        return self.pycolmap_version == PYCOLMAP_RECOMMENDED_VERSION

    @property
    def cuda_ready(self) -> bool:
        return self.available and self.cuda_devices > 0 and self.required_apis_available

    @property
    def backend_ready(self) -> bool:
        return self.available and self.required_apis_available

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
            "cuda_ready": self.cuda_ready,
            "backend_ready": self.backend_ready,
            "error": self.error,
        }


def _probe_from_payload(python_executable: str, payload: dict[str, Any]) -> PycolmapEnvironmentProbe:
    apis_raw = payload.get("apis", {})
    if not isinstance(apis_raw, dict):
        raise ReconstructionValidationError("PyCOLMAP probe APIs payload must be an object.")
    return PycolmapEnvironmentProbe(
        python_executable=python_executable,
        available=bool(payload.get("available", False)),
        pycolmap_version=str(payload.get("pycolmap_version", "")),
        distribution=str(payload.get("distribution", "")),
        distribution_version=str(payload.get("distribution_version", "")),
        cuda_devices=int(payload.get("cuda_devices", 0)),
        apis=tuple((str(name), bool(value)) for name, value in apis_raw.items()),
        error=str(payload.get("error", "")),
    )


def probe_pycolmap_environment(
    python_executable: str,
    *,
    timeout_s: float = 15.0,
) -> PycolmapEnvironmentProbe:
    """Probe PyCOLMAP in a separate interpreter without importing it here."""

    executable = require_nonempty(python_executable, field_name="python_executable")
    timeout = float(timeout_s)
    if timeout <= 0.0:
        raise ReconstructionValidationError("timeout_s must be positive.")
    try:
        completed = subprocess.run(
            [executable, "-c", _PROBE_CODE],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PycolmapEnvironmentProbe(
            python_executable=executable,
            available=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        detail = completed.stderr.strip() or f"probe exited with code {completed.returncode}"
        return PycolmapEnvironmentProbe(
            python_executable=executable,
            available=False,
            error=detail,
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return PycolmapEnvironmentProbe(
            python_executable=executable,
            available=False,
            error=f"Invalid probe JSON: {exc}",
        )
    if not isinstance(payload, dict):
        return PycolmapEnvironmentProbe(
            python_executable=executable,
            available=False,
            error="PyCOLMAP probe returned a non-object JSON payload.",
        )
    return _probe_from_payload(executable, payload)
