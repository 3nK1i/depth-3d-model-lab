"""Depth Anything 3 isolation, provenance and normalization contracts for Phase 8.7F2.

The production reconstruction core never imports torch, xformers or depth_anything_3.
Those dependencies live in a separately managed interpreter and communicate through
JSON/NumPy artifacts.  DA3 output is supporting relative geometry unless a later
phase explicitly aligns and certifies scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.reconstruction.errors import ReconstructionError, ReconstructionValidationError
from core.reconstruction.sfm_coordinates import invert_rigid_matrix, validate_rigid_matrix

DA3_SCHEMA_VERSION = 1
DA3_PROVIDER_ID = "depth-anything-3-external"
DA3_ADAPTER_VERSION = "phase8-7f2-1"
DA3_UPSTREAM_REPOSITORY = "https://github.com/ByteDance-Seed/Depth-Anything-3"
DA3_EFFICIENT_MODEL_ID = "depth-anything/DA3-SMALL"
DA3_BALANCED_MODEL_ID = "depth-anything/DA3-BASE"
DA3_ALLOWED_MODEL_IDS = (DA3_EFFICIENT_MODEL_ID, DA3_BALANCED_MODEL_ID)
DA3_OUTPUT_CAMERA_AXES = "opencv_x_right_y_down_z_forward"
DA3_OUTPUT_EXTRINSICS_CONVENTION = "camera_from_reconstruction_w2c"
DA3_INTERNAL_POSE_CONVENTION = "T_reconstruction_from_camera"
DA3_GEOMETRY_SCALE = "relative_arbitrary_scale"
DA3_TURING_COMPATIBILITY_PATCH = "turing_native_bf16_autocast_pr269_equivalent"

DA3_RUNTIME_PYTHON = "3.11"
DA3_RUNTIME_TORCH = "2.7.1"
DA3_RUNTIME_TORCHVISION = "0.22.1"
DA3_RUNTIME_XFORMERS = "0.0.31"
DA3_RUNTIME_CUDA = "12.8"


class DA3WorkerExecutionError(ReconstructionError):
    """Raised when the isolated DA3 worker cannot complete safely."""


def _sha256(value: object, field_name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ReconstructionValidationError(f"{field_name} must be a SHA-256 hex digest.")
    return digest


def _git_revision(value: object, field_name: str = "source_revision") -> str:
    revision = str(value).strip().lower()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ReconstructionValidationError(f"{field_name} must be a full 40-character Git revision.")
    return revision


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ReconstructionValidationError(f"{field_name} must be a positive integer.")
    parsed = int(value)
    if parsed < 1:
        raise ReconstructionValidationError(f"{field_name} must be a positive integer.")
    return parsed


@dataclass(frozen=True, slots=True)
class DA3RuntimeLock:
    source_revision: str
    patch_sha256: str
    python_version: str
    torch_version: str
    torchvision_version: str
    xformers_version: str
    torch_cuda_version: str
    compatibility_patch: str = DA3_TURING_COMPATIBILITY_PATCH
    schema_version: int = DA3_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_revision", _git_revision(self.source_revision))
        object.__setattr__(self, "patch_sha256", _sha256(self.patch_sha256, "patch_sha256"))
        for name in (
            "python_version",
            "torch_version",
            "torchvision_version",
            "xformers_version",
            "torch_cuda_version",
            "compatibility_patch",
        ):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), field_name=name))
        if int(self.schema_version) != DA3_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported DA3 runtime-lock schema version.")
        object.__setattr__(self, "schema_version", DA3_SCHEMA_VERSION)

    @property
    def expected_versions_match(self) -> bool:
        return (
            self.python_version.startswith(DA3_RUNTIME_PYTHON + ".")
            and self.torch_version.split("+", 1)[0] == DA3_RUNTIME_TORCH
            and self.torchvision_version.split("+", 1)[0] == DA3_RUNTIME_TORCHVISION
            and self.xformers_version == DA3_RUNTIME_XFORMERS
            and self.torch_cuda_version == DA3_RUNTIME_CUDA
            and self.compatibility_patch == DA3_TURING_COMPATIBILITY_PATCH
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_revision": self.source_revision,
            "patch_sha256": self.patch_sha256,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "xformers_version": self.xformers_version,
            "torch_cuda_version": self.torch_cuda_version,
            "compatibility_patch": self.compatibility_patch,
            "expected_versions_match": self.expected_versions_match,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DA3RuntimeLock":
        if not isinstance(payload, Mapping):
            raise ReconstructionValidationError("DA3 runtime lock must be a JSON object.")
        try:
            return cls(
                source_revision=str(payload["source_revision"]),
                patch_sha256=str(payload["patch_sha256"]),
                python_version=str(payload["python_version"]),
                torch_version=str(payload["torch_version"]),
                torchvision_version=str(payload["torchvision_version"]),
                xformers_version=str(payload["xformers_version"]),
                torch_cuda_version=str(payload["torch_cuda_version"]),
                compatibility_patch=str(payload.get("compatibility_patch", DA3_TURING_COMPATIBILITY_PATCH)),
                schema_version=int(payload.get("schema_version", DA3_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionValidationError(f"Invalid DA3 runtime lock: {exc}") from exc


def read_da3_runtime_lock(path: str | Path) -> DA3RuntimeLock:
    lock_path = Path(path)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read DA3 runtime lock: {exc}") from exc
    return DA3RuntimeLock.from_payload(payload)


@dataclass(frozen=True, slots=True)
class DA3EnvironmentProbe:
    python_executable: str
    repository_root: str
    source_revision: str = ""
    python_version: str = ""
    torch_version: str = ""
    torchvision_version: str = ""
    xformers_version: str = ""
    torch_cuda_version: str = ""
    cuda_available: bool = False
    device_names: tuple[str, ...] = ()
    device_capabilities: tuple[tuple[int, int], ...] = ()
    depth_anything_3_importable: bool = False
    turing_compatibility_patch_applied: bool = False
    runtime_lock_valid: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        python = Path(str(self.python_executable))
        repo = Path(str(self.repository_root))
        if not python.is_absolute() or not repo.is_absolute():
            raise ReconstructionValidationError("DA3 environment paths must be absolute.")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "repository_root", str(repo))
        revision = str(self.source_revision).strip().lower()
        if revision:
            _git_revision(revision)
        object.__setattr__(self, "source_revision", revision)
        names = tuple(str(v).strip() for v in self.device_names if str(v).strip())
        caps = tuple((int(a), int(b)) for a, b in self.device_capabilities)
        if len(names) != len(caps):
            raise ReconstructionValidationError("DA3 GPU names/capabilities must have equal length.")
        object.__setattr__(self, "device_names", names)
        object.__setattr__(self, "device_capabilities", caps)
        object.__setattr__(self, "error", str(self.error).strip())

    @property
    def available(self) -> bool:
        return bool(
            Path(self.python_executable).is_file()
            and Path(self.repository_root).is_dir()
            and self.source_revision
            and self.depth_anything_3_importable
            and self.cuda_available
            and self.device_names
            and self.turing_compatibility_patch_applied
            and self.runtime_lock_valid
            and not self.error
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "python_executable": self.python_executable,
            "repository_root": self.repository_root,
            "source_revision": self.source_revision,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "xformers_version": self.xformers_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_available": self.cuda_available,
            "device_names": list(self.device_names),
            "device_capabilities": [list(v) for v in self.device_capabilities],
            "depth_anything_3_importable": self.depth_anything_3_importable,
            "turing_compatibility_patch_applied": self.turing_compatibility_patch_applied,
            "runtime_lock_valid": self.runtime_lock_valid,
            "error": self.error,
        }


_DA3_PROBE_CODE = r'''
import importlib
import json
import platform
payload = {
    "python_version": platform.python_version(),
    "torch_version": "",
    "torchvision_version": "",
    "xformers_version": "",
    "torch_cuda_version": "",
    "cuda_available": False,
    "device_names": [],
    "device_capabilities": [],
    "depth_anything_3_importable": False,
    "error": "",
}
try:
    torch = importlib.import_module("torch")
    torchvision = importlib.import_module("torchvision")
    xformers = importlib.import_module("xformers")
    importlib.import_module("addict")
    importlib.import_module("depth_anything_3.api")
    payload["depth_anything_3_importable"] = True
    payload["torch_version"] = str(torch.__version__)
    payload["torchvision_version"] = str(torchvision.__version__)
    payload["xformers_version"] = str(xformers.__version__)
    payload["torch_cuda_version"] = str(torch.version.cuda or "")
    payload["cuda_available"] = bool(torch.cuda.is_available())
    if payload["cuda_available"]:
        payload["device_names"] = [str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
        payload["device_capabilities"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
'''


def probe_da3_environment(
    python_executable: str | Path,
    repository_root: str | Path,
    *,
    runtime_lock_path: str | Path | None = None,
    timeout_s: float = 30.0,
) -> DA3EnvironmentProbe:
    python = Path(python_executable)
    repo = Path(repository_root)
    lock_path = Path(runtime_lock_path) if runtime_lock_path is not None else repo / ".eia_da3_runtime_lock.json"
    timeout = float(timeout_s)
    if not isfinite(timeout) or timeout <= 0:
        raise ReconstructionValidationError("timeout_s must be finite and positive.")
    error = ""
    revision = ""
    if repo.is_dir():
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10.0,
            )
            if completed.returncode == 0:
                revision = completed.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"git_probe_failed:{type(exc).__name__}:{exc}"
    patch_applied = False
    api_path = repo / "src/depth_anything_3/api.py"
    if api_path.is_file():
        try:
            patch_applied = "is_bf16_supported(including_emulation=False)" in api_path.read_text(encoding="utf-8")
        except OSError:
            patch_applied = False
    lock_valid = False
    if lock_path.is_file():
        try:
            lock = read_da3_runtime_lock(lock_path)
            lock_valid = lock.expected_versions_match and bool(revision) and lock.source_revision == revision
        except ReconstructionValidationError as exc:
            if not error:
                error = f"runtime_lock_invalid:{exc}"
    payload: dict[str, Any] = {}
    if python.is_file() and not error:
        try:
            completed = subprocess.run(
                [str(python), "-c", _DA3_PROBE_CODE],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
            else:
                error = completed.stderr.strip() or f"probe_exit_{completed.returncode}"
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            error = f"runtime_probe_failed:{type(exc).__name__}:{exc}"
    elif not python.is_file() and not error:
        error = "python_executable_not_found"
    return DA3EnvironmentProbe(
        python_executable=str(python.absolute()),
        repository_root=str(repo.absolute()),
        source_revision=revision,
        python_version=str(payload.get("python_version", "")),
        torch_version=str(payload.get("torch_version", "")),
        torchvision_version=str(payload.get("torchvision_version", "")),
        xformers_version=str(payload.get("xformers_version", "")),
        torch_cuda_version=str(payload.get("torch_cuda_version", "")),
        cuda_available=bool(payload.get("cuda_available", False)),
        device_names=tuple(payload.get("device_names", ())),
        device_capabilities=tuple(tuple(v) for v in payload.get("device_capabilities", ())),
        depth_anything_3_importable=bool(payload.get("depth_anything_3_importable", False)),
        turing_compatibility_patch_applied=patch_applied,
        runtime_lock_valid=lock_valid,
        error=error or str(payload.get("error", "")),
    )


def da3_opencv_w2c_to_reconstruction_from_camera(value: object) -> np.ndarray:
    """Convert DA3/OpenCV camera-from-world extrinsics to internal camera-to-map poses."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] not in {(3, 4), (4, 4)}:
        raise ReconstructionValidationError("DA3 extrinsics must have shape (N,3,4) or (N,4,4).")
    poses: list[np.ndarray] = []
    for index, matrix in enumerate(array):
        homogeneous = np.eye(4, dtype=np.float64)
        if matrix.shape == (3, 4):
            homogeneous[:3, :] = matrix
        else:
            homogeneous[:, :] = matrix
        camera_from_reconstruction = validate_rigid_matrix(
            homogeneous, field_name=f"da3_extrinsics[{index}]"
        )
        poses.append(invert_rigid_matrix(camera_from_reconstruction))
    result = np.stack(poses, axis=0)
    result.setflags(write=False)
    return result


def validate_da3_intrinsics(value: object, *, frame_count: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != (3, 3) or not np.isfinite(array).all():
        raise ReconstructionValidationError("DA3 intrinsics must be finite with shape (N,3,3).")
    if frame_count is not None and array.shape[0] != int(frame_count):
        raise ReconstructionValidationError("DA3 intrinsics frame count mismatch.")
    if np.any(array[:, 0, 0] <= 0) or np.any(array[:, 1, 1] <= 0):
        raise ReconstructionValidationError("DA3 focal lengths must be positive.")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def validate_da3_depth_confidence(depth: object, confidence: object) -> tuple[np.ndarray, np.ndarray]:
    depth_array = np.asarray(depth, dtype=np.float32)
    conf_array = np.asarray(confidence, dtype=np.float32)
    if depth_array.ndim != 3 or conf_array.shape != depth_array.shape:
        raise ReconstructionValidationError("DA3 depth/confidence must share shape (N,H,W).")
    if not np.isfinite(depth_array).all() or not np.isfinite(conf_array).all():
        raise ReconstructionValidationError("DA3 depth/confidence must be finite.")
    if np.any(depth_array <= 0):
        raise ReconstructionValidationError("DA3 depth must be strictly positive.")
    if np.any(conf_array < 0):
        raise ReconstructionValidationError("DA3 confidence must be non-negative.")
    d = np.ascontiguousarray(depth_array).copy()
    c = np.ascontiguousarray(conf_array).copy()
    d.setflags(write=False)
    c.setflags(write=False)
    return d, c


@dataclass(frozen=True, slots=True)
class DA3ResultManifest:
    request_id: str
    model_id: str
    model_revision: str
    source_revision: str
    output_npz_path: str
    frame_count: int
    width: int
    height: int
    process_res: int
    inference_duration_s: float
    peak_vram_mib: float
    gpu_name: str
    camera_axes: str = DA3_OUTPUT_CAMERA_AXES
    extrinsics_convention: str = DA3_OUTPUT_EXTRINSICS_CONVENTION
    geometry_scale: str = DA3_GEOMETRY_SCALE
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = DA3_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        model = require_nonempty(self.model_id, field_name="model_id")
        if model not in DA3_ALLOWED_MODEL_IDS:
            raise ReconstructionValidationError(f"Unsupported F2 DA3 model_id: {model!r}.")
        object.__setattr__(self, "model_id", model)
        revision = str(self.model_revision).strip().lower()
        if revision and (len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision)):
            raise ReconstructionValidationError("model_revision must be empty or a 40-character hex revision.")
        object.__setattr__(self, "model_revision", revision)
        object.__setattr__(self, "source_revision", _git_revision(self.source_revision))
        path = Path(require_nonempty(self.output_npz_path, field_name="output_npz_path"))
        if not path.is_absolute():
            raise ReconstructionValidationError("DA3 output_npz_path must be absolute.")
        object.__setattr__(self, "output_npz_path", str(path))
        object.__setattr__(self, "frame_count", _positive_int(self.frame_count, "frame_count"))
        object.__setattr__(self, "width", _positive_int(self.width, "width"))
        object.__setattr__(self, "height", _positive_int(self.height, "height"))
        object.__setattr__(self, "process_res", _positive_int(self.process_res, "process_res"))
        duration = float(self.inference_duration_s)
        peak = float(self.peak_vram_mib)
        if not isfinite(duration) or duration <= 0 or not isfinite(peak) or peak < 0:
            raise ReconstructionValidationError("DA3 duration/VRAM evidence is invalid.")
        object.__setattr__(self, "inference_duration_s", duration)
        object.__setattr__(self, "peak_vram_mib", peak)
        object.__setattr__(self, "gpu_name", require_nonempty(self.gpu_name, field_name="gpu_name"))
        if self.camera_axes != DA3_OUTPUT_CAMERA_AXES or self.extrinsics_convention != DA3_OUTPUT_EXTRINSICS_CONVENTION:
            raise ReconstructionValidationError("DA3 camera convention drift detected.")
        if self.geometry_scale != DA3_GEOMETRY_SCALE:
            raise ReconstructionValidationError("F2 DA3 output must remain relative/arbitrary scale.")
        if bool(self.canonical_geometry_authoritative) or bool(self.certified_metrology):
            raise ReconstructionValidationError("F2 DA3 output is supporting geometry only and not certified metrology.")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        if int(self.schema_version) != DA3_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported DA3 result schema version.")
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", DA3_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "source_revision": self.source_revision,
            "output_npz_path": self.output_npz_path,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "process_res": self.process_res,
            "inference_duration_s": self.inference_duration_s,
            "peak_vram_mib": self.peak_vram_mib,
            "gpu_name": self.gpu_name,
            "camera_axes": self.camera_axes,
            "extrinsics_convention": self.extrinsics_convention,
            "internal_pose_convention": DA3_INTERNAL_POSE_CONVENTION,
            "geometry_scale": self.geometry_scale,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DA3ResultManifest":
        if not isinstance(payload, Mapping):
            raise ReconstructionValidationError("DA3 result manifest must be a JSON object.")
        try:
            return cls(
                request_id=str(payload["request_id"]),
                model_id=str(payload["model_id"]),
                model_revision=str(payload.get("model_revision", "")),
                source_revision=str(payload["source_revision"]),
                output_npz_path=str(payload["output_npz_path"]),
                frame_count=int(payload["frame_count"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
                process_res=int(payload["process_res"]),
                inference_duration_s=float(payload["inference_duration_s"]),
                peak_vram_mib=float(payload["peak_vram_mib"]),
                gpu_name=str(payload["gpu_name"]),
                camera_axes=str(payload.get("camera_axes", DA3_OUTPUT_CAMERA_AXES)),
                extrinsics_convention=str(payload.get("extrinsics_convention", DA3_OUTPUT_EXTRINSICS_CONVENTION)),
                geometry_scale=str(payload.get("geometry_scale", DA3_GEOMETRY_SCALE)),
                canonical_geometry_authoritative=bool(payload.get("canonical_geometry_authoritative", False)),
                certified_metrology=bool(payload.get("certified_metrology", False)),
                metadata=payload.get("metadata", {}),
                schema_version=int(payload.get("schema_version", DA3_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionValidationError(f"Invalid DA3 result manifest: {exc}") from exc


def read_da3_result_manifest(path: str | Path) -> DA3ResultManifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read DA3 result manifest: {exc}") from exc
    return DA3ResultManifest.from_payload(payload)


def validate_da3_npz_result(manifest: DA3ResultManifest) -> dict[str, np.ndarray]:
    if not isinstance(manifest, DA3ResultManifest):
        raise TypeError("manifest must be DA3ResultManifest.")
    path = Path(manifest.output_npz_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReconstructionValidationError("DA3 result NPZ is missing or empty.")
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"depth", "confidence", "extrinsics_w2c", "intrinsics"}
            if not required.issubset(payload.files):
                raise ReconstructionValidationError("DA3 result NPZ is missing required arrays.")
            depth, confidence = validate_da3_depth_confidence(payload["depth"], payload["confidence"])
            poses = da3_opencv_w2c_to_reconstruction_from_camera(payload["extrinsics_w2c"])
            intrinsics = validate_da3_intrinsics(payload["intrinsics"], frame_count=manifest.frame_count)
    except (OSError, ValueError) as exc:
        raise ReconstructionValidationError(f"Unable to read DA3 result NPZ: {exc}") from exc
    if depth.shape[0] != manifest.frame_count or depth.shape[1:] != (manifest.height, manifest.width):
        raise ReconstructionValidationError("DA3 result dimensions do not match manifest evidence.")
    if poses.shape[0] != manifest.frame_count:
        raise ReconstructionValidationError("DA3 pose count does not match manifest evidence.")
    return {"depth": depth, "confidence": confidence, "T_reconstruction_from_camera": poses, "intrinsics": intrinsics}
