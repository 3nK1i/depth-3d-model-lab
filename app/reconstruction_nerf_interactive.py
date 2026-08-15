"""Phase 9.5E dependency-light contracts for interactive persisted NeRF viewing.

The application never imports Nerfstudio/tiny-cuda-nn here.  This module only
validates the persisted 9.5D load specification and converts the existing
Reconstruction turntable presentation camera into a Nerfstudio/OpenGL c2w
render request.  The external renderer remains appearance-only; canonical
geometry stays the sole picking/measurement authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import cos, radians, sin, tan
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.contracts.validation import metadata_as_dict
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.persistent_scene import verify_persistent_scene_file_references

PHASE9_5E_INTERACTIVE_NERF_SCHEMA_VERSION = 1
PHASE9_5E_DEFAULT_RENDER_WIDTH = 640
PHASE9_5E_DEFAULT_RENDER_HEIGHT = 360
PHASE9_5E_MIN_RENDER_DIMENSION = 160
PHASE9_5E_MAX_RENDER_DIMENSION = 1280

# viewer X = reconstruction X, viewer Y = reconstruction Z,
# viewer Z = -reconstruction Y.  Orthogonal, so inverse == transpose.
_VIEWER_TO_RECONSTRUCTION = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        converted = as_dict()
        if isinstance(converted, Mapping):
            return converted
    try:
        return metadata_as_dict(value)
    except (TypeError, ValueError):
        return {}


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not parse {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must be a JSON object.")
    return payload


def _absolute_existing_file(value: Any, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ReconstructionValidationError(f"{label} is missing.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ReconstructionValidationError(f"{label} must be absolute.")
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReconstructionValidationError(f"{label} is unavailable: {path}")
    return path


def _absolute_existing_dir(value: Any, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ReconstructionValidationError(f"{label} is missing.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ReconstructionValidationError(f"{label} must be absolute.")
    path = path.resolve()
    if not path.is_dir():
        raise ReconstructionValidationError(f"{label} is unavailable: {path}")
    return path


@dataclass(frozen=True, slots=True)
class InteractiveNerfPresentation:
    scene_id: str
    artifact_id: str
    provider_id: str
    backend_id: str
    backend_version: str
    backend_revision: str
    method_name: str
    load_spec_path: str
    checkpoint_path: str
    immutable_config_path: str
    dataset_root: str
    transforms_path: str
    preview_path: str
    dataset_fingerprint_sha256: str
    metric_geometry_status: str
    unit_to_metre: float | None
    ready: bool = True
    scene_role: str = "appearance_only"
    canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False
    schema_version: int = PHASE9_5E_INTERACTIVE_NERF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != PHASE9_5E_INTERACTIVE_NERF_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported Phase 9.5E interactive NeRF schema version.")
        for value, label in (
            (self.scene_id, "scene_id"),
            (self.artifact_id, "artifact_id"),
            (self.backend_id, "backend_id"),
            (self.method_name, "method_name"),
            (self.load_spec_path, "load_spec_path"),
            (self.checkpoint_path, "checkpoint_path"),
            (self.immutable_config_path, "immutable_config_path"),
            (self.dataset_root, "dataset_root"),
            (self.transforms_path, "transforms_path"),
        ):
            if not str(value).strip():
                raise ReconstructionValidationError(f"Interactive NeRF {label} is required.")
        if str(self.scene_role).strip() != "appearance_only":
            raise ReconstructionValidationError("Interactive NeRF must remain appearance_only.")
        if any(
            bool(value)
            for value in (
                self.canonical_geometry_authoritative,
                self.metric_scale_authoritative,
                self.measurement_authoritative,
                self.qc_authoritative,
                self.certified_metrology,
            )
        ):
            raise ReconstructionValidationError(
                "Interactive NeRF cannot claim geometry/metric/measurement/QC/metrology authority."
            )
        fingerprint = str(self.dataset_fingerprint_sha256).strip().lower()
        if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
            raise ReconstructionValidationError("Interactive NeRF dataset fingerprint must be SHA-256.")
        object.__setattr__(self, "dataset_fingerprint_sha256", fingerprint)
        object.__setattr__(self, "metric_geometry_status", str(self.metric_geometry_status).strip().lower())
        object.__setattr__(self, "ready", bool(self.ready))
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "metric_scale_authoritative", False)
        object.__setattr__(self, "measurement_authoritative", False)
        object.__setattr__(self, "qc_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", PHASE9_5E_INTERACTIVE_NERF_SCHEMA_VERSION)

    @property
    def authority_text(self) -> str:
        return "Appearance only • canonical geometry remains picking/measurement authority"

    @property
    def state_text(self) -> str:
        return "Interactive Nerfacto ready • external renderer on demand" if self.ready else "Interactive NeRF unavailable"


@dataclass(frozen=True, slots=True)
class NerfInteractiveRenderRequest:
    request_id: int
    camera_to_world_gl: tuple[tuple[float, float, float, float], ...]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    scene_id: str
    artifact_id: str
    schema_version: int = PHASE9_5E_INTERACTIVE_NERF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_to_world_gl, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ReconstructionValidationError("Interactive NeRF camera_to_world_gl must be finite 4x4.")
        if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
            raise ReconstructionValidationError("Interactive NeRF camera pose must be homogeneous.")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ReconstructionValidationError("Interactive NeRF camera rotation must be orthonormal.")
        if np.linalg.det(rotation) <= 0.0:
            raise ReconstructionValidationError("Interactive NeRF camera rotation must be right-handed.")
        width = int(self.width)
        height = int(self.height)
        if not PHASE9_5E_MIN_RENDER_DIMENSION <= min(width, height):
            raise ReconstructionValidationError("Interactive NeRF render resolution is too small.")
        if max(width, height) > PHASE9_5E_MAX_RENDER_DIMENSION:
            raise ReconstructionValidationError("Interactive NeRF render resolution exceeds the 9.5E bound.")
        for value, label in ((self.fx, "fx"), (self.fy, "fy"), (self.cx, "cx"), (self.cy, "cy")):
            if not np.isfinite(float(value)):
                raise ReconstructionValidationError(f"Interactive NeRF {label} must be finite.")
        if float(self.fx) <= 0.0 or float(self.fy) <= 0.0:
            raise ReconstructionValidationError("Interactive NeRF focal lengths must be positive.")
        if not str(self.scene_id).strip() or not str(self.artifact_id).strip():
            raise ReconstructionValidationError("Interactive NeRF request requires scene/artifact identity.")
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "camera_to_world_gl", tuple(tuple(float(v) for v in row) for row in matrix))

    def as_protocol_payload(self) -> dict[str, Any]:
        return {
            "type": "render",
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "scene_id": self.scene_id,
            "artifact_id": self.artifact_id,
            "camera_to_world_gl": [list(row) for row in self.camera_to_world_gl],
            "width": self.width,
            "height": self.height,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "scene_role": "appearance_only",
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
        }


def build_interactive_nerf_presentation(scene: Any) -> InteractiveNerfPresentation | None:
    """Validate a persisted 9.5D NeRF artifact for 9.5E external rendering."""

    if scene is None:
        return None
    record = getattr(scene, "nerf_artifact", None)
    if record is None:
        return None
    raw = _as_mapping(record)
    if str(raw.get("kind", "")).strip().lower() != "nerf_scene":
        raise ReconstructionValidationError("Interactive NeRF received a non-NeRF artifact.")
    if str(raw.get("authority", "")).strip().lower() != "appearance_only":
        raise ReconstructionValidationError("Persisted NeRF changed appearance-only authority.")
    if bool(raw.get("canonical_geometry_authoritative", False)) or bool(raw.get("certified_metrology", False)):
        raise ReconstructionValidationError("Persisted NeRF claims forbidden geometry/metrology authority.")
    snapshot = _as_mapping(raw.get("payload_snapshot", {}))
    load_spec_path = _absolute_existing_file(snapshot.get("runtime_load_spec_path"), label="runtime_load_spec.json")
    load_spec = _load_json_object(load_spec_path, label="Phase 9.5D NeRF runtime load spec")
    if load_spec.get("phase") != "9.5D":
        raise ReconstructionValidationError("Interactive NeRF requires a Phase 9.5D runtime load spec.")
    if load_spec.get("load_strategy") != "isolated_staged_rebind_from_immutable_training_config":
        raise ReconstructionValidationError("Interactive NeRF runtime load strategy is unsupported.")
    if load_spec.get("scene_role") != "appearance_only":
        raise ReconstructionValidationError("Interactive NeRF runtime load spec changed appearance authority.")
    forbidden = (
        "canonical_geometry_authoritative",
        "metric_scale_authoritative",
        "measurement_authoritative",
        "qc_authoritative",
        "certified_metrology",
        "backend_start_performed",
        "rendering_performed",
    )
    if any(bool(load_spec.get(key, False)) for key in forbidden):
        raise ReconstructionValidationError("Interactive NeRF runtime load spec contains forbidden authority/execution state.")

    checkpoint = _absolute_existing_file(load_spec.get("checkpoint_path"), label="persisted NeRF checkpoint")
    config = _absolute_existing_file(load_spec.get("immutable_training_config_path"), label="persisted immutable NeRF config")
    dataset_root = _absolute_existing_dir(load_spec.get("dataset_root"), label="persisted NeRF dataset")
    transforms = _absolute_existing_file(load_spec.get("transforms_path"), label="persisted NeRF transforms")
    try:
        transforms.relative_to(dataset_root)
    except ValueError as exc:
        raise ReconstructionValidationError("Persisted NeRF transforms escaped the durable dataset root.") from exc
    preview_raw = str(snapshot.get("preview_path", "") or "").strip()
    preview = _absolute_existing_file(preview_raw, label="persisted NeRF preview") if preview_raw else None
    details = dict(_as_mapping(raw.get("metadata", {})))
    details.update({key: value for key, value in snapshot.items() if value not in (None, "")})
    fingerprint = str(load_spec.get("dataset_fingerprint_sha256", "")).strip().lower()
    if fingerprint != str(details.get("dataset_fingerprint_sha256", "")).strip().lower():
        raise ReconstructionValidationError("Interactive NeRF dataset fingerprint does not match persisted artifact metadata.")
    if str(load_spec.get("checkpoint_path")) != str(checkpoint) or str(load_spec.get("immutable_training_config_path")) != str(config):
        raise ReconstructionValidationError("Interactive NeRF load spec paths did not resolve exactly.")

    return InteractiveNerfPresentation(
        scene_id=str(getattr(scene, "scene_id", "")).strip(),
        artifact_id=str(raw.get("artifact_id", "")).strip(),
        provider_id=str(raw.get("provider_id", "")).strip(),
        backend_id=str(details.get("backend_id", "nerfstudio-nerfacto")),
        backend_version=str(details.get("backend_version", "")),
        backend_revision=str(details.get("backend_revision", "")),
        method_name=str(details.get("method_name", load_spec.get("method_name", "nerfacto"))),
        load_spec_path=str(load_spec_path),
        checkpoint_path=str(checkpoint),
        immutable_config_path=str(config),
        dataset_root=str(dataset_root),
        transforms_path=str(transforms),
        preview_path=str(preview) if preview is not None else "",
        dataset_fingerprint_sha256=fingerprint,
        metric_geometry_status=str(details.get("metric_geometry_status", "unknown")),
        unit_to_metre=details.get("unit_to_metre"),
        ready=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_interactive_nerf_integrity(
    scene: Any,
    presentation: InteractiveNerfPresentation,
) -> None:
    """Verify persistent top-level and dataset evidence once before backend start.

    This can hash a large checkpoint, so the UI calls it only on explicit
    Interactive activation rather than during lightweight sidebar refreshes.
    """

    verify_persistent_scene_file_references(scene)
    load_spec = _load_json_object(Path(presentation.load_spec_path), label="Phase 9.5D NeRF runtime load spec")
    integrity_path = _absolute_existing_file(
        load_spec.get("dataset_integrity_manifest_path"),
        label="persisted NeRF dataset integrity manifest",
    )
    payload = _load_json_object(integrity_path, label="NeRF dataset integrity manifest")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ReconstructionValidationError("NeRF dataset integrity manifest contains no files.")
    root = Path(presentation.dataset_root).resolve()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ReconstructionValidationError("NeRF dataset integrity entry is invalid.")
        relative = str(entry.get("relative_path", "")).strip()
        relative_path = Path(relative)
        if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ReconstructionValidationError("NeRF dataset integrity path is unsafe.")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ReconstructionValidationError("NeRF dataset integrity path escapes dataset root.") from exc
        if not path.is_file():
            raise ReconstructionValidationError(f"Persisted NeRF dataset file is missing: {path}")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ReconstructionValidationError(f"Persisted NeRF dataset file size changed: {path}")
        if _sha256_file(path) != str(entry.get("sha256", "")):
            raise ReconstructionValidationError(f"Persisted NeRF dataset file checksum changed: {path}")


def turntable_state_to_nerf_render_request(
    *,
    request_id: int,
    center_viewer: tuple[float, float, float],
    distance: float,
    azimuth_degrees: float,
    elevation_degrees: float,
    fov_degrees: float,
    scene_id: str,
    artifact_id: str,
    width: int = PHASE9_5E_DEFAULT_RENDER_WIDTH,
    height: int = PHASE9_5E_DEFAULT_RENDER_HEIGHT,
) -> NerfInteractiveRenderRequest:
    """Convert the existing Z-up turntable camera into a Nerfstudio OpenGL c2w.

    This is a presentation transform only.  It does not mutate or supersede any
    persisted canonical camera pose.
    """

    center = np.asarray(center_viewer, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ReconstructionValidationError("Turntable center must be a finite viewer 3-vector.")
    distance = float(distance)
    fov = float(fov_degrees)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ReconstructionValidationError("Turntable distance must be positive and finite.")
    if not np.isfinite(fov) or not 5.0 <= fov <= 140.0:
        raise ReconstructionValidationError("Turntable field of view is outside the supported range.")
    az = radians(float(azimuth_degrees))
    el = radians(float(elevation_degrees))
    # VisPy TurntableCamera with its default +Z up points along +Y when
    # azimuth=elevation=0, placing the camera on -Y. Positive elevation
    # raises the camera above the XY plane.
    radial = np.asarray(
        [cos(el) * sin(az), -cos(el) * cos(az), sin(el)],
        dtype=np.float64,
    )
    position_viewer = center + distance * radial
    forward_viewer = center - position_viewer
    forward_viewer /= np.linalg.norm(forward_viewer)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right_viewer = np.cross(forward_viewer, world_up)
    right_norm = float(np.linalg.norm(right_viewer))
    if right_norm < 1e-8:
        # Pole-safe alternate up direction.
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right_viewer = np.cross(forward_viewer, world_up)
        right_norm = float(np.linalg.norm(right_viewer))
    right_viewer /= right_norm
    up_viewer = np.cross(right_viewer, forward_viewer)
    up_viewer /= np.linalg.norm(up_viewer)
    back_viewer = -forward_viewer

    position_rec = _VIEWER_TO_RECONSTRUCTION @ position_viewer
    right_rec = _VIEWER_TO_RECONSTRUCTION @ right_viewer
    up_rec = _VIEWER_TO_RECONSTRUCTION @ up_viewer
    back_rec = _VIEWER_TO_RECONSTRUCTION @ back_viewer
    rotation = np.column_stack([right_rec, up_rec, back_rec])
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position_rec

    width_i = int(width)
    height_i = int(height)
    # VisPy's perspective ``fov`` is vertical. Nerfstudio uses pixel focal
    # lengths, so derive both fx/fy from the image height for square pixels.
    focal = 0.5 * float(height_i) / tan(radians(fov) * 0.5)
    return NerfInteractiveRenderRequest(
        request_id=int(request_id),
        camera_to_world_gl=tuple(tuple(float(value) for value in row) for row in c2w),
        width=width_i,
        height=height_i,
        fx=focal,
        fy=focal,
        cx=(width_i - 1.0) * 0.5,
        cy=(height_i - 1.0) * 0.5,
        scene_id=str(scene_id),
        artifact_id=str(artifact_id),
    )


__all__ = [
    "PHASE9_5E_INTERACTIVE_NERF_SCHEMA_VERSION",
    "PHASE9_5E_DEFAULT_RENDER_WIDTH",
    "PHASE9_5E_DEFAULT_RENDER_HEIGHT",
    "InteractiveNerfPresentation",
    "NerfInteractiveRenderRequest",
    "build_interactive_nerf_presentation",
    "verify_interactive_nerf_integrity",
    "turntable_state_to_nerf_render_request",
]
