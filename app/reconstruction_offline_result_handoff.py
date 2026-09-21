"""I6.2C offline reconstruction result-handoff contract.

This dependency-light module validates the terminal worker result at the
product boundary.  It does not execute reconstruction, persist scenes, choose
providers, activate Auto, or promote sparse SfM into canonical dense geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID


class OfflineResultHandoffAvailability(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    PERSISTENT = "persistent"


@dataclass(frozen=True, slots=True)
class OfflineReconstructionResultHandoff:
    session_id: str
    output_dir: str
    result_path: str
    status: str
    availability: OfflineResultHandoffAvailability
    camera_pose_available: bool
    canonical_geometry_authoritative: bool
    persistent_scene_ready: bool
    certified_metrology: bool
    scale_status: str
    visualization_path: str
    sparse_reconstruction_path: str
    viewer_ready: bool
    sparse_export_ready: bool
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.availability, OfflineResultHandoffAvailability):
            raise TypeError("availability must be OfflineResultHandoffAvailability.")
        object.__setattr__(self, "session_id", _canonical_uuid(self.session_id, "session_id"))
        output = _absolute_path(self.output_dir, "output_dir")
        result = _absolute_path(self.result_path, "result_path")
        object.__setattr__(self, "output_dir", str(output))
        object.__setattr__(self, "result_path", str(result))
        object.__setattr__(self, "status", str(self.status).strip().lower())
        object.__setattr__(self, "camera_pose_available", bool(self.camera_pose_available))
        object.__setattr__(
            self,
            "canonical_geometry_authoritative",
            bool(self.canonical_geometry_authoritative),
        )
        object.__setattr__(self, "persistent_scene_ready", bool(self.persistent_scene_ready))
        if bool(self.certified_metrology):
            raise ValueError("I6.2C cannot admit a result that claims certified metrology.")
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "scale_status", str(self.scale_status).strip().lower() or "unqualified")
        object.__setattr__(self, "visualization_path", _optional_path_text(self.visualization_path))
        object.__setattr__(
            self,
            "sparse_reconstruction_path",
            _optional_path_text(self.sparse_reconstruction_path),
        )
        object.__setattr__(self, "viewer_ready", bool(self.viewer_ready))
        object.__setattr__(self, "sparse_export_ready", bool(self.sparse_export_ready))
        object.__setattr__(self, "warnings", tuple(str(item).strip() for item in self.warnings if str(item).strip()))

    @property
    def partial_result_available(self) -> bool:
        return self.availability is OfflineResultHandoffAvailability.PARTIAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "output_dir": self.output_dir,
            "result_path": self.result_path,
            "status": self.status,
            "availability": self.availability.value,
            "camera_pose_available": self.camera_pose_available,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "persistent_scene_ready": self.persistent_scene_ready,
            "certified_metrology": False,
            "scale_status": self.scale_status,
            "visualization_path": self.visualization_path,
            "sparse_reconstruction_path": self.sparse_reconstruction_path,
            "viewer_ready": self.viewer_ready,
            "sparse_export_ready": self.sparse_export_ready,
            "warnings": list(self.warnings),
        }


def build_offline_result_handoff(
    payload: Mapping[str, object],
    output_dir: str | Path,
    *,
    expected_session_id: str,
    expected_output_dir: str | Path,
) -> OfflineReconstructionResultHandoff:
    """Validate one completed worker result for product-layer handoff."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    root = _absolute_path(output_dir, "output_dir")
    expected_root = _absolute_path(expected_output_dir, "expected_output_dir")
    if root != expected_root:
        raise ValueError("Offline result output directory does not match the admitted output directory.")

    session_id = _canonical_uuid(payload.get("session_id", ""), "offline result session_id")
    expected_sid = _canonical_uuid(expected_session_id, "expected_session_id")
    if session_id != expected_sid:
        raise ValueError("Offline result session identity does not match the admitted session identity.")

    status = str(payload.get("status", "")).strip().lower()
    if status != "completed" or not bool(payload.get("pass", False)):
        raise ValueError("I6.2C only admits a completed passing offline result.")
    if bool(payload.get("certified_metrology", False)):
        raise ValueError("I6.2C cannot admit a result that claims certified metrology.")

    camera_pose_available = bool(payload.get("camera_pose_available", False))
    canonical = bool(payload.get("canonical_geometry_authoritative", False))
    persistent_ready = bool(payload.get("persistent_scene_ready", False))
    if persistent_ready and not canonical:
        raise ValueError("Persistent-scene readiness requires canonical geometry authority.")

    artifacts_obj = payload.get("artifacts", {})
    if artifacts_obj is None:
        artifacts_obj = {}
    if not isinstance(artifacts_obj, Mapping):
        raise ValueError("Offline result artifacts must be a mapping when present.")
    artifacts = dict(artifacts_obj)

    visual, visual_warning = _optional_artifact_path(
        artifacts.get("offline_visualization_json", ""),
        root,
        "offline_visualization_json",
    )
    sparse, sparse_warning = _optional_artifact_path(
        artifacts.get("sparse_reconstruction_ply", ""),
        root,
        "sparse_reconstruction_ply",
    )

    warnings: list[str] = []
    if visual_warning:
        warnings.append(visual_warning)
    if sparse_warning:
        warnings.append(sparse_warning)
    viewer_ready = bool(visual and visual.is_file())
    sparse_export_ready = bool(sparse and sparse.is_file())
    if camera_pose_available and not viewer_ready:
        warnings.append("Sparse result is accepted but its visualization artifact is unavailable.")
    if camera_pose_available and not sparse_export_ready:
        warnings.append("Sparse result is accepted but its sparse reconstruction PLY is unavailable.")

    if persistent_ready:
        availability = OfflineResultHandoffAvailability.PERSISTENT
    elif camera_pose_available:
        availability = OfflineResultHandoffAvailability.PARTIAL
    else:
        availability = OfflineResultHandoffAvailability.NONE

    gauge = payload.get("sfm_summary", {})
    scale_status = "unqualified"
    if isinstance(gauge, Mapping):
        gauge_obj = gauge.get("gauge", {})
        if isinstance(gauge_obj, Mapping):
            scale_status = str(gauge_obj.get("scale_status", "")).strip().lower() or scale_status
    if availability is OfflineResultHandoffAvailability.PARTIAL and scale_status == "unqualified":
        scale_status = "arbitrary"

    result_path = root / "offline_result.json"
    if not result_path.is_file():
        raise ValueError("Accepted offline result is missing offline_result.json in its admitted output directory.")

    return OfflineReconstructionResultHandoff(
        session_id=session_id,
        output_dir=str(root),
        result_path=str(result_path),
        status=status,
        availability=availability,
        camera_pose_available=camera_pose_available,
        canonical_geometry_authoritative=canonical,
        persistent_scene_ready=persistent_ready,
        certified_metrology=False,
        scale_status=scale_status,
        visualization_path=str(visual or ""),
        sparse_reconstruction_path=str(sparse or ""),
        viewer_ready=viewer_ready,
        sparse_export_ready=sparse_export_ready,
        warnings=tuple(warnings),
    )


def _canonical_uuid(value: object, field_name: str) -> str:
    text = str(value).strip()
    try:
        return str(UUID(text))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID.") from exc


def _absolute_path(value: str | Path, field_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute.")
    return path.resolve()


def _optional_path_text(value: object) -> str:
    text = str(value or "").strip()
    return str(Path(text).expanduser().resolve()) if text else ""


def _optional_artifact_path(
    value: object,
    root: Path,
    field_name: str,
) -> tuple[Path | None, str]:
    """Admit optional presentation/export artifacts without changing result authority."""

    text = str(value or "").strip()
    if not text:
        return None, ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        return None, f"Offline result artifact {field_name} is not an absolute path."
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None, (
            f"Offline result artifact {field_name} is outside the admitted output directory "
            "and was not handed to the product viewer/export layer."
        )
    return path, ""
