"""Persistent canonical dense RGB geometry for Phase G.

Phase G turns the qualified, filtered dense geometry produced by Phase F into a
self-contained, integrity-bound and reloadable scene bundle.  The bundle uses
relative references so it can be moved as one directory without depending on
``/tmp`` validation artifacts.

This module deliberately does not reuse the mature Phase 8 integrated
``PersistentReconstructionScene`` manifest because that contract requires a
surface-mesh, semantic and inspection stage.  Phase G must not fabricate those
later-stage artifacts merely to persist Dense RGB Points.

Canonical geometry authority is promoted for the persisted dense point cloud,
while physical metric scale, measurement authority, certified metrology, UI and
surface-mesh authority remain explicitly unpromoted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import require_iso8601_utc, require_sha256, require_uuid

DENSE_RGB_PERSISTENCE_SCHEMA_VERSION = 1
DENSE_RGB_PERSISTENCE_PHASE = "G"
DENSE_RGB_SCENE_MANIFEST = "persistent_dense_rgb_scene.json"
DENSE_RGB_SCENE_CHECKSUM = "persistent_dense_rgb_scene.json.sha256"
DENSE_RGB_RECONSTRUCTION_FRAME_ID = "offline-sfm-reconstruction"


class DenseSceneFileRole(str, Enum):
    PRIMARY_EXTENDED_NPZ = "primary_extended_npz"
    PRIMARY_EXTENDED_PLY = "primary_extended_ply"
    HIGH_CONFIDENCE_CORE_NPZ = "high_confidence_core_npz"
    HIGH_CONFIDENCE_CORE_PLY = "high_confidence_core_ply"
    REJECTED_SUPPORTED_DIAGNOSTIC = "rejected_supported_diagnostic"
    ATTACHED_SINGLE_VIEW_FRINGE_DIAGNOSTIC = "attached_single_view_fringe_diagnostic"
    DETACHED_SINGLE_VIEW_ARTIFACT_DIAGNOSTIC = "detached_single_view_artifact_diagnostic"
    COVERAGE_CANDIDATE_DIAGNOSTIC = "coverage_candidate_diagnostic"
    SOURCE_F_SUMMARY = "source_f_summary"
    SOURCE_E3_SUMMARY = "source_e3_summary"


_REQUIRED_FILE_ROLES = {
    DenseSceneFileRole.PRIMARY_EXTENDED_NPZ,
    DenseSceneFileRole.PRIMARY_EXTENDED_PLY,
    DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ,
    DenseSceneFileRole.HIGH_CONFIDENCE_CORE_PLY,
    DenseSceneFileRole.REJECTED_SUPPORTED_DIAGNOSTIC,
    DenseSceneFileRole.ATTACHED_SINGLE_VIEW_FRINGE_DIAGNOSTIC,
    DenseSceneFileRole.DETACHED_SINGLE_VIEW_ARTIFACT_DIAGNOSTIC,
    DenseSceneFileRole.COVERAGE_CANDIDATE_DIAGNOSTIC,
    DenseSceneFileRole.SOURCE_F_SUMMARY,
    DenseSceneFileRole.SOURCE_E3_SUMMARY,
}


_ROLE_DESTINATIONS: dict[DenseSceneFileRole, str] = {
    DenseSceneFileRole.PRIMARY_EXTENDED_NPZ: "geometry/dense_rgb_extended.npz",
    DenseSceneFileRole.PRIMARY_EXTENDED_PLY: "exports/dense_rgb_extended.ply",
    DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ: "geometry/dense_rgb_core.npz",
    DenseSceneFileRole.HIGH_CONFIDENCE_CORE_PLY: "exports/dense_rgb_core.ply",
    DenseSceneFileRole.REJECTED_SUPPORTED_DIAGNOSTIC: "diagnostics/rejected_supported.npz",
    DenseSceneFileRole.ATTACHED_SINGLE_VIEW_FRINGE_DIAGNOSTIC: "diagnostics/attached_single_view_fringe.npz",
    DenseSceneFileRole.DETACHED_SINGLE_VIEW_ARTIFACT_DIAGNOSTIC: "diagnostics/detached_single_view_artifacts.npz",
    DenseSceneFileRole.COVERAGE_CANDIDATE_DIAGNOSTIC: "diagnostics/coverage_candidate.npz",
    DenseSceneFileRole.SOURCE_F_SUMMARY: "provenance/f_summary.json",
    DenseSceneFileRole.SOURCE_E3_SUMMARY: "provenance/e3_summary.json",
}


_F_ARTIFACT_TO_ROLE: dict[str, DenseSceneFileRole] = {
    "extended_npz": DenseSceneFileRole.PRIMARY_EXTENDED_NPZ,
    "extended_ply": DenseSceneFileRole.PRIMARY_EXTENDED_PLY,
    "core_npz": DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ,
    "core_ply": DenseSceneFileRole.HIGH_CONFIDENCE_CORE_PLY,
    "rejected_supported_npz": DenseSceneFileRole.REJECTED_SUPPORTED_DIAGNOSTIC,
    "attached_single_view_fringe_npz": DenseSceneFileRole.ATTACHED_SINGLE_VIEW_FRINGE_DIAGNOSTIC,
    "detached_single_view_artifacts_npz": DenseSceneFileRole.DETACHED_SINGLE_VIEW_ARTIFACT_DIAGNOSTIC,
    "coverage_candidate_npz": DenseSceneFileRole.COVERAGE_CANDIDATE_DIAGNOSTIC,
}


@dataclass(frozen=True, slots=True)
class DenseSceneFileReference:
    role: DenseSceneFileRole
    relative_path: str
    size_bytes: int
    sha256: str
    media_type: str
    canonical_geometry: bool
    diagnostic_only: bool

    def __post_init__(self) -> None:
        if not isinstance(self.role, DenseSceneFileRole):
            raise TypeError("role must be DenseSceneFileRole.")
        relative = _safe_relative_path(self.relative_path)
        size = int(self.size_bytes)
        if size < 0:
            raise ReconstructionValidationError("Dense scene file size must be non-negative.")
        digest = require_sha256(self.sha256, field_name="sha256", allow_empty=False)
        media = str(self.media_type).strip()
        if not media:
            raise ReconstructionValidationError("Dense scene file media_type must be non-empty.")
        canonical = bool(self.canonical_geometry)
        diagnostic = bool(self.diagnostic_only)
        geometry_roles = {
            DenseSceneFileRole.PRIMARY_EXTENDED_NPZ,
            DenseSceneFileRole.PRIMARY_EXTENDED_PLY,
            DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ,
            DenseSceneFileRole.HIGH_CONFIDENCE_CORE_PLY,
        }
        if self.role in geometry_roles and (not canonical or diagnostic):
            raise ReconstructionValidationError("Primary/core dense geometry references must be canonical, not diagnostic.")
        if self.role not in geometry_roles and canonical:
            raise ReconstructionValidationError("Diagnostic/provenance references cannot claim canonical geometry.")
        object.__setattr__(self, "relative_path", relative)
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "media_type", media)
        object.__setattr__(self, "canonical_geometry", canonical)
        object.__setattr__(self, "diagnostic_only", diagnostic)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "canonical_geometry": self.canonical_geometry,
            "diagnostic_only": self.diagnostic_only,
        }


@dataclass(frozen=True, slots=True)
class PersistentDenseRgbScene:
    scene_id: str
    scene_fingerprint_sha256: str
    created_at_utc: str
    source_session_id: str
    reconstruction_frame_id: str
    selected_workspace_orders: tuple[int, ...]
    selected_keyframe_ids: tuple[str, ...]
    selected_view_count: int
    voxel_size_reconstruction_units: float
    primary_point_count: int
    core_point_count: int
    source_supported_point_count: int
    source_all_voxel_count: int
    source_f_summary_sha256: str
    source_e3_summary_sha256: str
    file_references: tuple[DenseSceneFileReference, ...]
    quality_snapshot: Mapping[str, Any]
    filter_policy_snapshot: Mapping[str, Any]
    schema_version: int = DENSE_RGB_PERSISTENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        fingerprint = require_sha256(
            self.scene_fingerprint_sha256,
            field_name="scene_fingerprint_sha256",
            allow_empty=False,
        )
        if str(self.scene_id) != f"pdrgb-{fingerprint[:24]}":
            raise ReconstructionValidationError("Persistent dense scene ID must derive from its fingerprint.")
        created = require_iso8601_utc(self.created_at_utc, field_name="created_at_utc")
        session = require_uuid(self.source_session_id, field_name="source_session_id")
        frame = str(self.reconstruction_frame_id).strip()
        if frame != DENSE_RGB_RECONSTRUCTION_FRAME_ID:
            raise ReconstructionValidationError("Persistent dense scene uses an unexpected reconstruction frame.")
        orders = tuple(int(value) for value in self.selected_workspace_orders)
        keyframes = tuple(str(value).strip() for value in self.selected_keyframe_ids)
        view_count = int(self.selected_view_count)
        if view_count < 2 or len(orders) != view_count or len(keyframes) != view_count:
            raise ReconstructionValidationError("Persistent dense scene selected-view provenance is incomplete.")
        if list(orders) != sorted(orders) or len(set(orders)) != len(orders):
            raise ReconstructionValidationError("Persistent dense scene workspace orders must be increasing and unique.")
        if any(not value for value in keyframes) or len(set(keyframes)) != len(keyframes):
            raise ReconstructionValidationError("Persistent dense scene keyframe IDs must be unique and non-empty.")
        voxel = float(self.voxel_size_reconstruction_units)
        if not isfinite(voxel) or voxel <= 0.0:
            raise ReconstructionValidationError("Persistent dense scene voxel size must be finite and positive.")
        primary = int(self.primary_point_count)
        core = int(self.core_point_count)
        supported = int(self.source_supported_point_count)
        all_count = int(self.source_all_voxel_count)
        if not (0 < core <= primary <= supported <= all_count):
            raise ReconstructionValidationError("Persistent dense scene point-count hierarchy is invalid.")
        f_sha = require_sha256(self.source_f_summary_sha256, field_name="source_f_summary_sha256", allow_empty=False)
        e3_sha = require_sha256(self.source_e3_summary_sha256, field_name="source_e3_summary_sha256", allow_empty=False)
        references = tuple(self.file_references)
        if any(not isinstance(item, DenseSceneFileReference) for item in references):
            raise TypeError("file_references must contain DenseSceneFileReference values.")
        roles = {item.role for item in references}
        if roles != _REQUIRED_FILE_ROLES or len(references) != len(_REQUIRED_FILE_ROLES):
            missing = sorted(role.value for role in _REQUIRED_FILE_ROLES.difference(roles))
            extra = sorted(role.value for role in roles.difference(_REQUIRED_FILE_ROLES))
            raise ReconstructionValidationError(
                f"Persistent dense scene file-role set is invalid; missing={missing!r} extra={extra!r}."
            )
        if len({item.relative_path for item in references}) != len(references):
            raise ReconstructionValidationError("Persistent dense scene file references must have unique paths.")
        if int(self.schema_version) != DENSE_RGB_PERSISTENCE_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported persistent dense RGB scene schema version.")
        expected = hashlib.sha256(_canonical_json(self._fingerprint_payload(references=references))).hexdigest()
        if fingerprint != expected:
            raise ReconstructionValidationError("Persistent dense RGB scene fingerprint does not match content.")
        object.__setattr__(self, "scene_fingerprint_sha256", fingerprint)
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "source_session_id", session)
        object.__setattr__(self, "reconstruction_frame_id", frame)
        object.__setattr__(self, "selected_workspace_orders", orders)
        object.__setattr__(self, "selected_keyframe_ids", keyframes)
        object.__setattr__(self, "selected_view_count", view_count)
        object.__setattr__(self, "voxel_size_reconstruction_units", voxel)
        object.__setattr__(self, "primary_point_count", primary)
        object.__setattr__(self, "core_point_count", core)
        object.__setattr__(self, "source_supported_point_count", supported)
        object.__setattr__(self, "source_all_voxel_count", all_count)
        object.__setattr__(self, "source_f_summary_sha256", f_sha)
        object.__setattr__(self, "source_e3_summary_sha256", e3_sha)
        object.__setattr__(self, "file_references", references)
        object.__setattr__(self, "quality_snapshot", _json_object(self.quality_snapshot, field_name="quality_snapshot"))
        object.__setattr__(self, "filter_policy_snapshot", _json_object(self.filter_policy_snapshot, field_name="filter_policy_snapshot"))
        object.__setattr__(self, "schema_version", DENSE_RGB_PERSISTENCE_SCHEMA_VERSION)

    @property
    def canonical_geometry_authoritative(self) -> bool:
        return True

    @property
    def canonical_persistence_promoted(self) -> bool:
        return True

    @property
    def metric_scale_claimed(self) -> bool:
        return False

    @property
    def measurement_authoritative(self) -> bool:
        return False

    @property
    def certified_metrology(self) -> bool:
        return False

    @property
    def surface_mesh_generated(self) -> bool:
        return False

    @property
    def reconstruction_ui_enabled(self) -> bool:
        return False

    def _fingerprint_payload(
        self,
        *,
        references: Sequence[DenseSceneFileReference] | None = None,
    ) -> dict[str, Any]:
        refs = tuple(self.file_references if references is None else references)
        return _scene_fingerprint_payload(
            source_session_id=self.source_session_id,
            reconstruction_frame_id=self.reconstruction_frame_id,
            selected_workspace_orders=self.selected_workspace_orders,
            selected_keyframe_ids=self.selected_keyframe_ids,
            selected_view_count=self.selected_view_count,
            voxel_size_reconstruction_units=self.voxel_size_reconstruction_units,
            primary_point_count=self.primary_point_count,
            core_point_count=self.core_point_count,
            source_supported_point_count=self.source_supported_point_count,
            source_all_voxel_count=self.source_all_voxel_count,
            source_f_summary_sha256=self.source_f_summary_sha256,
            source_e3_summary_sha256=self.source_e3_summary_sha256,
            file_references=refs,
            quality_snapshot=self.quality_snapshot,
            filter_policy_snapshot=self.filter_policy_snapshot,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = self._fingerprint_payload()
        payload.update(
            {
                "scene_id": self.scene_id,
                "scene_fingerprint_sha256": self.scene_fingerprint_sha256,
                "created_at_utc": self.created_at_utc,
                "representation": {
                    "kind": "dense_rgb_points",
                    "primary": DenseSceneFileRole.PRIMARY_EXTENDED_NPZ.value,
                    "high_confidence": DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ.value,
                    "primary_export": DenseSceneFileRole.PRIMARY_EXTENDED_PLY.value,
                    "high_confidence_export": DenseSceneFileRole.HIGH_CONFIDENCE_CORE_PLY.value,
                },
                "future_phase_ownership": {
                    "ui_integration": "H",
                    "offline_quality_campaign": "I",
                    "surface_mesh": "J",
                },
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class RestoredDenseRgbGeometry:
    scene_id: str
    representation: str
    points_reconstruction: np.ndarray
    colors_rgb: np.ndarray
    view_observation_counts: np.ndarray
    filter_confidence: np.ndarray
    voxel_size_reconstruction_units: float
    selected_workspace_orders: tuple[int, ...]
    selected_keyframe_ids: tuple[str, ...]
    source_file_sha256: str

    def __post_init__(self) -> None:
        points = np.asarray(self.points_reconstruction, dtype=np.float64)
        colors = np.asarray(self.colors_rgb, dtype=np.uint8)
        counts = np.asarray(self.view_observation_counts, dtype=np.uint32)
        confidence = np.asarray(self.filter_confidence, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
            raise ReconstructionValidationError("Restored dense RGB geometry requires aligned Nx3 XYZ/RGB arrays.")
        if counts.shape != (len(points),) or confidence.shape != (len(points),):
            raise ReconstructionValidationError("Restored dense RGB geometry evidence vectors are misaligned.")
        if np.any(~np.isfinite(points)) or np.any(~np.isfinite(confidence)):
            raise ReconstructionValidationError("Restored dense RGB geometry contains non-finite values.")
        if np.any(counts < 2) or np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ReconstructionValidationError("Restored dense RGB geometry evidence is outside its valid range.")
        voxel = float(self.voxel_size_reconstruction_units)
        if not isfinite(voxel) or voxel <= 0.0:
            raise ReconstructionValidationError("Restored dense RGB geometry has invalid voxel size.")
        representation = str(self.representation).strip().lower()
        if representation not in {"extended", "core"}:
            raise ReconstructionValidationError("Restored dense representation must be 'extended' or 'core'.")
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "points_reconstruction", points)
        object.__setattr__(self, "colors_rgb", colors)
        object.__setattr__(self, "view_observation_counts", counts)
        object.__setattr__(self, "filter_confidence", confidence)
        object.__setattr__(self, "voxel_size_reconstruction_units", voxel)
        object.__setattr__(self, "selected_workspace_orders", tuple(int(v) for v in self.selected_workspace_orders))
        object.__setattr__(self, "selected_keyframe_ids", tuple(str(v) for v in self.selected_keyframe_ids))
        object.__setattr__(self, "source_file_sha256", require_sha256(self.source_file_sha256, field_name="source_file_sha256", allow_empty=False))

    @property
    def point_count(self) -> int:
        return int(len(self.points_reconstruction))


@dataclass(frozen=True, slots=True)
class DenseScenePersistenceResult:
    scene: PersistentDenseRgbScene
    manifest_path: Path
    restored_extended: RestoredDenseRgbGeometry
    restored_core: RestoredDenseRgbGeometry
    source_independent: bool
    portable_relative_references: bool

    def __post_init__(self) -> None:
        if self.restored_extended.scene_id != self.scene.scene_id or self.restored_core.scene_id != self.scene.scene_id:
            raise ReconstructionValidationError("Restored dense geometry scene identity changed during persistence.")
        if self.restored_extended.point_count != self.scene.primary_point_count:
            raise ReconstructionValidationError("Restored Extended point count disagrees with persistent manifest.")
        if self.restored_core.point_count != self.scene.core_point_count:
            raise ReconstructionValidationError("Restored Core point count disagrees with persistent manifest.")
        if not self.source_independent or not self.portable_relative_references:
            raise ReconstructionValidationError("Persistent dense scene must be self-contained and portable.")


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: Mapping[str, Any] | object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{field_name} must be a JSON object.")
    try:
        normalized = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"{field_name} must contain JSON-safe values.") from exc
    if not isinstance(normalized, dict):
        raise ReconstructionValidationError(f"{field_name} must normalize to an object.")
    return normalized


def _safe_relative_path(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ReconstructionValidationError("Persistent dense scene references must be safe relative paths.")
    return path.as_posix()


def _scene_fingerprint_payload(
    *,
    source_session_id: str,
    reconstruction_frame_id: str,
    selected_workspace_orders: Sequence[int],
    selected_keyframe_ids: Sequence[str],
    selected_view_count: int,
    voxel_size_reconstruction_units: float,
    primary_point_count: int,
    core_point_count: int,
    source_supported_point_count: int,
    source_all_voxel_count: int,
    source_f_summary_sha256: str,
    source_e3_summary_sha256: str,
    file_references: Sequence[DenseSceneFileReference],
    quality_snapshot: Mapping[str, Any],
    filter_policy_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": DENSE_RGB_PERSISTENCE_SCHEMA_VERSION,
        "phase": DENSE_RGB_PERSISTENCE_PHASE,
        "source_session_id": source_session_id,
        "reconstruction_frame_id": reconstruction_frame_id,
        "selected_workspace_orders": [int(v) for v in selected_workspace_orders],
        "selected_keyframe_ids": [str(v) for v in selected_keyframe_ids],
        "selected_view_count": int(selected_view_count),
        "voxel_size_reconstruction_units": float(voxel_size_reconstruction_units),
        "primary_point_count": int(primary_point_count),
        "core_point_count": int(core_point_count),
        "source_supported_point_count": int(source_supported_point_count),
        "source_all_voxel_count": int(source_all_voxel_count),
        "source_f_summary_sha256": source_f_summary_sha256,
        "source_e3_summary_sha256": source_e3_summary_sha256,
        "file_references": [item.as_dict() for item in file_references],
        "quality_snapshot": _json_object(quality_snapshot, field_name="quality_snapshot"),
        "filter_policy_snapshot": _json_object(filter_policy_snapshot, field_name="filter_policy_snapshot"),
        "authority": _authority_payload(),
    }


def _authority_payload() -> dict[str, Any]:
    return {
        "canonical_geometry_authoritative": True,
        "canonical_persistence_promoted": True,
        "geometry_authority_scope": ["geometry", "picking", "spatial_relationships"],
        "metric_scale_claimed": False,
        "sfm_scale_status": "arbitrary",
        "sfm_world_alignment_status": "arbitrary",
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "reconstruction_ui_enabled": False,
        "gaussian_nerf_geometry_authority": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"JSON evidence root must be an object: {path}")
    return payload


def _validate_f_summary(summary: Mapping[str, Any]) -> None:
    if str(summary.get("phase", "")) != "F" or not bool(summary.get("pass")) or str(summary.get("status", "")) != "pass":
        raise ReconstructionValidationError("G requires a passing Phase F summary.")
    decision = summary.get("decision")
    if not isinstance(decision, Mapping) or not bool(decision.get("authorizes_next_canonical_dense_persistence_milestone")):
        raise ReconstructionValidationError("G requires explicit F authorization for canonical dense persistence.")
    authority = summary.get("authority")
    if not isinstance(authority, Mapping) or not bool(authority.get("filtered_dense_geometry_qualified")):
        raise ReconstructionValidationError("G requires F-qualified filtered dense geometry.")
    forbidden = (
        "metric_scale_claimed",
        "canonical_persistence_promoted",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    )
    if any(bool(authority.get(name)) for name in forbidden):
        raise ReconstructionValidationError("G source F summary contains premature authority promotion.")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or any(name not in artifacts for name in _F_ARTIFACT_TO_ROLE):
        raise ReconstructionValidationError("G requires all formal F geometry/diagnostic artifacts.")
    classification = summary.get("classification")
    if not isinstance(classification, Mapping):
        raise ReconstructionValidationError("G requires F classification evidence.")
    if int(classification.get("core_voxels", 0)) <= 0 or int(classification.get("extended_voxels", 0)) <= 0:
        raise ReconstructionValidationError("G requires non-empty F Core and Extended geometry.")


def _validate_e3_summary(summary: Mapping[str, Any], f_summary: Mapping[str, Any]) -> None:
    if str(summary.get("phase", "")) != "E3" or not bool(summary.get("pass")) or str(summary.get("status", "")) != "pass":
        raise ReconstructionValidationError("G source provenance requires a passing E3 summary.")
    if int(summary.get("selected_view_count", 0)) != int(f_summary.get("source_e3_selected_view_count", -1)):
        raise ReconstructionValidationError("G F/E3 selected-view counts disagree.")
    if list(summary.get("selected_workspace_orders", [])) != list(f_summary.get("selected_workspace_orders", [])):
        raise ReconstructionValidationError("G F/E3 workspace-order provenance disagrees.")
    if list(summary.get("selected_keyframe_ids", [])) != list(f_summary.get("selected_keyframe_ids", [])):
        raise ReconstructionValidationError("G F/E3 keyframe provenance disagrees.")
    if not str(summary.get("source_sfm_session_id", "")).strip():
        raise ReconstructionValidationError("G E3 provenance lacks source_sfm_session_id.")


def _inspect_filtered_npz(path: Path, *, representation: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReconstructionValidationError(f"Filtered dense geometry file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "points_reconstruction",
            "colors_rgb",
            "view_observation_counts",
            "filter_confidence",
            "voxel_size_reconstruction_units",
            "selected_workspace_orders",
            "selected_keyframe_ids",
            "artifact_reason_code",
            "class_code",
            "schema_version",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ReconstructionValidationError(f"Filtered dense geometry NPZ is missing fields: {missing!r}.")
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        counts = np.asarray(payload["view_observation_counts"], dtype=np.uint32)
        confidence = np.asarray(payload["filter_confidence"], dtype=np.float32)
        reasons = np.asarray(payload["artifact_reason_code"], dtype=np.uint16)
        classes = np.asarray(payload["class_code"], dtype=np.uint8)
        voxel = float(np.asarray(payload["voxel_size_reconstruction_units"]).item())
        orders = tuple(int(v) for v in np.asarray(payload["selected_workspace_orders"], dtype=np.int32).tolist())
        keyframes = tuple(str(v) for v in np.asarray(payload["selected_keyframe_ids"], dtype="U160").tolist())
        schema_version = int(np.asarray(payload["schema_version"]).item())
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ReconstructionValidationError("Filtered dense geometry XYZ/RGB arrays are invalid.")
    if counts.shape != (len(points),) or confidence.shape != (len(points),):
        raise ReconstructionValidationError("Filtered dense geometry evidence vectors are misaligned.")
    if reasons.shape != (len(points),) or classes.shape != (len(points),):
        raise ReconstructionValidationError("Filtered dense geometry classification vectors are misaligned.")
    if len(points) == 0 or np.any(~np.isfinite(points)) or np.any(~np.isfinite(confidence)):
        raise ReconstructionValidationError("Filtered dense geometry is empty or non-finite.")
    if np.any(counts < 2) or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ReconstructionValidationError("Filtered dense geometry evidence is invalid.")
    if np.any(reasons != 0):
        raise ReconstructionValidationError("Retained Core/Extended geometry cannot carry rejection reasons.")
    if representation == "core":
        if np.any(classes != 3):
            raise ReconstructionValidationError("Core persistent geometry must contain only class_code=3 voxels.")
    elif representation == "extended":
        if np.any(~np.isin(classes, np.asarray([2, 3], dtype=np.uint8))):
            raise ReconstructionValidationError("Extended persistent geometry may contain only class_code 2/3 voxels.")
    else:
        raise ReconstructionValidationError("Unknown filtered dense representation.")
    if not isfinite(voxel) or voxel <= 0.0:
        raise ReconstructionValidationError("Filtered dense geometry voxel size is invalid.")
    return {
        "point_count": int(len(points)),
        "points": points,
        "voxel_size": voxel,
        "orders": orders,
        "keyframes": keyframes,
        "schema_version": schema_version,
    }


def _voxel_key_set(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    keys = np.floor(np.asarray(points, dtype=np.float64) / float(voxel_size)).astype(np.int64)
    return {tuple(map(int, key)) for key in keys}


def _ply_vertex_count(path: Path) -> int:
    if not path.is_file():
        raise ReconstructionValidationError(f"Dense RGB PLY export is missing: {path}")
    vertex_count: int | None = None
    with path.open("rb") as handle:
        for _ in range(64):
            line = handle.readline()
            if not line:
                break
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ReconstructionValidationError("Dense RGB PLY header is not ASCII.") from exc
            if text.startswith("element vertex "):
                try:
                    vertex_count = int(text.split()[-1])
                except ValueError as exc:
                    raise ReconstructionValidationError("Dense RGB PLY vertex count is malformed.") from exc
            if text == "end_header":
                break
        else:
            raise ReconstructionValidationError("Dense RGB PLY header exceeds safety limit.")
    if vertex_count is None or vertex_count < 0:
        raise ReconstructionValidationError("Dense RGB PLY is missing a valid vertex count.")
    return vertex_count


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return "application/x-npz"
    if suffix == ".ply":
        return "model/ply"
    if suffix == ".json":
        return "application/json"
    return "application/octet-stream"


def _copy_reference(stage_root: Path, source: Path, role: DenseSceneFileRole) -> DenseSceneFileReference:
    if not source.is_file():
        raise ReconstructionValidationError(f"G source artifact is missing for role {role.value}: {source}")
    relative = _ROLE_DESTINATIONS[role]
    destination = stage_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size or _sha256_file(destination) != _sha256_file(source):
        raise ReconstructionValidationError(f"G copied artifact failed integrity verification: {role.value}")
    geometry = role in {
        DenseSceneFileRole.PRIMARY_EXTENDED_NPZ,
        DenseSceneFileRole.PRIMARY_EXTENDED_PLY,
        DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ,
        DenseSceneFileRole.HIGH_CONFIDENCE_CORE_PLY,
    }
    return DenseSceneFileReference(
        role=role,
        relative_path=relative,
        size_bytes=destination.stat().st_size,
        sha256=_sha256_file(destination),
        media_type=_media_type(destination),
        canonical_geometry=geometry,
        diagnostic_only=not geometry,
    )


def _scene_from_mapping(payload: Mapping[str, Any]) -> PersistentDenseRgbScene:
    references = tuple(
        DenseSceneFileReference(
            role=DenseSceneFileRole(str(item["role"])),
            relative_path=str(item["relative_path"]),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
            media_type=str(item["media_type"]),
            canonical_geometry=bool(item.get("canonical_geometry", False)),
            diagnostic_only=bool(item.get("diagnostic_only", False)),
        )
        for item in payload.get("file_references", [])
        if isinstance(item, Mapping)
    )
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or dict(authority) != _authority_payload():
        raise ReconstructionValidationError("Persistent dense scene authority payload changed.")
    if str(payload.get("phase", "")) != DENSE_RGB_PERSISTENCE_PHASE:
        raise ReconstructionValidationError("Persistent dense scene phase marker is invalid.")
    return PersistentDenseRgbScene(
        scene_id=str(payload["scene_id"]),
        scene_fingerprint_sha256=str(payload["scene_fingerprint_sha256"]),
        created_at_utc=str(payload["created_at_utc"]),
        source_session_id=str(payload["source_session_id"]),
        reconstruction_frame_id=str(payload["reconstruction_frame_id"]),
        selected_workspace_orders=tuple(int(v) for v in payload.get("selected_workspace_orders", [])),
        selected_keyframe_ids=tuple(str(v) for v in payload.get("selected_keyframe_ids", [])),
        selected_view_count=int(payload["selected_view_count"]),
        voxel_size_reconstruction_units=float(payload["voxel_size_reconstruction_units"]),
        primary_point_count=int(payload["primary_point_count"]),
        core_point_count=int(payload["core_point_count"]),
        source_supported_point_count=int(payload["source_supported_point_count"]),
        source_all_voxel_count=int(payload["source_all_voxel_count"]),
        source_f_summary_sha256=str(payload["source_f_summary_sha256"]),
        source_e3_summary_sha256=str(payload["source_e3_summary_sha256"]),
        file_references=references,
        quality_snapshot=payload.get("quality_snapshot", {}),
        filter_policy_snapshot=payload.get("filter_policy_snapshot", {}),
        schema_version=int(payload.get("schema_version", 0)),
    )


def reference_for_role(scene: PersistentDenseRgbScene, role: DenseSceneFileRole) -> DenseSceneFileReference:
    if not isinstance(scene, PersistentDenseRgbScene):
        raise TypeError("scene must be PersistentDenseRgbScene.")
    if not isinstance(role, DenseSceneFileRole):
        raise TypeError("role must be DenseSceneFileRole.")
    for item in scene.file_references:
        if item.role is role:
            return item
    raise ReconstructionValidationError(f"Persistent dense scene lacks role: {role.value}")


def resolve_dense_scene_reference(
    manifest_or_dir: str | Path,
    scene: PersistentDenseRgbScene,
    role: DenseSceneFileRole,
) -> Path:
    target = Path(manifest_or_dir).expanduser().resolve()
    root = target if target.is_dir() else target.parent
    reference = reference_for_role(scene, role)
    path = (root / reference.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReconstructionValidationError("Persistent dense scene reference escapes the scene directory.") from exc
    return path


def verify_persistent_dense_scene_files(
    scene: PersistentDenseRgbScene,
    manifest_or_dir: str | Path,
) -> tuple[DenseSceneFileReference, ...]:
    if not isinstance(scene, PersistentDenseRgbScene):
        raise TypeError("scene must be PersistentDenseRgbScene.")
    for reference in scene.file_references:
        path = resolve_dense_scene_reference(manifest_or_dir, scene, reference.role)
        if not path.is_file():
            raise ReconstructionValidationError(f"Persistent dense scene referenced file is missing: {path}")
        if path.stat().st_size != reference.size_bytes:
            raise ReconstructionValidationError(f"Persistent dense scene referenced file size changed: {path}")
        if _sha256_file(path) != reference.sha256:
            raise ReconstructionValidationError(f"Persistent dense scene referenced file checksum changed: {path}")
    return scene.file_references


def write_persistent_dense_rgb_scene(scene: PersistentDenseRgbScene, output_dir: str | Path) -> Path:
    if not isinstance(scene, PersistentDenseRgbScene):
        raise TypeError("scene must be PersistentDenseRgbScene.")
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Persistent dense scene output_dir must be absolute.")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / DENSE_RGB_SCENE_MANIFEST
    payload = json.dumps(scene.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    manifest.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (root / DENSE_RGB_SCENE_CHECKSUM).write_text(f"{digest}  {DENSE_RGB_SCENE_MANIFEST}\n", encoding="utf-8")
    return manifest


def read_persistent_dense_rgb_scene(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> PersistentDenseRgbScene:
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        target = target / DENSE_RGB_SCENE_MANIFEST
    if not target.is_file():
        raise ReconstructionValidationError(f"Persistent dense RGB scene manifest does not exist: {target}")
    checksum = target.parent / DENSE_RGB_SCENE_CHECKSUM
    if not checksum.is_file():
        raise ReconstructionValidationError("Persistent dense RGB scene checksum sidecar is missing.")
    parts = checksum.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[1] != DENSE_RGB_SCENE_MANIFEST:
        raise ReconstructionValidationError("Persistent dense RGB scene checksum sidecar is malformed.")
    expected = require_sha256(parts[0], field_name="manifest_sha256", allow_empty=False)
    if _sha256_file(target) != expected:
        raise ReconstructionValidationError("Persistent dense RGB scene manifest checksum mismatch.")
    payload = _read_json(target)
    scene = _scene_from_mapping(payload)
    if verify_files:
        verify_persistent_dense_scene_files(scene, target)
    return scene


def restore_persistent_dense_geometry(
    path: str | Path,
    *,
    representation: str = "extended",
) -> RestoredDenseRgbGeometry:
    scene = read_persistent_dense_rgb_scene(path, verify_files=True)
    kind = str(representation).strip().lower()
    if kind == "extended":
        role = DenseSceneFileRole.PRIMARY_EXTENDED_NPZ
        expected_count = scene.primary_point_count
    elif kind == "core":
        role = DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ
        expected_count = scene.core_point_count
    else:
        raise ReconstructionValidationError("representation must be 'extended' or 'core'.")
    reference = reference_for_role(scene, role)
    npz_path = resolve_dense_scene_reference(path, scene, role)
    inspected = _inspect_filtered_npz(npz_path, representation=kind)
    if inspected["point_count"] != expected_count:
        raise ReconstructionValidationError("Restored dense geometry point count disagrees with manifest.")
    if not np.isclose(inspected["voxel_size"], scene.voxel_size_reconstruction_units, rtol=0.0, atol=1e-12):
        raise ReconstructionValidationError("Restored dense geometry voxel size disagrees with manifest.")
    if inspected["orders"] != scene.selected_workspace_orders or inspected["keyframes"] != scene.selected_keyframe_ids:
        raise ReconstructionValidationError("Restored dense geometry selected-view provenance disagrees with manifest.")
    with np.load(npz_path, allow_pickle=False) as payload:
        return RestoredDenseRgbGeometry(
            scene_id=scene.scene_id,
            representation=kind,
            points_reconstruction=np.asarray(payload["points_reconstruction"], dtype=np.float64),
            colors_rgb=np.asarray(payload["colors_rgb"], dtype=np.uint8),
            view_observation_counts=np.asarray(payload["view_observation_counts"], dtype=np.uint32),
            filter_confidence=np.asarray(payload["filter_confidence"], dtype=np.float32),
            voxel_size_reconstruction_units=float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
            selected_workspace_orders=scene.selected_workspace_orders,
            selected_keyframe_ids=scene.selected_keyframe_ids,
            source_file_sha256=reference.sha256,
        )


def persist_filtered_dense_rgb_scene(
    f_summary_path: str | Path,
    output_dir: str | Path,
    *,
    created_at_utc: str | None = None,
) -> DenseScenePersistenceResult:
    """Persist a passing Phase-F dense scene as one portable canonical bundle."""

    f_path = Path(f_summary_path).expanduser().resolve()
    if not f_path.is_file():
        raise ReconstructionValidationError(f"G F summary does not exist: {f_path}")
    f_summary = _read_json(f_path)
    _validate_f_summary(f_summary)

    e3_dir = Path(str(f_summary["source_e3_dir"])).expanduser().resolve()
    e3_path = e3_dir / "e3_summary.json"
    if not e3_path.is_file():
        raise ReconstructionValidationError("G requires the source E3 summary while creating the persistent bundle.")
    e3_summary = _read_json(e3_path)
    _validate_e3_summary(e3_summary, f_summary)

    artifacts = f_summary["artifacts"]
    assert isinstance(artifacts, Mapping)
    extended_source = Path(str(artifacts["extended_npz"])).expanduser().resolve()
    core_source = Path(str(artifacts["core_npz"])).expanduser().resolve()
    extended = _inspect_filtered_npz(extended_source, representation="extended")
    core = _inspect_filtered_npz(core_source, representation="core")

    classification = f_summary["classification"]
    assert isinstance(classification, Mapping)
    if extended["point_count"] != int(classification["extended_voxels"]):
        raise ReconstructionValidationError("G F Extended point count disagrees with F summary.")
    if core["point_count"] != int(classification["core_voxels"]):
        raise ReconstructionValidationError("G F Core point count disagrees with F summary.")
    if not np.isclose(extended["voxel_size"], core["voxel_size"], rtol=0.0, atol=1e-12):
        raise ReconstructionValidationError("G Core/Extended voxel sizes disagree.")
    if not np.isclose(extended["voxel_size"], float(f_summary["voxel_size_reconstruction_units"]), rtol=0.0, atol=1e-12):
        raise ReconstructionValidationError("G F geometry voxel size disagrees with F summary.")
    expected_orders = tuple(int(v) for v in f_summary["selected_workspace_orders"])
    expected_keyframes = tuple(str(v) for v in f_summary["selected_keyframe_ids"])
    if not (extended["orders"] == core["orders"] == expected_orders):
        raise ReconstructionValidationError("G Core/Extended workspace-order provenance disagrees.")
    if not (extended["keyframes"] == core["keyframes"] == expected_keyframes):
        raise ReconstructionValidationError("G Core/Extended keyframe provenance disagrees.")
    extended_keys = _voxel_key_set(extended["points"], extended["voxel_size"])
    core_keys = _voxel_key_set(core["points"], core["voxel_size"])
    if len(extended_keys) != extended["point_count"] or len(core_keys) != core["point_count"]:
        raise ReconstructionValidationError("G expects one retained point per fused voxel key.")
    if not core_keys.issubset(extended_keys):
        raise ReconstructionValidationError("G requires Core geometry to be a voxel subset of Extended geometry.")

    extended_ply = Path(str(artifacts["extended_ply"])).expanduser().resolve()
    core_ply = Path(str(artifacts["core_ply"])).expanduser().resolve()
    if _ply_vertex_count(extended_ply) != extended["point_count"]:
        raise ReconstructionValidationError("G Extended PLY vertex count disagrees with NPZ geometry.")
    if _ply_vertex_count(core_ply) != core["point_count"]:
        raise ReconstructionValidationError("G Core PLY vertex count disagrees with NPZ geometry.")

    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("G output_dir must be absolute.")
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ReconstructionValidationError("G refuses to overwrite a non-empty persistent dense scene directory.")
    if root.exists():
        root.rmdir()
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))).resolve()

    try:
        references: list[DenseSceneFileReference] = []
        for artifact_name, role in _F_ARTIFACT_TO_ROLE.items():
            references.append(
                _copy_reference(stage, Path(str(artifacts[artifact_name])).expanduser().resolve(), role)
            )
        references.append(_copy_reference(stage, f_path, DenseSceneFileRole.SOURCE_F_SUMMARY))
        references.append(_copy_reference(stage, e3_path, DenseSceneFileRole.SOURCE_E3_SUMMARY))
        references.sort(key=lambda item: item.role.value)

        f_sha = _sha256_file(f_path)
        e3_sha = _sha256_file(e3_path)
        source_session_id = str(e3_summary["source_sfm_session_id"])
        quality_snapshot = {
            "classification": f_summary.get("classification", {}),
            "confidence": f_summary.get("confidence", {}),
            "dispersion": f_summary.get("dispersion", {}),
            "topology": f_summary.get("topology", {}),
            "pass_conditions": f_summary.get("pass_conditions", {}),
        }
        policy_snapshot = f_summary.get("policy", {})
        created = created_at_utc or datetime.now(timezone.utc).isoformat()
        fingerprint_payload = _scene_fingerprint_payload(
            source_session_id=source_session_id,
            reconstruction_frame_id=DENSE_RGB_RECONSTRUCTION_FRAME_ID,
            selected_workspace_orders=expected_orders,
            selected_keyframe_ids=expected_keyframes,
            selected_view_count=len(expected_orders),
            voxel_size_reconstruction_units=float(extended["voxel_size"]),
            primary_point_count=int(extended["point_count"]),
            core_point_count=int(core["point_count"]),
            source_supported_point_count=int(classification["source_supported_voxels"]),
            source_all_voxel_count=int(classification["source_all_voxels"]),
            source_f_summary_sha256=f_sha,
            source_e3_summary_sha256=e3_sha,
            file_references=references,
            quality_snapshot=quality_snapshot,
            filter_policy_snapshot=policy_snapshot,
        )
        fingerprint = hashlib.sha256(_canonical_json(fingerprint_payload)).hexdigest()
        scene = PersistentDenseRgbScene(
            scene_id=f"pdrgb-{fingerprint[:24]}",
            scene_fingerprint_sha256=fingerprint,
            created_at_utc=created,
            source_session_id=source_session_id,
            reconstruction_frame_id=DENSE_RGB_RECONSTRUCTION_FRAME_ID,
            selected_workspace_orders=expected_orders,
            selected_keyframe_ids=expected_keyframes,
            selected_view_count=len(expected_orders),
            voxel_size_reconstruction_units=float(extended["voxel_size"]),
            primary_point_count=int(extended["point_count"]),
            core_point_count=int(core["point_count"]),
            source_supported_point_count=int(classification["source_supported_voxels"]),
            source_all_voxel_count=int(classification["source_all_voxels"]),
            source_f_summary_sha256=f_sha,
            source_e3_summary_sha256=e3_sha,
            file_references=tuple(references),
            quality_snapshot=quality_snapshot,
            filter_policy_snapshot=policy_snapshot,
        )
        manifest_stage = write_persistent_dense_rgb_scene(scene, stage)
        verify_persistent_dense_scene_files(scene, manifest_stage)
        os.replace(stage, root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    manifest = root / DENSE_RGB_SCENE_MANIFEST
    loaded = read_persistent_dense_rgb_scene(manifest, verify_files=True)
    restored_extended = restore_persistent_dense_geometry(manifest, representation="extended")
    restored_core = restore_persistent_dense_geometry(manifest, representation="core")
    source_independent = all(
        resolve_dense_scene_reference(manifest, loaded, reference.role).is_relative_to(root)
        for reference in loaded.file_references
    )
    portable = all(not PurePosixPath(reference.relative_path).is_absolute() for reference in loaded.file_references)
    return DenseScenePersistenceResult(
        scene=loaded,
        manifest_path=manifest,
        restored_extended=restored_extended,
        restored_core=restored_core,
        source_independent=source_independent,
        portable_relative_references=portable,
    )


__all__ = [
    "DENSE_RGB_PERSISTENCE_PHASE",
    "DENSE_RGB_PERSISTENCE_SCHEMA_VERSION",
    "DENSE_RGB_RECONSTRUCTION_FRAME_ID",
    "DENSE_RGB_SCENE_CHECKSUM",
    "DENSE_RGB_SCENE_MANIFEST",
    "DenseSceneFileReference",
    "DenseSceneFileRole",
    "DenseScenePersistenceResult",
    "PersistentDenseRgbScene",
    "RestoredDenseRgbGeometry",
    "persist_filtered_dense_rgb_scene",
    "read_persistent_dense_rgb_scene",
    "reference_for_role",
    "resolve_dense_scene_reference",
    "restore_persistent_dense_geometry",
    "verify_persistent_dense_scene_files",
    "write_persistent_dense_rgb_scene",
]
