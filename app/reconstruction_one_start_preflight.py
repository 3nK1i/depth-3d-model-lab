"""I6.2A dependency-light One-Start offline reconstruction preflight contract.

This module defines what one user-facing offline ``Start Reconstruction`` action
is allowed to launch before I6.2B wires that contract into the controller.  It
is intentionally fail-closed and descriptive: it does not start a worker,
create directories, release application resources, select a dense provider,
activate production Auto, persist a scene, or change reconstruction authority.

The contract revalidates source/runtime/output/storage evidence immediately
before launch.  The storage floor reuses the already-existing MVS house safety
floor rather than introducing a new hidden production threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import UUID

from core.reconstruction import ReconstructionQualityProfile
from core.reconstruction.mvs_environment import MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB
from core.reconstruction.sources import SUPPORTED_IMAGE_EXTENSIONS

CANONICAL_CAMERA_AUTHORITY_OWNER = "pycolmap_sfm"
PRODUCTION_AUTO_POLICY_SCOPE = "offline_shadow_only"
ONE_START_EXECUTION_SCOPE = "sparse_sfm_only"
ONE_START_MIN_FREE_DISK_BYTES = int(MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB * 1024**3)

_SUPPORTED_OFFLINE_WORKFLOWS = frozenset({"image_sequence", "recorded_video"})
_SUPPORTED_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


class OneStartPreflightStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


class OneStartBlockerCode(str, Enum):
    CONTROLLER_SHUTDOWN = "controller_shutdown"
    LIVE_RECONSTRUCTION_ACTIVE = "live_reconstruction_active"
    OFFLINE_RECONSTRUCTION_ACTIVE = "offline_reconstruction_active"
    DA3_INTEGRATION_ACTIVE = "da3_integration_active"
    UNSUPPORTED_WORKFLOW = "unsupported_workflow"
    SOURCE_NOT_READY = "source_not_ready"
    SOURCE_PATH_MISSING = "source_path_missing"
    SOURCE_PATH_INVALID = "source_path_invalid"
    IMAGE_COUNT_TOO_SMALL = "image_count_too_small"
    VIDEO_EXTENSION_UNSUPPORTED = "video_extension_unsupported"
    PYCOLMAP_RUNTIME_MISSING = "pycolmap_runtime_missing"
    PYCOLMAP_RUNTIME_NOT_EXECUTABLE = "pycolmap_runtime_not_executable"
    OUTPUT_ROOT_INVALID = "output_root_invalid"
    OUTPUT_ROOT_NOT_WRITABLE = "output_root_not_writable"
    OUTPUT_DIR_NOT_EMPTY = "output_dir_not_empty"
    DISK_SAFETY_FLOOR_NOT_MET = "disk_safety_floor_not_met"
    APPLICATION_RESOURCE_CONFLICT = "application_resource_conflict"


@dataclass(frozen=True, slots=True)
class OneStartBlocker:
    code: OneStartBlockerCode
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, OneStartBlockerCode):
            raise TypeError("code must be OneStartBlockerCode.")
        detail = str(self.detail).strip()
        if not detail:
            raise ValueError("One-Start blocker detail must be non-empty.")
        object.__setattr__(self, "detail", detail)

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class OfflineOneStartPreflight:
    """Immutable admission result for one future offline Start action."""

    status: OneStartPreflightStatus
    blockers: tuple[OneStartBlocker, ...]
    workflow: str
    source_path: str
    source_item_count: int
    session_id: str
    profile: str
    sfm_python: str
    output_root: str
    proposed_output_dir: str
    disk_free_bytes: int
    minimum_free_disk_bytes: int
    application_resource_guard_required: bool
    camera_authority_owner: str
    execution_scope: str
    dense_execution_authorized: bool
    persistent_scene_promotion_authorized: bool
    certified_metrology: bool
    production_auto_policy_scope: str
    production_auto_qualified: bool
    production_auto_authorized: bool
    hardware_adaptation_owner: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, OneStartPreflightStatus):
            raise TypeError("status must be OneStartPreflightStatus.")
        blockers = tuple(self.blockers)
        if any(not isinstance(item, OneStartBlocker) for item in blockers):
            raise TypeError("blockers must contain OneStartBlocker values.")
        object.__setattr__(self, "blockers", blockers)
        if (self.status is OneStartPreflightStatus.READY) != (not blockers):
            raise ValueError("One-Start status must be READY exactly when no blockers exist.")

        sid = _canonical_uuid(self.session_id, field_name="session_id")
        object.__setattr__(self, "session_id", sid)
        object.__setattr__(self, "workflow", str(self.workflow).strip())
        object.__setattr__(self, "source_path", _path_text(self.source_path))
        object.__setattr__(self, "sfm_python", _path_text(self.sfm_python))
        object.__setattr__(self, "output_root", _path_text(self.output_root))
        object.__setattr__(self, "proposed_output_dir", _path_text(self.proposed_output_dir))

        count = int(self.source_item_count)
        free = int(self.disk_free_bytes)
        minimum = int(self.minimum_free_disk_bytes)
        if count < 0:
            raise ValueError("source_item_count must be non-negative.")
        if free < 0 or minimum < 0:
            raise ValueError("disk byte counts must be non-negative.")
        object.__setattr__(self, "source_item_count", count)
        object.__setattr__(self, "disk_free_bytes", free)
        object.__setattr__(self, "minimum_free_disk_bytes", minimum)

        if self.camera_authority_owner != CANONICAL_CAMERA_AUTHORITY_OWNER:
            raise ValueError("I6.2A must preserve PyCOLMAP as canonical camera authority.")
        if self.execution_scope != ONE_START_EXECUTION_SCOPE:
            raise ValueError("I6.2A may only authorize the currently qualified sparse SfM scope.")
        if bool(self.dense_execution_authorized):
            raise ValueError("I6.2A cannot authorize dense execution.")
        if bool(self.persistent_scene_promotion_authorized):
            raise ValueError("I6.2A cannot promote sparse output to a persistent scene.")
        if bool(self.certified_metrology):
            raise ValueError("I6.2A cannot claim certified metrology.")
        if self.production_auto_policy_scope != PRODUCTION_AUTO_POLICY_SCOPE:
            raise ValueError("I6.2A must preserve the offline/shadow Auto policy scope.")
        if bool(self.production_auto_qualified) or bool(self.production_auto_authorized):
            raise ValueError("I6.2A cannot qualify or authorize production Auto.")
        if not bool(self.application_resource_guard_required):
            raise ValueError("I6.2A requires the existing application resource guard at launch commit.")
        if self.hardware_adaptation_owner != "offline_worker_adaptive_pre_sfm":
            raise ValueError("I6.2A must preserve worker-owned hardware adaptation.")

        object.__setattr__(self, "dense_execution_authorized", False)
        object.__setattr__(self, "persistent_scene_promotion_authorized", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "production_auto_qualified", False)
        object.__setattr__(self, "production_auto_authorized", False)
        object.__setattr__(self, "application_resource_guard_required", True)

    @property
    def ready(self) -> bool:
        return self.status is OneStartPreflightStatus.READY

    @property
    def blocker_codes(self) -> tuple[OneStartBlockerCode, ...]:
        return tuple(item.code for item in self.blockers)

    @property
    def blocker_details(self) -> str:
        return " | ".join(item.detail for item in self.blockers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "ready": self.ready,
            "blockers": [item.as_dict() for item in self.blockers],
            "source": {
                "workflow": self.workflow,
                "path": self.source_path,
                "item_count": self.source_item_count,
            },
            "launch": {
                "session_id": self.session_id,
                "profile": self.profile,
                "sfm_python": self.sfm_python,
                "output_root": self.output_root,
                "proposed_output_dir": self.proposed_output_dir,
                "application_resource_guard_required": True,
                "hardware_adaptation_owner": self.hardware_adaptation_owner,
            },
            "storage": {
                "free_bytes": self.disk_free_bytes,
                "minimum_free_bytes": self.minimum_free_disk_bytes,
                "safety_floor_met": self.disk_free_bytes >= self.minimum_free_disk_bytes,
            },
            "authority": {
                "camera_authority_owner": self.camera_authority_owner,
                "certified_metrology": False,
            },
            "execution": {
                "scope": self.execution_scope,
                "dense_execution_authorized": False,
                "persistent_scene_promotion_authorized": False,
            },
            "production_auto": {
                "policy_scope": self.production_auto_policy_scope,
                "qualified": False,
                "authorized": False,
            },
        }


def build_offline_one_start_preflight(
    *,
    source_selection: Any,
    controller_shutdown: bool,
    live_active: bool,
    offline_active: bool,
    da3_active: bool,
    sfm_python: str | Path,
    output_root: str | Path,
    profile: ReconstructionQualityProfile,
    session_id: str,
    disk_free_bytes: int | None = None,
    minimum_free_disk_bytes: int = ONE_START_MIN_FREE_DISK_BYTES,
    application_resource_guard_blocker: str | None = None,
) -> OfflineOneStartPreflight:
    """Build a fail-closed, non-executing preflight for the current offline source."""

    if not isinstance(profile, ReconstructionQualityProfile):
        raise TypeError("profile must be ReconstructionQualityProfile.")
    sid = _canonical_uuid(session_id, field_name="session_id")
    sfm = Path(sfm_python).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    proposed = root / sid
    minimum = int(minimum_free_disk_bytes)
    if minimum < 0:
        raise ValueError("minimum_free_disk_bytes must be non-negative.")

    blockers: list[OneStartBlocker] = []
    if bool(controller_shutdown):
        blockers.append(_block(OneStartBlockerCode.CONTROLLER_SHUTDOWN, "Reconstruction controller is shut down."))
    if bool(live_active):
        blockers.append(_block(OneStartBlockerCode.LIVE_RECONSTRUCTION_ACTIVE, "Live reconstruction is active; stop it before starting offline reconstruction."))
    if bool(offline_active):
        blockers.append(_block(OneStartBlockerCode.OFFLINE_RECONSTRUCTION_ACTIVE, "An offline reconstruction job is already active or finalizing."))
    if bool(da3_active):
        blockers.append(_block(OneStartBlockerCode.DA3_INTEGRATION_ACTIVE, "DA3 canonical integration is active; cancel it before starting offline reconstruction."))

    workflow = _value(getattr(source_selection, "workflow", None)) or "none"
    source_ready = bool(getattr(source_selection, "ready", False))
    source_raw = getattr(source_selection, "path", None)
    source = Path(source_raw).expanduser().resolve() if source_raw is not None else None
    source_count = 0

    if workflow not in _SUPPORTED_OFFLINE_WORKFLOWS:
        blockers.append(_block(OneStartBlockerCode.UNSUPPORTED_WORKFLOW, "One Start offline reconstruction requires Image sequence or Recorded video."))
    if not source_ready:
        detail = str(getattr(source_selection, "detail", "")).strip() or "The selected offline source is not ready."
        blockers.append(_block(OneStartBlockerCode.SOURCE_NOT_READY, detail))

    if source is None or not source.exists():
        blockers.append(_block(OneStartBlockerCode.SOURCE_PATH_MISSING, "The selected offline source path no longer exists."))
    elif workflow == "image_sequence":
        if not source.is_dir():
            blockers.append(_block(OneStartBlockerCode.SOURCE_PATH_INVALID, "The selected image-sequence source is no longer a directory."))
        else:
            source_count = sum(
                1
                for item in source.iterdir()
                if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
            )
            if source_count < 3:
                blockers.append(_block(OneStartBlockerCode.IMAGE_COUNT_TOO_SMALL, f"Image-sequence reconstruction requires at least 3 supported top-level images; found {source_count}."))
    elif workflow == "recorded_video":
        if not source.is_file():
            blockers.append(_block(OneStartBlockerCode.SOURCE_PATH_INVALID, "The selected recorded-video source is no longer a file."))
        else:
            source_count = 1
            if source.suffix.casefold() not in _SUPPORTED_VIDEO_EXTENSIONS:
                blockers.append(_block(OneStartBlockerCode.VIDEO_EXTENSION_UNSUPPORTED, f"Recorded-video extension '{source.suffix or '<none>'}' is not supported."))

    if not sfm.is_file():
        blockers.append(_block(OneStartBlockerCode.PYCOLMAP_RUNTIME_MISSING, f"Isolated PyCOLMAP interpreter is missing: {sfm}"))
    elif not os.access(sfm, os.X_OK):
        blockers.append(_block(OneStartBlockerCode.PYCOLMAP_RUNTIME_NOT_EXECUTABLE, f"Isolated PyCOLMAP interpreter is not executable: {sfm}"))

    writable_parent = _nearest_existing_parent(root)
    if root.exists() and not root.is_dir():
        blockers.append(_block(OneStartBlockerCode.OUTPUT_ROOT_INVALID, f"Reconstruction output root is not a directory: {root}"))
    elif writable_parent is None:
        blockers.append(_block(OneStartBlockerCode.OUTPUT_ROOT_INVALID, f"No existing parent is available for reconstruction output root: {root}"))
    elif not os.access(writable_parent, os.W_OK | os.X_OK):
        blockers.append(_block(OneStartBlockerCode.OUTPUT_ROOT_NOT_WRITABLE, f"Reconstruction output root is not writable through parent: {writable_parent}"))

    if proposed.exists():
        if not proposed.is_dir():
            blockers.append(_block(OneStartBlockerCode.OUTPUT_ROOT_INVALID, f"Proposed reconstruction output path is not a directory: {proposed}"))
        elif any(proposed.iterdir()):
            blockers.append(_block(OneStartBlockerCode.OUTPUT_DIR_NOT_EMPTY, "Proposed reconstruction output directory is not empty; stale-result reuse is forbidden."))

    if disk_free_bytes is None:
        free = observe_offline_one_start_disk_free_bytes(root)
    else:
        free = int(disk_free_bytes)
        if free < 0:
            raise ValueError("disk_free_bytes must be non-negative.")
    if free < minimum:
        blockers.append(_block(OneStartBlockerCode.DISK_SAFETY_FLOOR_NOT_MET, f"Reconstruction storage safety floor is not met: {free} free bytes < {minimum} required bytes."))

    guard_detail = str(application_resource_guard_blocker or "").strip()
    if guard_detail:
        blockers.append(_block(OneStartBlockerCode.APPLICATION_RESOURCE_CONFLICT, guard_detail))

    status = OneStartPreflightStatus.READY if not blockers else OneStartPreflightStatus.BLOCKED
    return OfflineOneStartPreflight(
        status=status,
        blockers=tuple(blockers),
        workflow=workflow,
        source_path=str(source or ""),
        source_item_count=source_count,
        session_id=sid,
        profile=profile.value,
        sfm_python=str(sfm),
        output_root=str(root),
        proposed_output_dir=str(proposed),
        disk_free_bytes=free,
        minimum_free_disk_bytes=minimum,
        application_resource_guard_required=True,
        camera_authority_owner=CANONICAL_CAMERA_AUTHORITY_OWNER,
        execution_scope=ONE_START_EXECUTION_SCOPE,
        dense_execution_authorized=False,
        persistent_scene_promotion_authorized=False,
        certified_metrology=False,
        production_auto_policy_scope=PRODUCTION_AUTO_POLICY_SCOPE,
        production_auto_qualified=False,
        production_auto_authorized=False,
        hardware_adaptation_owner="offline_worker_adaptive_pre_sfm",
    )


def observe_offline_one_start_disk_free_bytes(output_root: str | Path) -> int:
    """Observe free storage without creating or filling reconstruction paths."""

    root = Path(output_root).expanduser().resolve()
    parent = _nearest_existing_parent(root)
    if parent is None:
        return 0
    try:
        return int(shutil.disk_usage(parent).free)
    except OSError:
        return 0


def offline_one_start_transition_allowed(current: Any, target: Any) -> bool:
    """Return whether a runtime transition matches the existing offline controller."""

    current_value = _value(current)
    target_value = _value(target)
    allowed = {
        "idle": frozenset({"starting"}),
        "completed": frozenset({"starting"}),
        "failed": frozenset({"starting"}),
        "cancelled": frozenset({"starting"}),
        "starting": frozenset({"running", "failed"}),
        "running": frozenset({"cancelling", "completed", "failed"}),
        "cancelling": frozenset({"cancelled"}),
    }
    return target_value in allowed.get(current_value, frozenset())


def _block(code: OneStartBlockerCode, detail: str) -> OneStartBlocker:
    return OneStartBlocker(code=code, detail=detail)


def _canonical_uuid(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    try:
        return str(UUID(text))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID.") from exc


def _value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().casefold() if raw is not None else ""


def _path_text(value: str | Path) -> str:
    text = str(value).strip()
    return str(Path(text).expanduser().resolve()) if text else ""


def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = path
    while True:
        if candidate.exists():
            return candidate if candidate.is_dir() else candidate.parent
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent
