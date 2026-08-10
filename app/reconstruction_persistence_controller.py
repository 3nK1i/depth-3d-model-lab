"""Application bridge for reconstruction scene persistence and resume contracts.

Phase 8.8A4 does not invent a second persistence format.  It exposes the
already-qualified Phase 8.7F5 ``PersistentReconstructionScene`` and Phase
8.7G2 whole-engine checkpoint/resume contracts to the PyQt application while
keeping Phase 8.8 UI navigation disabled.

A Phase 8.8A3 offline result is intentionally treated as *partial* while it
contains sparse SfM/camera-pose evidence only.  It is never promoted to a
persistent scene unless a future integrated offline worker explicitly produces
a valid F5 persistent-scene manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Collection, Mapping, Sequence
from uuid import UUID

from PyQt6.QtCore import QObject, pyqtSignal

from core.reconstruction import (
    PersistentReconstructionScene,
    PersistentReconstructionSceneState,
    ReconstructionEngineCheckpoint,
    ReconstructionPipelinePlan,
    ReconstructionPipelineRunResult,
    ReconstructionProviderDescriptor,
    ReconstructionResumeCompatibilityResult,
    ReconstructionStageRunResult,
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    read_reconstruction_engine_checkpoint,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
    write_reconstruction_engine_checkpoint,
)
from core.reconstruction.errors import ReconstructionValidationError


class ReconstructionSceneAvailability(str, Enum):
    """Application-facing availability, not a replacement engine lifecycle."""

    NONE = "none"
    PARTIAL = "partial"
    READY = "ready"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class ReconstructionResumePreparation:
    """Validated resume prefix ready for the existing pipeline orchestrator."""

    checkpoint: ReconstructionEngineCheckpoint
    compatibility: ReconstructionResumeCompatibilityResult

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, ReconstructionEngineCheckpoint):
            raise TypeError("checkpoint must be ReconstructionEngineCheckpoint.")
        if not isinstance(self.compatibility, ReconstructionResumeCompatibilityResult):
            raise TypeError("compatibility must be ReconstructionResumeCompatibilityResult.")
        self.compatibility.require_compatible()

    @property
    def resume_stage_results(self) -> tuple[ReconstructionStageRunResult, ...]:
        return self.checkpoint.stage_results

    @property
    def next_stage(self):
        return self.checkpoint.next_stage

    @property
    def reused_stage_count(self) -> int:
        return self.compatibility.reused_stage_count


class ReconstructionPersistenceController(QObject):
    """Own application references to partial results, scenes and checkpoints."""

    offline_result_changed = pyqtSignal(object, str)
    scene_availability_changed = pyqtSignal(object)  # ReconstructionSceneAvailability
    persistent_scene_changed = pyqtSignal(object, str)  # scene or None, manifest path
    checkpoint_changed = pyqtSignal(object, str)  # checkpoint or None, manifest path
    resume_compatibility_changed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._offline_result: dict[str, object] | None = None
        self._offline_result_path: Path | None = None
        self._scene: PersistentReconstructionScene | None = None
        self._scene_manifest: Path | None = None
        self._checkpoint: ReconstructionEngineCheckpoint | None = None
        self._checkpoint_manifest: Path | None = None
        self._resume_compatibility: ReconstructionResumeCompatibilityResult | None = None
        self._availability = ReconstructionSceneAvailability.NONE
        self._shutdown_started = False

    @property
    def offline_result(self) -> dict[str, object] | None:
        return None if self._offline_result is None else dict(self._offline_result)

    @property
    def offline_result_path(self) -> Path | None:
        return self._offline_result_path

    @property
    def scene(self) -> PersistentReconstructionScene | None:
        return self._scene

    @property
    def scene_manifest(self) -> Path | None:
        return self._scene_manifest

    @property
    def scene_availability(self) -> ReconstructionSceneAvailability:
        return self._availability

    @property
    def checkpoint(self) -> ReconstructionEngineCheckpoint | None:
        return self._checkpoint

    @property
    def checkpoint_manifest(self) -> Path | None:
        return self._checkpoint_manifest

    @property
    def resume_compatibility(self) -> ReconstructionResumeCompatibilityResult | None:
        return self._resume_compatibility

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_started

    def accept_offline_result(self, payload: Mapping[str, object], output_dir: str | Path) -> None:
        """Record a worker result without silently promoting sparse SfM to a scene."""

        self._require_active()
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        copied = dict(payload)
        session_id = str(copied.get("session_id", "")).strip()
        self._require_uuid(session_id, field_name="offline result session_id")
        root = Path(output_dir).expanduser()
        if not root.is_absolute():
            raise ReconstructionValidationError("Offline result output_dir must be absolute.")
        root = root.resolve()
        result_path = root / "offline_result.json"
        self._offline_result = copied
        self._offline_result_path = result_path

        status = str(copied.get("status", "")).strip().lower()
        persistent_ready = bool(copied.get("persistent_scene_ready", False))
        if persistent_ready:
            if not bool(copied.get("canonical_geometry_authoritative", False)):
                raise ReconstructionValidationError(
                    "An offline result cannot claim a persistent scene without canonical geometry authority."
                )
            if bool(copied.get("certified_metrology", False)):
                raise ReconstructionValidationError(
                    "Phase 8 reconstruction persistence cannot silently claim certified metrology."
                )
            manifest = self._persistent_manifest_from_result(copied)
            scene = self.load_scene(
                manifest,
                expected_session_id=session_id,
                verify_file_references=True,
            )
            if scene.session_id != session_id:  # defensive; load_scene already enforces this
                raise ReconstructionValidationError("Persistent scene session identity mismatch.")
        elif self._scene is None:
            availability = (
                ReconstructionSceneAvailability.PARTIAL
                if status == "completed" and bool(copied.get("camera_pose_available", False))
                else ReconstructionSceneAvailability.NONE
            )
            self._set_availability(availability)

        self.offline_result_changed.emit(dict(copied), str(result_path))

    def adopt_pipeline_result(
        self,
        result: ReconstructionPipelineRunResult,
        output_dir: str | Path,
        *,
        payload_snapshots_by_kind: Mapping[str, Mapping[str, object]] | None = None,
    ) -> PersistentReconstructionScene:
        """Materialize an already-completed integrated pipeline via the F5 contract."""

        self._require_active()
        if not isinstance(result, ReconstructionPipelineRunResult):
            raise TypeError("result must be ReconstructionPipelineRunResult.")
        scene = persistent_scene_from_pipeline_result(
            result,
            payload_snapshots_by_kind=payload_snapshots_by_kind,
        )
        manifest = write_persistent_reconstruction_scene(scene, output_dir)
        return self._set_scene(scene, manifest)

    def load_scene(
        self,
        path: str | Path,
        *,
        expected_session_id: str | None = None,
        verify_file_references: bool = True,
    ) -> PersistentReconstructionScene:
        self._require_active()
        scene = read_persistent_reconstruction_scene(path)
        if expected_session_id is not None:
            expected = self._require_uuid(expected_session_id, field_name="expected_session_id")
            if scene.session_id != expected:
                raise ReconstructionValidationError(
                    "Persistent scene session identity does not match the expected reconstruction session."
                )
        if verify_file_references:
            verify_persistent_scene_file_references(scene)
        manifest = self._normalize_scene_manifest(path)
        return self._set_scene(scene, manifest)

    def save_scene(self, output_dir: str | Path) -> Path:
        self._require_active()
        if self._scene is None:
            raise ReconstructionValidationError("No persistent reconstruction scene is loaded.")
        manifest = write_persistent_reconstruction_scene(self._scene, output_dir)
        self._scene_manifest = manifest.resolve()
        self.persistent_scene_changed.emit(self._scene, str(self._scene_manifest))
        return self._scene_manifest

    def clear_scene(self) -> None:
        self._scene = None
        self._scene_manifest = None
        next_availability = (
            ReconstructionSceneAvailability.PARTIAL
            if self._offline_result is not None
            and str(self._offline_result.get("status", "")).lower() == "completed"
            and bool(self._offline_result.get("camera_pose_available", False))
            and not bool(self._offline_result.get("persistent_scene_ready", False))
            else ReconstructionSceneAvailability.NONE
        )
        self._set_availability(next_availability)
        self.persistent_scene_changed.emit(None, "")

    def create_checkpoint(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        stage_results: Sequence[ReconstructionStageRunResult],
        output_dir: str | Path,
    ) -> ReconstructionEngineCheckpoint:
        self._require_active()
        checkpoint = create_reconstruction_engine_checkpoint(
            plan=plan,
            providers=providers,
            stage_results=stage_results,
        )
        manifest = write_reconstruction_engine_checkpoint(checkpoint, output_dir)
        return self._set_checkpoint(checkpoint, manifest)

    def load_checkpoint(self, path: str | Path) -> ReconstructionEngineCheckpoint:
        self._require_active()
        checkpoint = read_reconstruction_engine_checkpoint(path)
        manifest = self._normalize_checkpoint_manifest(path)
        return self._set_checkpoint(checkpoint, manifest)

    def check_resume_compatibility(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        available_payload_refs: Collection[str],
    ) -> ReconstructionResumeCompatibilityResult:
        self._require_active()
        if self._checkpoint is None:
            raise ReconstructionValidationError("No reconstruction engine checkpoint is loaded.")
        compatibility = check_reconstruction_engine_resume_compatibility(
            self._checkpoint,
            plan=plan,
            providers=providers,
            available_payload_refs=available_payload_refs,
        )
        self._resume_compatibility = compatibility
        self.resume_compatibility_changed.emit(compatibility)
        return compatibility

    def prepare_resume(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        available_payload_refs: Collection[str],
    ) -> ReconstructionResumePreparation:
        compatibility = self.check_resume_compatibility(
            plan=plan,
            providers=providers,
            available_payload_refs=available_payload_refs,
        )
        compatibility.require_compatible()
        assert self._checkpoint is not None
        return ReconstructionResumePreparation(
            checkpoint=self._checkpoint,
            compatibility=compatibility,
        )

    def clear_checkpoint(self) -> None:
        self._checkpoint = None
        self._checkpoint_manifest = None
        self._resume_compatibility = None
        self.checkpoint_changed.emit(None, "")

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._resume_compatibility = None

    def _set_scene(
        self,
        scene: PersistentReconstructionScene,
        manifest: Path,
    ) -> PersistentReconstructionScene:
        self._scene = scene
        self._scene_manifest = manifest.resolve()
        availability = (
            ReconstructionSceneAvailability.DEGRADED
            if scene.state is PersistentReconstructionSceneState.DEGRADED
            else ReconstructionSceneAvailability.READY
        )
        self._set_availability(availability)
        self.persistent_scene_changed.emit(scene, str(self._scene_manifest))
        return scene

    def _set_checkpoint(
        self,
        checkpoint: ReconstructionEngineCheckpoint,
        manifest: Path,
    ) -> ReconstructionEngineCheckpoint:
        self._checkpoint = checkpoint
        self._checkpoint_manifest = manifest.resolve()
        self._resume_compatibility = None
        self.checkpoint_changed.emit(checkpoint, str(self._checkpoint_manifest))
        return checkpoint

    def _set_availability(self, availability: ReconstructionSceneAvailability) -> None:
        if availability is self._availability:
            return
        self._availability = availability
        self.scene_availability_changed.emit(availability)

    def _persistent_manifest_from_result(self, payload: Mapping[str, object]) -> Path:
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ReconstructionValidationError(
                "Persistent offline result is missing its artifacts mapping."
            )
        for key in (
            "persistent_scene_manifest",
            "persistent_reconstruction_scene_manifest",
        ):
            value = str(artifacts.get(key, "")).strip()
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    raise ReconstructionValidationError(
                        "Persistent scene manifest emitted by an offline worker must be absolute."
                    )
                return path.resolve()
        raise ReconstructionValidationError(
            "Offline result claims persistent_scene_ready but does not expose a persistent scene manifest."
        )

    @staticmethod
    def _normalize_scene_manifest(path: str | Path) -> Path:
        target = Path(path).expanduser()
        if not target.is_absolute():
            raise ReconstructionValidationError("Persistent scene path must be absolute.")
        target = target.resolve()
        return target / "persistent_reconstruction_scene.json" if target.is_dir() else target

    @staticmethod
    def _normalize_checkpoint_manifest(path: str | Path) -> Path:
        target = Path(path).expanduser()
        if not target.is_absolute():
            raise ReconstructionValidationError("Checkpoint path must be absolute.")
        target = target.resolve()
        return target / "reconstruction_engine_checkpoint.json" if target.is_dir() else target

    @staticmethod
    def _require_uuid(value: str, *, field_name: str) -> str:
        try:
            return str(UUID(str(value).strip())).lower()
        except (ValueError, AttributeError, TypeError) as exc:
            raise ReconstructionValidationError(f"{field_name} must be a valid UUID.") from exc

    def _require_active(self) -> None:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction persistence controller is shut down."
            )


__all__ = [
    "ReconstructionPersistenceController",
    "ReconstructionResumePreparation",
    "ReconstructionSceneAvailability",
]
