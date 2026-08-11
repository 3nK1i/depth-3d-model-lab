"""Qt application coordinator for Phase 8 reconstruction.

Phase 8.8A1 established one dormant application-facing ownership boundary.
Phase 8.8A2 exposed the existing live-SLAM lifecycle. Phase 8.8A3 composes an
offline process bridge for real image-sequence/recorded-video ingestion and
isolated PyCOLMAP execution. Phase 8.8A4 exposes the existing F5 persistent
scene and G2 checkpoint/resume contracts. Phase 8.8A5 hardens resource
interlocks, result admission, cancellation/restart, and shutdown. Phase 8.8B2
adds dependency-light source/workflow selection. Phase 8.8B3 adds state-driven
runtime launch/pause/resume/stop helpers while Reconstruction navigation remains
disabled. Phase 8.8G2 exposes explicit persistent-scene load/save/reload interaction
without adding a second persistence format or checkpoint-resume behavior.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Collection, Mapping, Sequence

from PyQt6.QtCore import QObject, pyqtSignal

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveSlamCaptureController, LiveStellaLaunchSettings
from app.offline_reconstruction_controller import (
    OFFLINE_RECONSTRUCTION_WORKER,
    OfflineReconstructionConfiguration,
    OfflineReconstructionController,
    OfflineReconstructionRunState,
    OfflineReconstructionSourceKind,
    build_offline_reconstruction_command,
)
from app.reconstruction_session_model import (
    ReconstructionSessionModel,
    build_reconstruction_session_model,
)
from app.reconstruction_persistence_controller import (
    ReconstructionPersistenceController,
    ReconstructionResumePreparation,
    ReconstructionSceneAvailability,
)
from core.capture.base_source import SourceStatus
from core.capture.camera_source import CameraSource
from core.capture.image_source import ImageSource
from core.capture.rtsp_source import RtspSource
from core.capture.video_source import VideoSource
from core.contracts import ImageSize
from core.reconstruction import (
    CameraIntrinsicsPolicy,
    LiveCaptureRuntimeState,
    LiveCaptureRuntimeStats,
    PersistentReconstructionScene,
    ReconstructionCamera,
    ReconstructionEngineCheckpoint,
    ReconstructionPipelinePlan,
    ReconstructionPipelineRunResult,
    ReconstructionProviderDescriptor,
    ReconstructionQualityProfile,
    ReconstructionResumeCompatibilityResult,
    ReconstructionStageRunResult,
    estimate_camera_geometry_from_horizontal_fov,
    new_reconstruction_session_id,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sources import SUPPORTED_IMAGE_EXTENSIONS


ReconstructionStartGuard = Callable[[], str | None]


class ReconstructionWorkflow(str, Enum):
    """User-facing Reconstruction input workflow selected in Phase 8.8B2."""

    NONE = "none"
    LIVE_SOURCE = "live_source"
    IMAGE_SEQUENCE = "image_sequence"
    RECORDED_VIDEO = "recorded_video"


SUPPORTED_RECONSTRUCTION_VIDEO_EXTENSIONS = (
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
)

DEFAULT_RECONSTRUCTION_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "eia_depth_reconstruction"
DEFAULT_PYCOLMAP_PYTHON = Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python"
DEFAULT_STELLA_INSTALL_PREFIX = Path.home() / "Desktop/eia-stella-vslam-probe"
DEFAULT_STELLA_WORKER = (
    Path.home() / "Desktop/eia-stella-live-worker/bin/eia_stella_live_worker"
)
DEFAULT_STELLA_VOCABULARY = (
    Path.home() / "Desktop/phase8_4_stella_probe_data/orb_vocab.fbow"
)
DEFAULT_TEST_HORIZONTAL_FOV_DEGREES = 70.0
DEFAULT_TEST_FOCAL_LENGTH_RELATIVE_UNCERTAINTY = 0.15


def _runtime_path_from_environment(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else fallback.expanduser().resolve()


def default_pycolmap_python_path() -> Path:
    """Return the isolated PyCOLMAP interpreter used by B3 offline starts."""

    path = _runtime_path_from_environment(
        "EIA_RECONSTRUCTION_PYCOLMAP_PYTHON",
        DEFAULT_PYCOLMAP_PYTHON,
    )
    if not path.is_file():
        raise ReconstructionValidationError(
            "Offline reconstruction requires the isolated PyCOLMAP interpreter. "
            f"Expected: {path}. Set EIA_RECONSTRUCTION_PYCOLMAP_PYTHON to override it."
        )
    return path


def default_reconstruction_output_root() -> Path:
    """Return the transient B3 runtime-output root without creating it."""

    return _runtime_path_from_environment(
        "EIA_RECONSTRUCTION_OUTPUT_ROOT",
        DEFAULT_RECONSTRUCTION_OUTPUT_ROOT,
    )


def default_live_stella_launch_settings() -> LiveStellaLaunchSettings:
    """Resolve and validate the already-qualified external Stella runtime paths."""

    install_prefix = _runtime_path_from_environment(
        "EIA_RECONSTRUCTION_STELLA_PREFIX",
        DEFAULT_STELLA_INSTALL_PREFIX,
    )
    worker = _runtime_path_from_environment(
        "EIA_RECONSTRUCTION_STELLA_WORKER",
        DEFAULT_STELLA_WORKER,
    )
    vocabulary = _runtime_path_from_environment(
        "EIA_RECONSTRUCTION_STELLA_VOCAB",
        DEFAULT_STELLA_VOCABULARY,
    )
    if not install_prefix.is_dir():
        raise ReconstructionValidationError(
            f"Qualified Stella install prefix is missing: {install_prefix}"
        )
    if not worker.is_file():
        raise ReconstructionValidationError(
            f"Qualified Stella live worker is missing: {worker}"
        )
    if not vocabulary.is_file():
        raise ReconstructionValidationError(
            f"Stella ORB vocabulary is missing: {vocabulary}"
        )
    return LiveStellaLaunchSettings(
        install_prefix=install_prefix,
        worker_executable=worker,
        vocabulary_path=vocabulary,
    )


@dataclass(frozen=True, slots=True)
class ReconstructionSourceSelection:
    """Validated, dependency-light source selection prepared for B3 runtime controls."""

    workflow: ReconstructionWorkflow
    ready: bool
    display_name: str
    detail: str
    path: Path | None = None
    item_count: int | None = None
    live_source_kind: str | None = None

    @classmethod
    def none(cls) -> "ReconstructionSourceSelection":
        return cls(
            workflow=ReconstructionWorkflow.NONE,
            ready=False,
            display_name="No reconstruction source selected",
            detail="Choose Live source, Image sequence, or Recorded video.",
        )


class ReconstructionController(QObject):
    """Application-level reconstruction coordinator."""

    live_state_changed = pyqtSignal(object)  # LiveCaptureRuntimeState
    live_observation_ready = pyqtSignal(object)  # LiveSlamObservation
    offline_state_changed = pyqtSignal(object)  # OfflineReconstructionRunState
    offline_event_received = pyqtSignal(object)
    offline_log_received = pyqtSignal(str)
    offline_result_ready = pyqtSignal(object, str)
    offline_run_finished = pyqtSignal(object)
    scene_availability_changed = pyqtSignal(object)
    persistent_scene_changed = pyqtSignal(object, str)
    checkpoint_changed = pyqtSignal(object, str)
    resume_compatibility_changed = pyqtSignal(object)
    source_selection_changed = pyqtSignal(object)  # ReconstructionSourceSelection
    session_model_changed = pyqtSignal(object)  # ReconstructionSessionModel
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        application_controller: ApplicationController,
        parent: QObject | None = None,
        *,
        offline_worker_script: Path = OFFLINE_RECONSTRUCTION_WORKER,
    ) -> None:
        super().__init__(parent)
        if not isinstance(application_controller, ApplicationController):
            raise TypeError("application_controller must be ApplicationController.")

        self._application_controller = application_controller
        self._shutdown_started = False
        self._live_start_guard: ReconstructionStartGuard | None = None
        self._offline_start_guard: ReconstructionStartGuard | None = None
        self._live_session_id: str | None = None
        self._source_selection = ReconstructionSourceSelection.none()
        self._live_slam_controller = LiveSlamCaptureController(
            application_controller,
            self,
        )
        self._live_slam_controller.state_changed.connect(self.live_state_changed.emit)
        self._live_slam_controller.observation_ready.connect(self.live_observation_ready.emit)
        self._live_slam_controller.error_occurred.connect(self.error_occurred.emit)
        # G1 keeps the last controller-level live state separately from the runtime
        # object. Older UI/backward tests intentionally use compact runtime doubles
        # that expose telemetry but no ``state`` attribute; session composition must
        # remain driven by the public live-state signal rather than require additional
        # private runtime structure.
        self._session_live_state = self._live_slam_controller.state
        application_controller.source_type_changed.connect(self._on_application_source_changed)
        application_controller.source_status_changed.connect(self._on_application_source_changed)

        self._persistence_controller = ReconstructionPersistenceController(self)
        self._persistence_controller.scene_availability_changed.connect(
            self.scene_availability_changed.emit
        )
        self._persistence_controller.persistent_scene_changed.connect(
            self.persistent_scene_changed.emit
        )
        self._persistence_controller.checkpoint_changed.connect(self.checkpoint_changed.emit)
        self._persistence_controller.resume_compatibility_changed.connect(
            self.resume_compatibility_changed.emit
        )
        self._persistence_controller.error_occurred.connect(self.error_occurred.emit)

        self._offline_controller = OfflineReconstructionController(
            self,
            worker_script=offline_worker_script,
        )
        self._offline_controller.state_changed.connect(self.offline_state_changed.emit)
        self._offline_controller.event_received.connect(self.offline_event_received.emit)
        self._offline_controller.log_received.connect(self.offline_log_received.emit)
        self._offline_controller.result_ready.connect(self._on_offline_result_ready)
        self._offline_controller.run_finished.connect(self.offline_run_finished.emit)
        self._offline_controller.error_occurred.connect(self.error_occurred.emit)

        # G1 exposes one computed, application-level session model without taking
        # ownership from the existing live/offline/session/persistence contracts.
        for signal in (
            self.live_state_changed,
            self.offline_state_changed,
            self.source_selection_changed,
            self.scene_availability_changed,
            self.persistent_scene_changed,
            self.checkpoint_changed,
        ):
            signal.connect(self._emit_session_model_changed)

    @property
    def live_slam_controller(self) -> LiveSlamCaptureController:
        return self._live_slam_controller

    @property
    def persistence_controller(self) -> ReconstructionPersistenceController:
        return self._persistence_controller

    @property
    def offline_controller(self) -> OfflineReconstructionController:
        return self._offline_controller

    @property
    def live_state(self) -> LiveCaptureRuntimeState:
        return self._live_slam_controller.state

    @property
    def live_stats(self) -> LiveCaptureRuntimeStats | None:
        runtime = self._live_slam_controller.runtime
        return runtime.stats if runtime is not None else None

    @property
    def live_session_id(self) -> str | None:
        return self._live_session_id

    @property
    def offline_state(self) -> OfflineReconstructionRunState:
        return self._offline_controller.state

    @property
    def offline_session_id(self) -> str | None:
        return self._offline_controller.session_id

    @property
    def offline_output_dir(self) -> Path | None:
        return self._offline_controller.output_dir

    @property
    def offline_is_running(self) -> bool:
        return self._offline_controller.is_running

    @property
    def source_selection(self) -> ReconstructionSourceSelection:
        return self._source_selection

    @property
    def scene_availability(self) -> ReconstructionSceneAvailability:
        return self._persistence_controller.scene_availability

    @property
    def persistent_scene(self) -> PersistentReconstructionScene | None:
        return self._persistence_controller.scene

    @property
    def persistent_scene_manifest(self) -> Path | None:
        return self._persistence_controller.scene_manifest

    @property
    def engine_checkpoint(self) -> ReconstructionEngineCheckpoint | None:
        return self._persistence_controller.checkpoint

    @property
    def engine_checkpoint_manifest(self) -> Path | None:
        return self._persistence_controller.checkpoint_manifest

    @property
    def resume_compatibility(self) -> ReconstructionResumeCompatibilityResult | None:
        return self._persistence_controller.resume_compatibility

    @property
    def session_model(self) -> ReconstructionSessionModel:
        """Return the current G1 composition without creating new persistence state."""

        return build_reconstruction_session_model(
            source_selection=self._source_selection,
            live_session_id=self._live_session_id,
            offline_session_id=self._offline_controller.session_id,
            live_state=self._session_live_state,
            offline_state=self._offline_controller.state,
            scene_availability=self._persistence_controller.scene_availability,
            scene=self._persistence_controller.scene,
            scene_manifest=self._persistence_controller.scene_manifest,
            checkpoint=self._persistence_controller.checkpoint,
            checkpoint_manifest=self._persistence_controller.checkpoint_manifest,
        )

    @property
    def is_active(self) -> bool:
        return self._live_slam_controller.is_active or self._offline_controller.is_active

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_started

    def select_live_source(self) -> ReconstructionSourceSelection:
        """Select the existing Live 3D capture source without taking capture ownership."""

        self._require_idle_for_source_selection()
        selection = self._snapshot_live_source_selection()
        self._set_source_selection(selection)
        return selection

    def select_image_sequence(self, directory: str | Path) -> ReconstructionSourceSelection:
        """Validate and select one top-level overlapping image directory."""

        self._require_idle_for_source_selection()
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            raise ReconstructionValidationError(
                f"Image-sequence directory does not exist: {path}"
            )
        images = tuple(
            item
            for item in sorted(path.iterdir())
            if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if len(images) < 3:
            raise ReconstructionValidationError(
                "Image-sequence reconstruction requires at least 3 supported images "
                "in the selected directory."
            )
        selection = ReconstructionSourceSelection(
            workflow=ReconstructionWorkflow.IMAGE_SEQUENCE,
            ready=True,
            display_name=path.name or str(path),
            detail=f"{len(images)} supported images • top-level directory",
            path=path,
            item_count=len(images),
        )
        self._set_source_selection(selection)
        return selection

    def select_recorded_video(self, video_path: str | Path) -> ReconstructionSourceSelection:
        """Validate and select one recorded video without opening a reconstruction backend."""

        self._require_idle_for_source_selection()
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise ReconstructionValidationError(f"Recorded video does not exist: {path}")
        if path.suffix.casefold() not in SUPPORTED_RECONSTRUCTION_VIDEO_EXTENSIONS:
            supported = ", ".join(SUPPORTED_RECONSTRUCTION_VIDEO_EXTENSIONS)
            raise ReconstructionValidationError(
                f"Unsupported recorded-video extension '{path.suffix or '<none>'}'. "
                f"Supported: {supported}."
            )
        selection = ReconstructionSourceSelection(
            workflow=ReconstructionWorkflow.RECORDED_VIDEO,
            ready=True,
            display_name=path.name,
            detail="Recorded video • validated file selection",
            path=path,
            item_count=1,
        )
        self._set_source_selection(selection)
        return selection

    def clear_source_selection(self) -> None:
        self._require_idle_for_source_selection()
        self._set_source_selection(ReconstructionSourceSelection.none())

    def _require_idle_for_source_selection(self) -> None:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot change source selection."
            )
        if self.is_active:
            raise ReconstructionValidationError(
                "Reconstruction source cannot change while a reconstruction job is active."
            )

    def _set_source_selection(self, selection: ReconstructionSourceSelection) -> None:
        if not isinstance(selection, ReconstructionSourceSelection):
            raise TypeError("selection must be ReconstructionSourceSelection.")
        if selection == self._source_selection:
            return
        self._source_selection = selection
        self.source_selection_changed.emit(selection)

    def _snapshot_live_source_selection(self) -> ReconstructionSourceSelection:
        source = self._application_controller.active_source
        if source is None:
            return ReconstructionSourceSelection(
                workflow=ReconstructionWorkflow.LIVE_SOURCE,
                ready=False,
                display_name="No compatible Live 3D source",
                detail="Connect a Camera, RTSP stream, or Video in Live 3D first.",
                live_source_kind="none",
            )

        status = source.status
        ready = status is SourceStatus.CONNECTED
        status_text = status.value.replace("_", " ").title()
        if isinstance(source, CameraSource):
            return ReconstructionSourceSelection(
                workflow=ReconstructionWorkflow.LIVE_SOURCE,
                ready=ready,
                display_name=f"Camera {source.device_index}",
                detail=f"USB camera • {status_text}",
                live_source_kind="camera",
            )
        if isinstance(source, RtspSource):
            return ReconstructionSourceSelection(
                workflow=ReconstructionWorkflow.LIVE_SOURCE,
                ready=ready,
                display_name=source.sanitized_url,
                detail=f"RTSP stream • {status_text}",
                live_source_kind="rtsp",
            )
        if isinstance(source, VideoSource):
            path = Path(source.path).expanduser().resolve()
            return ReconstructionSourceSelection(
                workflow=ReconstructionWorkflow.LIVE_SOURCE,
                ready=ready,
                display_name=path.name,
                detail=f"Recorded video as simulated live • {status_text}",
                path=path,
                item_count=1,
                live_source_kind="recorded_video",
            )
        if isinstance(source, ImageSource):
            return ReconstructionSourceSelection(
                workflow=ReconstructionWorkflow.LIVE_SOURCE,
                ready=False,
                display_name="Static image is not a live source",
                detail="Choose Image sequence for offline reconstruction, or connect Camera/RTSP/Video.",
                live_source_kind="unsupported_image",
            )
        return ReconstructionSourceSelection(
            workflow=ReconstructionWorkflow.LIVE_SOURCE,
            ready=False,
            display_name=type(source).__name__,
            detail="The active Live 3D source is not supported by live reconstruction.",
            live_source_kind="unsupported",
        )

    def _on_application_source_changed(self, _value: object) -> None:
        if self._shutdown_started:
            return
        if self._source_selection.workflow is not ReconstructionWorkflow.LIVE_SOURCE:
            return
        # Source lifecycle changes are observational; they do not take or replace
        # capture ownership and may legitimately happen during a live SLAM session.
        self._set_source_selection(self._snapshot_live_source_selection())

    def configure_start_guards(
        self,
        *,
        live: ReconstructionStartGuard | None = None,
        offline: ReconstructionStartGuard | None = None,
    ) -> None:
        """Install application-level resource guards before Reconstruction UI is enabled.

        The controller remains usable in focused tests without guards. ``MainWindow``
        supplies guards that reject starts while another long-running application mode
        owns resources and releases an idle resident depth model before offline SfM.
        """

        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot be reconfigured."
            )
        if self.is_active:
            raise ReconstructionValidationError(
                "Reconstruction start guards cannot be changed while reconstruction is active."
            )
        if live is not None and not callable(live):
            raise TypeError("live reconstruction start guard must be callable or None.")
        if offline is not None and not callable(offline):
            raise TypeError("offline reconstruction start guard must be callable or None.")
        self._live_start_guard = live
        self._offline_start_guard = offline

    def start_selected_live(
        self,
        *,
        camera: ReconstructionCamera,
        settings: LiveStellaLaunchSettings | None = None,
        profile: ReconstructionQualityProfile = ReconstructionQualityProfile.EFFICIENT,
        session_id: str | None = None,
        camera_intrinsics_policy: CameraIntrinsicsPolicy = CameraIntrinsicsPolicy.CALIBRATED_ONLY,
    ) -> str:
        """Start the currently selected live workflow after B2 readiness validation."""

        selection = self._source_selection
        if selection.workflow is not ReconstructionWorkflow.LIVE_SOURCE:
            raise ReconstructionValidationError(
                "Select the Live source workflow before starting live reconstruction."
            )
        if not selection.ready:
            raise ReconstructionValidationError(
                selection.detail or "The selected live source is not ready."
            )
        if not isinstance(camera_intrinsics_policy, CameraIntrinsicsPolicy):
            raise TypeError("camera_intrinsics_policy must be CameraIntrinsicsPolicy.")
        resolved_settings = settings or default_live_stella_launch_settings()
        if camera_intrinsics_policy is CameraIntrinsicsPolicy.CALIBRATED_ONLY:
            return self.start_live(
                camera=camera,
                settings=resolved_settings,
                profile=profile,
                session_id=session_id,
            )
        return self.start_live(
            camera=camera,
            settings=resolved_settings,
            profile=profile,
            session_id=session_id,
            camera_intrinsics_policy=camera_intrinsics_policy,
        )

    def build_estimated_live_camera(
        self,
        *,
        horizontal_fov_degrees: float = DEFAULT_TEST_HORIZONTAL_FOV_DEGREES,
        focal_length_relative_uncertainty: float = (
            DEFAULT_TEST_FOCAL_LENGTH_RELATIVE_UNCERTAINTY
        ),
    ) -> ReconstructionCamera:
        """Build explicit provisional intrinsics for live test/reconstruction mode.

        This is intentionally application-side glue over the Phase 8.6 camera-geometry
        contract. It never labels the estimate as calibrated and never establishes metric
        scale. The current capture resolution comes from the latest frame when available;
        a configured USB-camera resolution is accepted as a pre-frame fallback.
        """

        selection = self._source_selection
        if selection.workflow is not ReconstructionWorkflow.LIVE_SOURCE:
            raise ReconstructionValidationError(
                "Select the Live source workflow before estimating test intrinsics."
            )
        if not selection.ready:
            raise ReconstructionValidationError(
                selection.detail or "The selected live source is not ready."
            )

        latest = self._application_controller.latest_source_frame
        if latest is not None:
            image_size = ImageSize(latest.width, latest.height)
        else:
            source = self._application_controller.active_source
            if (
                isinstance(source, CameraSource)
                and source.requested_width is not None
                and source.requested_height is not None
            ):
                image_size = ImageSize(source.requested_width, source.requested_height)
            else:
                raise ReconstructionValidationError(
                    "Estimated live intrinsics need the current capture resolution. "
                    "Wait for the first Camera/RTSP/Video frame, then try Start again."
                )

        kind = selection.live_source_kind or "source"
        profile = estimate_camera_geometry_from_horizontal_fov(
            profile_id=f"phase8-8b3-test-{kind}-{image_size.width}x{image_size.height}",
            camera_id=f"phase8-8b3-{kind}",
            image_size=image_size,
            horizontal_fov_degrees=horizontal_fov_degrees,
            focal_length_relative_uncertainty=focal_length_relative_uncertainty,
            notes=(
                "Phase 8.8B3 explicit Test Mode: assumed horizontal FOV, centered principal "
                "point, square pixels, zero distortion. Provisional relative reconstruction "
                "only; metric/metrology authority is not established."
            ),
        )
        return profile.camera

    def start_selected_offline(
        self,
        *,
        sfm_python: str | Path | None = None,
        output_root: str | Path | None = None,
        profile: ReconstructionQualityProfile = ReconstructionQualityProfile.EFFICIENT,
        max_keyframes: int = 80,
        session_id: str | None = None,
    ) -> str:
        """Start the selected image-sequence/video workflow in an isolated worker."""

        selection = self._source_selection
        if selection.workflow not in {
            ReconstructionWorkflow.IMAGE_SEQUENCE,
            ReconstructionWorkflow.RECORDED_VIDEO,
        }:
            raise ReconstructionValidationError(
                "Select Image sequence or Recorded video before starting offline reconstruction."
            )
        if not selection.ready or selection.path is None:
            raise ReconstructionValidationError(
                selection.detail or "The selected offline source is not ready."
            )
        if not isinstance(profile, ReconstructionQualityProfile):
            raise TypeError("profile must be ReconstructionQualityProfile.")

        sid = session_id or new_reconstruction_session_id()
        python_path = (
            Path(sfm_python).expanduser().resolve()
            if sfm_python is not None
            else default_pycolmap_python_path()
        )
        if not python_path.is_file():
            raise ReconstructionValidationError(
                f"External PyCOLMAP Python executable does not exist: {python_path}"
            )
        root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else default_reconstruction_output_root()
        )
        output_dir = root / sid
        source_kind = (
            OfflineReconstructionSourceKind.IMAGE_DIRECTORY
            if selection.workflow is ReconstructionWorkflow.IMAGE_SEQUENCE
            else OfflineReconstructionSourceKind.RECORDED_VIDEO
        )
        configuration = OfflineReconstructionConfiguration(
            source_kind=source_kind,
            inputs=(selection.path,),
            output_dir=output_dir,
            sfm_python=python_path,
            profile=profile,
            max_keyframes=max_keyframes,
        )
        return self.start_offline(configuration, session_id=sid)

    def start_live(
        self,
        *,
        camera: ReconstructionCamera,
        settings: LiveStellaLaunchSettings,
        profile: ReconstructionQualityProfile = ReconstructionQualityProfile.EFFICIENT,
        session_id: str | None = None,
        camera_intrinsics_policy: CameraIntrinsicsPolicy = CameraIntrinsicsPolicy.CALIBRATED_ONLY,
    ) -> str:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot start a live session."
            )
        if self._offline_controller.is_active:
            raise ReconstructionValidationError(
                "Offline reconstruction is active or finalizing; wait for it before starting live reconstruction."
            )
        self._run_start_guard(self._live_start_guard, operation="live reconstruction")
        if not isinstance(camera_intrinsics_policy, CameraIntrinsicsPolicy):
            raise TypeError("camera_intrinsics_policy must be CameraIntrinsicsPolicy.")
        if camera_intrinsics_policy is CameraIntrinsicsPolicy.CALIBRATED_ONLY:
            sid = self._live_slam_controller.start_stella_for_active_source(
                camera=camera,
                settings=settings,
                profile=profile,
                session_id=session_id,
            )
        else:
            sid = self._live_slam_controller.start_stella_for_active_source(
                camera=camera,
                settings=settings,
                profile=profile,
                session_id=session_id,
                camera_intrinsics_policy=camera_intrinsics_policy,
            )
        self._live_session_id = sid
        self._emit_session_model_changed()
        return sid

    def pause_live(self) -> bool:
        if self._shutdown_started:
            return False
        return self._live_slam_controller.pause()

    def resume_live(self) -> bool:
        if self._shutdown_started:
            return False
        return self._live_slam_controller.resume()

    def request_stop_live(self) -> bool:
        if self._shutdown_started:
            return False
        return self._live_slam_controller.request_stop()

    def stop_live(self, *, timeout_s: float = 7.0) -> bool:
        return self._live_slam_controller.stop(timeout_s=timeout_s)

    def start_offline(
        self,
        configuration: OfflineReconstructionConfiguration,
        *,
        session_id: str | None = None,
    ) -> str:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot start an offline session."
            )
        if self._live_slam_controller.is_active:
            raise ReconstructionValidationError(
                "Live reconstruction is active; stop it before starting offline reconstruction."
            )
        self._run_start_guard(self._offline_start_guard, operation="offline reconstruction")
        sid = self._offline_controller.start(configuration, session_id=session_id)
        self._emit_session_model_changed()
        return sid

    def cancel_offline(self) -> bool:
        if self._shutdown_started:
            return False
        return self._offline_controller.cancel()

    def adopt_pipeline_result(
        self,
        result: ReconstructionPipelineRunResult,
        output_dir: str | Path,
        *,
        payload_snapshots_by_kind: Mapping[str, Mapping[str, object]] | None = None,
    ) -> PersistentReconstructionScene:
        self._require_idle_for_persistence("adopt a persistent reconstruction scene")
        return self._persistence_controller.adopt_pipeline_result(
            result,
            output_dir,
            payload_snapshots_by_kind=payload_snapshots_by_kind,
        )

    def load_persistent_scene(
        self,
        path: str | Path,
        *,
        expected_session_id: str | None = None,
        verify_file_references: bool = True,
    ) -> PersistentReconstructionScene:
        self._require_idle_for_persistence("load a persistent reconstruction scene")
        return self._persistence_controller.load_scene(
            path,
            expected_session_id=expected_session_id,
            verify_file_references=verify_file_references,
        )

    def save_persistent_scene(self, output_dir: str | Path) -> Path:
        self._require_idle_for_persistence("save a persistent reconstruction scene")
        return self._persistence_controller.save_scene(output_dir)

    def reload_persistent_scene(self) -> PersistentReconstructionScene:
        """Reload the active manifest with integrity and same-session checks."""

        self._require_idle_for_persistence("reload the persistent reconstruction scene")
        scene = self._persistence_controller.scene
        manifest = self._persistence_controller.scene_manifest
        if scene is None or manifest is None:
            raise ReconstructionValidationError(
                "No persistent reconstruction scene manifest is available to reload."
            )
        return self._persistence_controller.load_scene(
            manifest,
            expected_session_id=scene.session_id,
            verify_file_references=True,
        )

    def clear_persistent_scene(self) -> None:
        self._require_idle_for_persistence("clear the persistent reconstruction scene")
        self._persistence_controller.clear_scene()

    def create_engine_checkpoint(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        stage_results: Sequence[ReconstructionStageRunResult],
        output_dir: str | Path,
    ) -> ReconstructionEngineCheckpoint:
        self._require_idle_for_persistence("create a reconstruction checkpoint")
        return self._persistence_controller.create_checkpoint(
            plan=plan,
            providers=providers,
            stage_results=stage_results,
            output_dir=output_dir,
        )

    def load_engine_checkpoint(self, path: str | Path) -> ReconstructionEngineCheckpoint:
        self._require_idle_for_persistence("load a reconstruction checkpoint")
        return self._persistence_controller.load_checkpoint(path)

    def check_engine_resume_compatibility(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        available_payload_refs: Collection[str],
    ) -> ReconstructionResumeCompatibilityResult:
        self._require_idle_for_persistence("check reconstruction resume compatibility")
        return self._persistence_controller.check_resume_compatibility(
            plan=plan,
            providers=providers,
            available_payload_refs=available_payload_refs,
        )

    def prepare_engine_resume(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        available_payload_refs: Collection[str],
    ) -> ReconstructionResumePreparation:
        self._require_idle_for_persistence("prepare reconstruction resume")
        return self._persistence_controller.prepare_resume(
            plan=plan,
            providers=providers,
            available_payload_refs=available_payload_refs,
        )

    def clear_engine_checkpoint(self) -> None:
        self._require_idle_for_persistence("clear the reconstruction checkpoint")
        self._persistence_controller.clear_checkpoint()

    def _emit_session_model_changed(self, *_args: object) -> None:
        # Cache the public state carried by ``live_state_changed`` before composing
        # the G1 model. This preserves backward compatibility with compact runtime
        # doubles that do not implement the full LiveCaptureSlamRuntime surface.
        if _args and isinstance(_args[0], LiveCaptureRuntimeState):
            self._session_live_state = _args[0]
        self.session_model_changed.emit(self.session_model)

    def _run_start_guard(
        self,
        guard: ReconstructionStartGuard | None,
        *,
        operation: str,
    ) -> None:
        if guard is None:
            return
        try:
            reason = guard()
        except Exception as exc:
            raise ReconstructionValidationError(
                f"{operation.capitalize()} preflight failed: {exc}"
            ) from exc
        if reason is None:
            return
        if not isinstance(reason, str):
            raise ReconstructionValidationError(
                f"{operation.capitalize()} preflight returned an invalid result."
            )
        reason = reason.strip()
        if reason:
            raise ReconstructionValidationError(reason)

    def _require_idle_for_persistence(self, operation: str) -> None:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                f"Cannot {operation}: reconstruction controller is shut down."
            )
        if self.is_active:
            raise ReconstructionValidationError(
                f"Cannot {operation} while a reconstruction session is active."
            )

    def _on_offline_result_ready(self, payload: object, output_dir: str) -> None:
        if isinstance(payload, dict):
            try:
                self._persistence_controller.accept_offline_result(payload, output_dir)
            except (OSError, TypeError, ValueError, ReconstructionValidationError) as exc:
                self.error_occurred.emit(f"Offline reconstruction persistence bridge rejected result: {exc}")
        self.offline_result_ready.emit(payload, output_dir)

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._offline_controller.shutdown()
        self._live_slam_controller.shutdown()
        self._persistence_controller.shutdown()


__all__ = [
    "ReconstructionController",
    "ReconstructionStartGuard",
    "OfflineReconstructionConfiguration",
    "OfflineReconstructionRunState",
    "OfflineReconstructionSourceKind",
    "ReconstructionResumePreparation",
    "ReconstructionSceneAvailability",
    "build_offline_reconstruction_command",
]
