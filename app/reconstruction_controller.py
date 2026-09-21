"""Qt application coordinator for Phase 8 reconstruction.

Phase 8.8A1 established one dormant application-facing ownership boundary.
Phase 8.8A2 exposed the existing live-SLAM lifecycle. Phase 8.8A3 composes an
offline process bridge for real image-sequence/recorded-video ingestion and
isolated PyCOLMAP execution. Phase 8.8A4 exposes the existing F5 persistent
scene and G2 checkpoint/resume contracts. Phase 8.8A5 hardens resource
interlocks, result admission, cancellation/restart, and shutdown. Phase 8.8B2
adds dependency-light source/workflow selection. Phase 8.8B3 adds state-driven
runtime launch/pause/resume/stop helpers while Reconstruction navigation remains
disabled. Phase 8.8G2 exposes explicit persistent-scene load/save/reload interaction.
Phase 8.8G3 adds explicit checkpoint loading and validated resume preparation without
launching resumed execution or changing the existing whole-engine checkpoint format.
I6.1B composes the dependency-light offline workspace snapshot from these existing
authoritative systems without taking ownership from them or activating production Auto.
I6.2B wires the I6.2A One-Start preflight into the selected offline launch path while
preserving the existing worker, session identity, resource guard, and sparse-SfM scope.
I6.2C adds a fail-closed terminal-result handoff between the worker, persistence, and UI
without promoting sparse SfM to canonical dense geometry or persistent-scene authority.
I6.3C composes the execution-free Reconstruction Advisor into one passive, deduplicated
controller state while keeping evidence-context binding explicit and side-effect-free.
I6.4B composes explicit manual dense-provider intent from the frozen I6.4A contract
without activating providers, changing execution settings, or enabling production Auto.
I6.4C exposes the frozen I5 production-Auto qualification boundary as a read-only
controller property; Auto remains deferred, unauthorized, non-selectable, and unwired.
I6.5C composes explicit frame/view planning intent and deterministic resource-aware
plan state without changing worker sampling, SfM execution, or PyCOLMAP authority.
I6.6C transports validated provisional sparse-SfM snapshots into controller-owned
passive state with session isolation and latest-only backpressure semantics.
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
from app.da3_canonical_integration_controller import (
    DA3_CANONICAL_WORKER,
    DA3CanonicalIntegrationConfiguration,
    DA3CanonicalIntegrationController,
    DA3CanonicalRunState,
    DA3CanonicalSourceKind,
    default_da3_python_path,
    default_da3_repository_path,
)
from app.offline_reconstruction_controller import (
    OFFLINE_RECONSTRUCTION_WORKER,
    OfflineReconstructionConfiguration,
    OfflineReconstructionController,
    OfflineReconstructionRunState,
    OfflineReconstructionSourceKind,
    build_offline_reconstruction_command,
)
from app.reconstruction_dense_job_controller import DenseJobState, ExplicitDenseJobController
from app.reconstruction_job_telemetry_relay import JobTelemetryQtRelay
from core.reconstruction.mvs_provider import I43B_PROVIDER_ID
from app.reconstruction_session_model import (
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionModel,
    build_reconstruction_session_model,
)
from app.reconstruction_offline_workspace import (
    OfflineReconstructionWorkspaceSnapshot,
    build_offline_reconstruction_workspace_snapshot,
)
from app.reconstruction_one_start_preflight import (
    OfflineOneStartPreflight,
    OneStartBlockerCode,
    build_offline_one_start_preflight,
)
from app.reconstruction_offline_result_handoff import (
    OfflineReconstructionResultHandoff,
    build_offline_result_handoff,
)
from app.reconstruction_advisor_state import (
    ReconstructionAdvisorEvidenceBinding,
    ReconstructionAdvisorLiveState,
    build_reconstruction_advisor_live_state,
)
from app.reconstruction_provider_selection_state import (
    ManualProviderSelectionStatus,
    ReconstructionProviderSelectionState,
    build_reconstruction_provider_selection_state,
)
from app.reconstruction_frame_view_plan_state import (
    FrameViewPlanIntentOrigin,
    ReconstructionFrameViewPlanState,
    build_reconstruction_frame_view_plan_state,
)
from app.reconstruction_checkpoint_resume import ReconstructionResumeContext
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
    CameraCalibrationProfile,
    MetricQualificationEvidence,
    PersistentReconstructionScene,
    PersistentSceneAuthoritySnapshot,
    ReconstructionCamera,
    ReconstructionEngineCheckpoint,
    ReconstructionPipelinePlan,
    ReconstructionPipelineRunResult,
    ReconstructionProviderDescriptor,
    ReconstructionQualityProfile,
    ReconstructionResumeCompatibilityResult,
    ReconstructionStageRunResult,
    build_persistent_scene_authority_snapshot,
    estimate_camera_geometry_from_horizontal_fov,
    new_reconstruction_session_id,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction import reconstruction_progressive_sfm_contract as progressive_sparse_contract
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
)
from core.reconstruction.reconstruction_advisor_contract import ReconstructionAdvisorDecision
from core.reconstruction.reconstruction_production_auto_gate import (
    ProductionAutoQualificationBoundary,
    build_production_auto_qualification_boundary,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderSelectionSnapshot,
)
from core.reconstruction.sources import SUPPORTED_IMAGE_EXTENSIONS


_validate_progressive_sparse_transition = getattr(
    progressive_sparse_contract,
    "validate_progressive_" "sfm_snapshot_transition",
)

ReconstructionStartGuard = Callable[[], str | None]
_AUTHORITY_UNSET = object()


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


def _configured_pycolmap_python_path() -> Path:
    """Resolve the configured isolated PyCOLMAP interpreter without validating it."""

    return _runtime_path_from_environment(
        "EIA_RECONSTRUCTION_PYCOLMAP_PYTHON",
        DEFAULT_PYCOLMAP_PYTHON,
    )


def default_pycolmap_python_path() -> Path:
    """Return the isolated PyCOLMAP interpreter used by B3 offline starts."""

    path = _configured_pycolmap_python_path()
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
    offline_result_handoff_ready = pyqtSignal(object)  # OfflineReconstructionResultHandoff
    offline_run_finished = pyqtSignal(object)
    dense_state_changed = pyqtSignal(object)
    dense_preview_snapshot_received = pyqtSignal(object)
    dense_admission_ready = pyqtSignal(object)
    dense_run_finished = pyqtSignal(object)
    dense_log_received = pyqtSignal(str)
    job_telemetry_received = pyqtSignal(object)  # passive I6.8D; no admission authority
    da3_canonical_state_changed = pyqtSignal(object)
    da3_canonical_event_received = pyqtSignal(object)
    da3_canonical_log_received = pyqtSignal(str)
    da3_canonical_result_ready = pyqtSignal(object, str)
    da3_canonical_run_finished = pyqtSignal(object)
    scene_availability_changed = pyqtSignal(object)
    persistent_scene_changed = pyqtSignal(object, str)
    persistent_authority_changed = pyqtSignal(object, str)
    checkpoint_changed = pyqtSignal(object, str)
    resume_compatibility_changed = pyqtSignal(object)
    resume_context_changed = pyqtSignal(object)
    resume_preparation_changed = pyqtSignal(object)
    source_selection_changed = pyqtSignal(object)  # ReconstructionSourceSelection
    session_model_changed = pyqtSignal(object)  # ReconstructionSessionModel
    offline_workspace_changed = pyqtSignal(object)  # OfflineReconstructionWorkspaceSnapshot
    reconstruction_advisor_changed = pyqtSignal(object)  # ReconstructionAdvisorLiveState
    reconstruction_provider_selection_changed = pyqtSignal(object)  # ReconstructionProviderSelectionState
    reconstruction_frame_view_plan_changed = pyqtSignal(object)  # ReconstructionFrameViewPlanState
    progressive_sparse_snapshot_changed = pyqtSignal(object)  # progressive_sparse_contract.ProgressiveSfmSnapshot | None
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        application_controller: ApplicationController,
        parent: QObject | None = None,
        *,
        offline_worker_script: Path = OFFLINE_RECONSTRUCTION_WORKER,
        da3_canonical_worker_script: Path = DA3_CANONICAL_WORKER,
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
        self._engine_resume_context: ReconstructionResumeContext | None = None
        self._engine_resume_preparation: ReconstructionResumePreparation | None = None
        self._latest_offline_workspace_event: dict[str, object] | None = None
        self._last_offline_one_start_preflight: OfflineOneStartPreflight | None = None
        # I6.2C terminal-result admission cache must exist from controller
        # construction onward so rejected results remain observably "no handoff"
        # rather than exposing an uninitialized property.
        self._last_offline_result_handoff: OfflineReconstructionResultHandoff | None = None
        self._offline_one_start_guard_committed = False
        self._last_emitted_offline_workspace_snapshot: (
            OfflineReconstructionWorkspaceSnapshot | None
        ) = None
        self._reconstruction_advisor_evidence_binding = (
            ReconstructionAdvisorEvidenceBinding.unresolved()
        )
        self._last_emitted_reconstruction_advisor_state: (
            ReconstructionAdvisorLiveState | None
        ) = None
        self._manual_reconstruction_provider_id = ""
        self._provider_selection_status = ManualProviderSelectionStatus.NONE
        self._provider_selection_reason_codes: tuple[str, ...] = ()
        self._last_emitted_reconstruction_provider_selection_state: (
            ReconstructionProviderSelectionState | None
        ) = None
        self._reconstruction_frame_view_plan_request = FrameViewPlanRequest(
            mode=FrameViewPlanMode.ADAPTIVE
        )
        self._reconstruction_frame_view_plan_intent_origin = (
            FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE
        )
        self._reconstruction_frame_view_resources: AdaptiveSelectionResourceSnapshot | None = None
        self._last_emitted_reconstruction_frame_view_plan_state: (
            ReconstructionFrameViewPlanState | None
        ) = None
        self._latest_progressive_sparse_snapshot: progressive_sparse_contract.ProgressiveSfmSnapshot | None = None
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
        self._persistence_controller.persistent_authority_changed.connect(
            self.persistent_authority_changed.emit
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
        self._offline_controller.state_changed.connect(self._on_offline_state_changed)
        self._offline_controller.event_received.connect(self._on_offline_event_received)
        self._offline_controller.progressive_sparse_snapshot_received.connect(
            self._on_progressive_sparse_snapshot_received
        )
        # I6.8D observes the frozen I6.7F offline controller from outside it.
        self._offline_telemetry_relay = JobTelemetryQtRelay(self)
        self._offline_telemetry_relay.observation_received.connect(
            self.job_telemetry_received.emit)
        self._offline_controller.log_received.connect(self.offline_log_received.emit)
        self._offline_controller.result_ready.connect(self._on_offline_result_ready)
        self._offline_controller.run_finished.connect(self.offline_run_finished.emit)
        self._offline_controller.error_occurred.connect(self.error_occurred.emit)

        # Separate post-sparse lifecycle: no implicit Auto or preview admission.
        self._dense_controller = ExplicitDenseJobController(self)
        self._dense_controller.state_changed.connect(self.dense_state_changed.emit)
        self._dense_controller.provisional_snapshot_received.connect(
            self.dense_preview_snapshot_received.emit
        )
        self._dense_controller.admission_ready.connect(self.dense_admission_ready.emit)
        self._dense_controller.run_finished.connect(self.dense_run_finished.emit)
        self._dense_controller.telemetry_received.connect(self.job_telemetry_received.emit)
        self._dense_controller.log_received.connect(self.dense_log_received.emit)
        self._dense_controller.error_occurred.connect(self.error_occurred.emit)

        self._da3_canonical_controller = DA3CanonicalIntegrationController(
            self, worker_script=da3_canonical_worker_script
        )
        self._da3_canonical_controller.state_changed.connect(self.da3_canonical_state_changed.emit)
        self._da3_canonical_controller.event_received.connect(self.da3_canonical_event_received.emit)
        self._da3_canonical_controller.log_received.connect(self.da3_canonical_log_received.emit)
        self._da3_canonical_controller.result_ready.connect(self._on_da3_canonical_result_ready)
        self._da3_canonical_controller.run_finished.connect(self.da3_canonical_run_finished.emit)
        self._da3_canonical_controller.error_occurred.connect(self.error_occurred.emit)

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

        # I6.1B composes the I6.1A passive workspace contract from existing
        # authoritative controller/persistence state. The workspace remains a
        # read-only product view; it does not own runtime, persistence, provider
        # selection, Auto policy, or reconstruction authority.
        self.session_model_changed.connect(self._emit_offline_workspace_changed)
        self._persistence_controller.offline_result_changed.connect(
            self._emit_offline_workspace_changed
        )
        self.persistent_authority_changed.connect(self._emit_offline_workspace_changed)
        self._last_emitted_offline_workspace_snapshot = self.offline_workspace_snapshot

        # I6.3C is a passive composition only.  Workspace/result changes may
        # recompute guidance, but the decision cannot select/activate a provider,
        # mutate settings, start work, or qualify production Auto.  Evidence context
        # and provider availability remain explicit inputs; no live classifier is
        # introduced here.
        self.offline_workspace_changed.connect(self._emit_reconstruction_advisor_changed)
        self._persistence_controller.offline_result_changed.connect(
            self._emit_reconstruction_advisor_changed
        )
        self._last_emitted_reconstruction_advisor_state = self.reconstruction_advisor_state

        # I6.4B owns only explicit manual provider intent. Advisor/evidence changes
        # may invalidate that intent, but recommendation is never copied into
        # selection and selection is never wired into provider execution.
        self.reconstruction_advisor_changed.connect(
            self._reconcile_reconstruction_provider_selection
        )
        self._last_emitted_reconstruction_provider_selection_state = (
            self.reconstruction_provider_selection_state
        )

        # I6.5C owns planning intent only. Source changes recompute a fresh plan
        # against the same explicit request; they never retain a stale effective
        # plan and never alter worker execution. Resource input is explicit.
        self.source_selection_changed.connect(self._emit_reconstruction_frame_view_plan_changed)
        self._last_emitted_reconstruction_frame_view_plan_state = (
            self.reconstruction_frame_view_plan_state
        )

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
    def dense_controller(self) -> ExplicitDenseJobController:
        return self._dense_controller

    @property
    def dense_state(self) -> DenseJobState:
        return self._dense_controller.state

    @property
    def dense_is_active(self) -> bool:
        return self._dense_controller.is_active

    @property
    def dense_ready_for_explicit_start(self) -> bool:
        handoff = self._last_offline_result_handoff
        return bool(
            not self._shutdown_started and not self.is_active
            and self._offline_controller.state is OfflineReconstructionRunState.COMPLETED
            and handoff is not None
            and handoff.session_id == self._offline_controller.session_id
            and handoff.output_dir == str(self._offline_controller.output_dir)
            and handoff.partial_result_available
            and self._dense_controller.sparse_session_id != handoff.session_id
        )

    @property
    def da3_canonical_controller(self) -> DA3CanonicalIntegrationController:
        return self._da3_canonical_controller

    @property
    def da3_canonical_state(self) -> DA3CanonicalRunState:
        return self._da3_canonical_controller.state

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
    def progressive_sparse_snapshot(self) -> progressive_sparse_contract.ProgressiveSfmSnapshot | None:
        """Return the latest validated provisional sparse-SfM snapshot, if active."""

        return self._latest_progressive_sparse_snapshot

    @property
    def progressive_sfm_transport_stats(self) -> dict[str, int]:
        """Expose lower-level transport counters without granting snapshot authority."""

        return self._offline_controller.progressive_sfm_transport_stats

    @property
    def last_offline_one_start_preflight(self) -> OfflineOneStartPreflight | None:
        """Return the most recent I6.2 One-Start admission evidence, if any."""

        return self._last_offline_one_start_preflight

    @property
    def last_offline_result_handoff(self) -> OfflineReconstructionResultHandoff | None:
        """Return the most recent accepted I6.2C terminal-result handoff."""

        return self._last_offline_result_handoff

    @property
    def offline_is_running(self) -> bool:
        return self._offline_controller.is_running

    @property
    def source_selection(self) -> ReconstructionSourceSelection:
        return self._source_selection

    @property
    def reconstruction_advisor_evidence_binding(self) -> ReconstructionAdvisorEvidenceBinding:
        """Return the explicit frozen evidence binding used by passive I6.3 guidance."""

        return self._reconstruction_advisor_evidence_binding

    @property
    def reconstruction_advisor_state(self) -> ReconstructionAdvisorLiveState:
        """Return the current passive I6.3 advisor request+decision composition."""

        return build_reconstruction_advisor_live_state(
            source_selection=self._source_selection,
            workspace=self.offline_workspace_snapshot,
            offline_result=self._persistence_controller.offline_result,
            evidence_binding=self._reconstruction_advisor_evidence_binding,
        )

    @property
    def reconstruction_advisor_decision(self) -> ReconstructionAdvisorDecision:
        """Return the current advisory decision without granting it production effect."""

        return self.reconstruction_advisor_state.decision

    def set_reconstruction_advisor_evidence_binding(
        self, binding: ReconstructionAdvisorEvidenceBinding
    ) -> None:
        """Install explicit frozen advisory evidence and recompute passively."""

        if not isinstance(binding, ReconstructionAdvisorEvidenceBinding):
            raise TypeError("binding must be ReconstructionAdvisorEvidenceBinding.")
        if binding == self._reconstruction_advisor_evidence_binding:
            return
        self._reconstruction_advisor_evidence_binding = binding
        self._emit_reconstruction_advisor_changed()

    def clear_reconstruction_advisor_evidence_binding(self) -> None:
        """Return guidance to unresolved evidence context without changing execution."""

        self.set_reconstruction_advisor_evidence_binding(
            ReconstructionAdvisorEvidenceBinding.unresolved()
        )

    @property
    def scene_availability(self) -> ReconstructionSceneAvailability:
        return self._persistence_controller.scene_availability

    @property
    def reconstruction_provider_selection_state(self) -> ReconstructionProviderSelectionState:
        """Return passive I6.4B manual provider intent; never an execution command."""

        return build_reconstruction_provider_selection_state(
            advisor_state=self.reconstruction_advisor_state,
            selected_provider_id=self._manual_reconstruction_provider_id,
            previous_status=self._provider_selection_status,
            previous_reason_codes=self._provider_selection_reason_codes,
        )

    @property
    def reconstruction_provider_selection_snapshot(self) -> ProductionProviderSelectionSnapshot:
        """Return the frozen I6.4A product snapshot composed by I6.4B."""

        return self.reconstruction_provider_selection_state.snapshot

    def select_reconstruction_provider(self, provider_id: str) -> ReconstructionProviderSelectionState:
        """Record explicit manual provider intent without activating or executing it."""

        normalized = str(provider_id or "").strip()
        if not normalized:
            raise ReconstructionValidationError("A concrete provider ID is required.")
        try:
            candidate = build_reconstruction_provider_selection_state(
                advisor_state=self.reconstruction_advisor_state,
                selected_provider_id=normalized,
            )
        except (TypeError, ValueError) as exc:
            raise ReconstructionValidationError(str(exc)) from exc
        if candidate.status is not ManualProviderSelectionStatus.SELECTED:
            reasons = ", ".join(candidate.reason_codes) or "manual_selection_not_allowed"
            raise ReconstructionValidationError(
                f"Provider {normalized!r} cannot be selected in the current explicit context: {reasons}."
            )
        if (
            self._manual_reconstruction_provider_id == normalized
            and self._provider_selection_status is ManualProviderSelectionStatus.SELECTED
        ):
            return self.reconstruction_provider_selection_state
        self._manual_reconstruction_provider_id = normalized
        self._provider_selection_status = ManualProviderSelectionStatus.SELECTED
        self._provider_selection_reason_codes = ()
        self._emit_reconstruction_provider_selection_changed()
        return self.reconstruction_provider_selection_state

    def clear_reconstruction_provider_selection(self) -> ReconstructionProviderSelectionState:
        """Clear manual provider intent without changing any reconstruction runtime."""

        if (
            not self._manual_reconstruction_provider_id
            and self._provider_selection_status is ManualProviderSelectionStatus.NONE
            and not self._provider_selection_reason_codes
        ):
            return self.reconstruction_provider_selection_state
        self._manual_reconstruction_provider_id = ""
        self._provider_selection_status = ManualProviderSelectionStatus.NONE
        self._provider_selection_reason_codes = ()
        self._emit_reconstruction_provider_selection_changed()
        return self.reconstruction_provider_selection_state

    @property
    def reconstruction_production_auto_gate(self) -> ProductionAutoQualificationBoundary:
        """Return the fail-closed I6.4C Auto qualification boundary.

        This is presentation/state only.  It does not evaluate the I5 Auto
        engine, select/activate a provider, mutate settings, or start work.
        """

        return build_production_auto_qualification_boundary(
            self.reconstruction_provider_selection_snapshot
        )

    @property
    def reconstruction_frame_view_plan_state(self) -> ReconstructionFrameViewPlanState:
        """Return passive I6.5C planning intent and deterministic resolution state."""

        return build_reconstruction_frame_view_plan_state(
            source_selection=self._source_selection,
            request=self._reconstruction_frame_view_plan_request,
            intent_origin=self._reconstruction_frame_view_plan_intent_origin,
            resources=self._reconstruction_frame_view_resources,
        )

    @property
    def reconstruction_frame_view_plan_request(self) -> FrameViewPlanRequest:
        """Return the requested frame/view mode without implying execution."""

        return self._reconstruction_frame_view_plan_request

    @property
    def reconstruction_frame_view_resource_snapshot(
        self,
    ) -> AdaptiveSelectionResourceSnapshot | None:
        """Return explicit planning resources; this controller does not probe hardware."""

        return self._reconstruction_frame_view_resources

    def set_reconstruction_frame_view_resource_snapshot(
        self, resources: AdaptiveSelectionResourceSnapshot | None
    ) -> ReconstructionFrameViewPlanState:
        """Install explicit planning resources and recompute without changing runtime."""

        if resources is not None and not isinstance(resources, AdaptiveSelectionResourceSnapshot):
            raise TypeError("resources must be AdaptiveSelectionResourceSnapshot or None.")
        if resources == self._reconstruction_frame_view_resources:
            return self.reconstruction_frame_view_plan_state
        self._reconstruction_frame_view_resources = resources
        self._emit_reconstruction_frame_view_plan_changed()
        return self.reconstruction_frame_view_plan_state

    def set_reconstruction_frame_view_plan(
        self,
        mode: FrameViewPlanMode | str,
        *,
        requested_max_views: int | None = None,
    ) -> ReconstructionFrameViewPlanState:
        """Record explicit user planning intent without sampling or starting SfM."""

        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot change frame/view planning intent."
            )
        if self.is_active:
            raise ReconstructionValidationError(
                "Frame/view planning intent cannot change while a reconstruction job is active."
            )
        try:
            normalized_mode = mode if isinstance(mode, FrameViewPlanMode) else FrameViewPlanMode(str(mode))
            request = FrameViewPlanRequest(
                mode=normalized_mode,
                requested_max_views=requested_max_views,
            )
        except (TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(str(exc)) from exc
        if (
            request == self._reconstruction_frame_view_plan_request
            and self._reconstruction_frame_view_plan_intent_origin
            is FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE
        ):
            return self.reconstruction_frame_view_plan_state
        self._reconstruction_frame_view_plan_request = request
        self._reconstruction_frame_view_plan_intent_origin = (
            FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE
        )
        self._emit_reconstruction_frame_view_plan_changed()
        return self.reconstruction_frame_view_plan_state

    def reset_reconstruction_frame_view_plan(
        self,
    ) -> ReconstructionFrameViewPlanState:
        """Restore default Adaptive intent without changing any running worker settings."""

        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot reset frame/view planning intent."
            )
        if self.is_active:
            raise ReconstructionValidationError(
                "Frame/view planning intent cannot change while a reconstruction job is active."
            )
        request = FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE)
        if (
            request == self._reconstruction_frame_view_plan_request
            and self._reconstruction_frame_view_plan_intent_origin
            is FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE
        ):
            return self.reconstruction_frame_view_plan_state
        self._reconstruction_frame_view_plan_request = request
        self._reconstruction_frame_view_plan_intent_origin = (
            FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE
        )
        self._emit_reconstruction_frame_view_plan_changed()
        return self.reconstruction_frame_view_plan_state

    @property
    def persistent_scene(self) -> PersistentReconstructionScene | None:
        return self._persistence_controller.scene

    @property
    def persistent_scene_manifest(self) -> Path | None:
        return self._persistence_controller.scene_manifest

    @property
    def persistent_authority_snapshot(self) -> PersistentSceneAuthoritySnapshot | None:
        return self._persistence_controller.authority_snapshot

    @property
    def persistent_authority_manifest(self) -> Path | None:
        return self._persistence_controller.authority_manifest

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
    def engine_resume_context(self) -> ReconstructionResumeContext | None:
        return self._engine_resume_context

    @property
    def engine_resume_preparation(self) -> ReconstructionResumePreparation | None:
        return self._engine_resume_preparation

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
    def offline_workspace_snapshot(self) -> OfflineReconstructionWorkspaceSnapshot:
        """Return the current passive I6 offline-workspace composition."""

        return build_offline_reconstruction_workspace_snapshot(
            session_model=self.session_model,
            source_selection=self._source_selection,
            offline_state=self._offline_controller.state,
            offline_output_dir=self._offline_controller.output_dir,
            offline_result=self._persistence_controller.offline_result,
            offline_result_path=self._persistence_controller.offline_result_path,
            latest_event=self._latest_offline_workspace_event,
            scene_availability=self._persistence_controller.scene_availability,
            persistent_authority_snapshot=self._persistence_controller.authority_snapshot,
        )

    @property
    def is_active(self) -> bool:
        return (
            self._live_slam_controller.is_active
            or self._offline_controller.is_active
            or self._da3_canonical_controller.is_active
            or self._dense_controller.is_active
        )

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
        # A semantic source change must never retain provider intent from the
        # previous source, even if an external frozen evidence binding has not yet
        # been refreshed. This is a selection-state invalidation only.
        if self._manual_reconstruction_provider_id:
            self._manual_reconstruction_provider_id = ""
            self._provider_selection_status = ManualProviderSelectionStatus.CLEARED_INVALID
            self._provider_selection_reason_codes = ("source_selection_changed",)
        self._source_selection = selection
        self.source_selection_changed.emit(selection)
        self._emit_reconstruction_provider_selection_changed()

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

    def current_application_image_size(self) -> ImageSize:
        """Return current capture size independent of Reconstruction workflow."""

        latest = self._application_controller.latest_source_frame
        if latest is not None:
            return ImageSize(latest.width, latest.height)
        source = self._application_controller.active_source
        if (
            isinstance(source, CameraSource)
            and source.requested_width is not None
            and source.requested_height is not None
        ):
            return ImageSize(source.requested_width, source.requested_height)
        raise ReconstructionValidationError(
            "Current source resolution is unavailable. Wait for the first source frame, then try again."
        )

    def current_application_source_label(self) -> str:
        """Return a non-sensitive display label for the active application source."""

        source = self._application_controller.active_source
        if isinstance(source, CameraSource):
            return f"USB camera {source.device_index}"
        if isinstance(source, RtspSource):
            return "RTSP stream"
        if isinstance(source, VideoSource):
            return Path(source.path).name
        if isinstance(source, ImageSource):
            return "Static image source"
        return "No source selected"

    def current_live_image_size(self) -> ImageSize:
        """Return the current live capture size without changing source ownership.

        Phase 9.1F reuses the exact same resolution evidence as Test Mode so a
        selected calibration profile can be safely applied before the existing
        live-SLAM exact-resolution guard receives it.
        """

        selection = self._source_selection
        if selection.workflow is not ReconstructionWorkflow.LIVE_SOURCE:
            raise ReconstructionValidationError(
                "Select the Live source workflow before resolving camera resolution."
            )
        if not selection.ready:
            raise ReconstructionValidationError(
                selection.detail or "The selected live source is not ready."
            )

        try:
            return self.current_application_image_size()
        except ReconstructionValidationError as exc:
            raise ReconstructionValidationError(
                "Live camera resolution is not available yet. Wait for the first "
                "Camera/RTSP/Video frame, then try again."
            ) from exc

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
        image_size = self.current_live_image_size()
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

    def preflight_selected_offline(
        self,
        *,
        sfm_python: str | Path | None = None,
        output_root: str | Path | None = None,
        profile: ReconstructionQualityProfile = ReconstructionQualityProfile.EFFICIENT,
        session_id: str | None = None,
        disk_free_bytes: int | None = None,
        application_resource_guard_blocker: str | None = None,
    ) -> OfflineOneStartPreflight:
        """Build passive I6.2 admission evidence for the selected offline source.

        This method does not run the application resource guard and does not start a
        worker. ``start_selected_offline`` calls it twice around the single launch-
        commit guard so source/storage/output evidence is revalidated immediately
        before execution.
        """

        if not isinstance(profile, ReconstructionQualityProfile):
            raise TypeError("profile must be ReconstructionQualityProfile.")
        sid = session_id or new_reconstruction_session_id()
        python_path = (
            Path(sfm_python).expanduser().resolve()
            if sfm_python is not None
            else _configured_pycolmap_python_path()
        )
        root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else default_reconstruction_output_root()
        )
        try:
            return build_offline_one_start_preflight(
                source_selection=self._source_selection,
                controller_shutdown=self._shutdown_started,
                live_active=self._live_slam_controller.is_active,
                offline_active=self._offline_controller.is_active,
                da3_active=self._da3_canonical_controller.is_active,
                sfm_python=python_path,
                output_root=root,
                profile=profile,
                session_id=sid,
                disk_free_bytes=disk_free_bytes,
                application_resource_guard_blocker=application_resource_guard_blocker,
            )
        except ValueError as exc:
            raise ReconstructionValidationError(str(exc)) from exc

    def start_selected_offline(
        self,
        *,
        sfm_python: str | Path | None = None,
        output_root: str | Path | None = None,
        profile: ReconstructionQualityProfile = ReconstructionQualityProfile.EFFICIENT,
        max_keyframes: int | None = None,
        session_id: str | None = None,
    ) -> str:
        """Run one admitted offline Start action through the existing isolated worker."""

        if not isinstance(profile, ReconstructionQualityProfile):
            raise TypeError("profile must be ReconstructionQualityProfile.")

        sid = session_id or new_reconstruction_session_id()
        first = self.preflight_selected_offline(
            sfm_python=sfm_python,
            output_root=output_root,
            profile=profile,
            session_id=sid,
        )
        self._last_offline_one_start_preflight = first
        self._raise_if_one_start_preflight_blocked(first)

        # Commit-time application guard. MainWindow's qualified offline guard may
        # release an idle resident inference model, so it is deliberately executed
        # once and only after the passive filesystem/runtime preflight has passed.
        try:
            guard_reason = self._evaluate_start_guard(
                self._offline_start_guard,
                operation="offline reconstruction",
            )
        except ReconstructionValidationError as exc:
            blocked = self.preflight_selected_offline(
                sfm_python=first.sfm_python,
                output_root=first.output_root,
                profile=profile,
                session_id=first.session_id,
                application_resource_guard_blocker=str(exc),
            )
            self._last_offline_one_start_preflight = blocked
            raise

        commit = self.preflight_selected_offline(
            sfm_python=first.sfm_python,
            output_root=first.output_root,
            profile=profile,
            session_id=first.session_id,
            application_resource_guard_blocker=guard_reason,
        )
        self._last_offline_one_start_preflight = commit
        self._raise_if_one_start_preflight_blocked(commit)

        source_kind = (
            OfflineReconstructionSourceKind.IMAGE_DIRECTORY
            if commit.workflow == ReconstructionWorkflow.IMAGE_SEQUENCE.value
            else OfflineReconstructionSourceKind.RECORDED_VIDEO
        )
        configuration = OfflineReconstructionConfiguration(
            source_kind=source_kind,
            inputs=(Path(commit.source_path),),
            output_dir=Path(commit.proposed_output_dir),
            sfm_python=Path(commit.sfm_python),
            profile=profile,
            max_keyframes=max_keyframes,
        )
        self._offline_one_start_guard_committed = True
        try:
            # Preserve the established B3 delegation seam so focused tests and
            # downstream callers continue to observe one public offline start path.
            return self.start_offline(configuration, session_id=commit.session_id)
        finally:
            self._offline_one_start_guard_committed = False

    def start_selected_da3_canonical(
        self,
        *,
        da3_python: str | Path | None = None,
        da3_repository: str | Path | None = None,
        output_root: str | Path | None = None,
        process_res: int = 392,
        pixel_stride: int = 4,
        max_frames: int = 3,
        session_id: str | None = None,
    ) -> str:
        """Run isolated DA3 and canonicalize its supporting geometry for the selected source.

        Calibration is optional. Without separate qualification the resulting
        canonical scene remains arbitrary-scale and never certified metrology.
        """

        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot start DA3 integration."
            )
        if self._live_slam_controller.is_active or self._offline_controller.is_active:
            raise ReconstructionValidationError(
                "Stop/cancel the active reconstruction workflow before starting DA3 integration."
            )
        if self._da3_canonical_controller.is_active:
            raise ReconstructionValidationError("A DA3 canonical integration job is already active.")
        selection = self._source_selection
        if selection.workflow not in {
            ReconstructionWorkflow.IMAGE_SEQUENCE,
            ReconstructionWorkflow.RECORDED_VIDEO,
        }:
            raise ReconstructionValidationError(
                "Select Image sequence or Recorded video before integrating DA3 geometry."
            )
        if not selection.ready or selection.path is None:
            raise ReconstructionValidationError(
                selection.detail or "The selected DA3 source is not ready."
            )
        self._run_start_guard(self._offline_start_guard, operation="DA3 canonical integration")
        sid = session_id or new_reconstruction_session_id()
        root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else default_reconstruction_output_root()
        )
        configuration = DA3CanonicalIntegrationConfiguration(
            source_kind=(
                DA3CanonicalSourceKind.IMAGE_DIRECTORY
                if selection.workflow is ReconstructionWorkflow.IMAGE_SEQUENCE
                else DA3CanonicalSourceKind.RECORDED_VIDEO
            ),
            source_path=selection.path,
            output_dir=root / f"da3-canonical-{sid}",
            da3_python=(
                Path(da3_python).expanduser().resolve()
                if da3_python is not None
                else default_da3_python_path()
            ),
            da3_repository=(
                Path(da3_repository).expanduser().resolve()
                if da3_repository is not None
                else default_da3_repository_path()
            ),
            process_res=process_res,
            pixel_stride=pixel_stride,
            max_frames=max_frames,
        )
        return self._da3_canonical_controller.start(configuration, session_id=sid)

    def cancel_da3_canonical(self) -> bool:
        if self._shutdown_started:
            return False
        return self._da3_canonical_controller.cancel()

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
        if self._da3_canonical_controller.is_active:
            raise ReconstructionValidationError(
                "DA3 canonical integration is active; cancel it before starting live reconstruction."
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
        """Backward-compatible direct offline start with the established resource guard."""

        self._require_offline_start_available()
        if not self._offline_one_start_guard_committed:
            self._run_start_guard(self._offline_start_guard, operation="offline reconstruction")
        return self._commit_offline_start(configuration, session_id=session_id)

    def cancel_offline(self) -> bool:
        if self._shutdown_started:
            return False
        return self._offline_controller.cancel()

    def start_explicit_post_sparse_dense(self, *, provider_id: str = I43B_PROVIDER_ID) -> str:
        """Start the selected COLMAP-MVS run only after One Start sparse admission."""
        if provider_id != I43B_PROVIDER_ID or not self.dense_ready_for_explicit_start:
            raise ReconstructionValidationError(
                'Complete and admit One Start sparse SfM first, then explicitly choose COLMAP-MVS.'
            )
        # Never silently run Auto, reuse an earlier session, or create a second run.
        handoff = self._last_offline_result_handoff
        assert handoff is not None
        result = self._persistence_controller.offline_result
        if not isinstance(result, dict):
            raise ReconstructionValidationError('No accepted sparse result in application persistence.')
        boundary = result.get('execution_boundary')
        if not isinstance(boundary, dict) or not boundary.get('pycolmap_python'):
            raise ReconstructionValidationError('Admitted sparse result does not bind its PyCOLMAP interpreter.')
        return self._dense_controller.start(
            admitted_sparse=handoff,
            pycolmap_python=Path(boundary['pycolmap_python']),
            provider_id=provider_id,
        )

    def cancel_explicit_post_sparse_dense(self) -> bool:
        return self._dense_controller.cancel() if not self._shutdown_started else False

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

    def save_persistent_scene(
        self,
        output_dir: str | Path,
        *,
        camera_calibration_profile: CameraCalibrationProfile | None | object = _AUTHORITY_UNSET,
        metric_qualification_evidence: MetricQualificationEvidence | None | object = _AUTHORITY_UNSET,
    ) -> Path:
        self._require_idle_for_persistence("save a persistent reconstruction scene")
        if camera_calibration_profile is _AUTHORITY_UNSET and metric_qualification_evidence is _AUTHORITY_UNSET:
            return self._persistence_controller.save_scene(output_dir)
        scene = self._persistence_controller.scene
        if scene is None:
            raise ReconstructionValidationError("No persistent reconstruction scene is loaded.")
        current = self._persistence_controller.authority_snapshot
        calibration = (
            current.camera_calibration_profile
            if camera_calibration_profile is _AUTHORITY_UNSET and current is not None
            else (None if camera_calibration_profile is _AUTHORITY_UNSET else camera_calibration_profile)
        )
        evidence = (
            current.metric_qualification_evidence
            if metric_qualification_evidence is _AUTHORITY_UNSET and current is not None
            else (None if metric_qualification_evidence is _AUTHORITY_UNSET else metric_qualification_evidence)
        )
        if calibration is not None and not isinstance(calibration, CameraCalibrationProfile):
            raise TypeError("camera_calibration_profile must be CameraCalibrationProfile or None.")
        if evidence is not None and not isinstance(evidence, MetricQualificationEvidence):
            raise TypeError("metric_qualification_evidence must be MetricQualificationEvidence or None.")
        if calibration is None and evidence is None:
            snapshot = None
        else:
            scale_profile_id = (
                current.metric_assessment_scale_profile_id
                if current is not None
                and evidence is current.metric_qualification_evidence
                and current.metric_assessment_scale_profile_id
                else None
            )
            snapshot = build_persistent_scene_authority_snapshot(
                scene,
                camera_calibration_profile=calibration,
                metric_qualification_evidence=evidence,
                metric_assessment_scale_profile_id=scale_profile_id,
            )
        return self._persistence_controller.save_scene_with_authority(
            output_dir,
            authority_snapshot=snapshot,
        )

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
        checkpoint = self._persistence_controller.load_checkpoint(path)
        self._set_engine_resume_preparation(None)
        if self._engine_resume_context is not None:
            self._validate_engine_resume_context()
        return checkpoint

    def set_engine_resume_context(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        available_payload_refs: Collection[str],
    ) -> ReconstructionResumeCompatibilityResult | None:
        """Register the current generic pipeline identity for bounded G3 resume validation."""

        self._require_idle_for_persistence("set reconstruction resume context")
        self._set_engine_resume_context(
            ReconstructionResumeContext(
                plan=plan,
                providers=providers,
                available_payload_refs=available_payload_refs,
            )
        )
        self._set_engine_resume_preparation(None)
        if self.engine_checkpoint is None:
            return None
        return self._validate_engine_resume_context()

    def clear_engine_resume_context(self) -> None:
        self._require_idle_for_persistence("clear reconstruction resume context")
        self._set_engine_resume_context(None)
        self._set_engine_resume_preparation(None)

    def check_engine_resume_compatibility(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        available_payload_refs: Collection[str],
    ) -> ReconstructionResumeCompatibilityResult:
        self._require_idle_for_persistence("check reconstruction resume compatibility")
        self._set_engine_resume_context(
            ReconstructionResumeContext(
                plan=plan,
                providers=providers,
                available_payload_refs=available_payload_refs,
            )
        )
        self._set_engine_resume_preparation(None)
        return self._validate_engine_resume_context()

    def prepare_engine_resume(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        available_payload_refs: Collection[str],
    ) -> ReconstructionResumePreparation:
        self._require_idle_for_persistence("prepare reconstruction resume")
        self._set_engine_resume_context(
            ReconstructionResumeContext(
                plan=plan,
                providers=providers,
                available_payload_refs=available_payload_refs,
            )
        )
        preparation = self._persistence_controller.prepare_resume(
            plan=plan,
            providers=providers,
            available_payload_refs=available_payload_refs,
        )
        self._set_engine_resume_preparation(preparation)
        return preparation

    def prepare_loaded_engine_resume(self) -> ReconstructionResumePreparation:
        """Prepare the loaded checkpoint from the last validated current context."""

        self._require_idle_for_persistence("prepare reconstruction resume")
        context = self._engine_resume_context
        if context is None:
            raise ReconstructionValidationError(
                "No current reconstruction resume context is registered."
            )
        if self.session_model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT:
            raise ReconstructionValidationError(
                "Cannot prepare reconstruction resume while session identity evidence conflicts."
            )
        return self.prepare_engine_resume(
            plan=context.plan,
            providers=context.providers,
            available_payload_refs=context.available_payload_refs,
        )

    def clear_engine_checkpoint(self) -> None:
        self._require_idle_for_persistence("clear the reconstruction checkpoint")
        self._persistence_controller.clear_checkpoint()
        self._set_engine_resume_preparation(None)

    def _validate_engine_resume_context(self) -> ReconstructionResumeCompatibilityResult:
        context = self._engine_resume_context
        if context is None:
            raise ReconstructionValidationError(
                "No current reconstruction resume context is registered."
            )
        return self._persistence_controller.check_resume_compatibility(
            plan=context.plan,
            providers=context.providers,
            available_payload_refs=context.available_payload_refs,
        )

    def _set_engine_resume_context(self, context: ReconstructionResumeContext | None) -> None:
        if context == self._engine_resume_context:
            return
        self._engine_resume_context = context
        self.resume_context_changed.emit(context)

    def _set_engine_resume_preparation(
        self, preparation: ReconstructionResumePreparation | None
    ) -> None:
        if preparation is self._engine_resume_preparation:
            return
        self._engine_resume_preparation = preparation
        self.resume_preparation_changed.emit(preparation)

    def _on_offline_state_changed(self, state: OfflineReconstructionRunState) -> None:
        # A newly-starting run must never inherit stage/progress/result presentation
        # from a previous run. Clear only passive product-layer caches; runtime
        # ownership remains with OfflineReconstructionController.
        if state is OfflineReconstructionRunState.STARTING:
            self._offline_telemetry_relay.begin(
                'offline_sfm', self._offline_controller.session_id, None,
                self._offline_controller.output_dir)
            self._latest_offline_workspace_event = None
            self._last_offline_result_handoff = None
            self._clear_progressive_sparse_snapshot()
        elif state in {
            OfflineReconstructionRunState.COMPLETED,
            OfflineReconstructionRunState.FAILED,
            OfflineReconstructionRunState.CANCELLED,
        }:
            self._offline_telemetry_relay.poll()
            self._offline_telemetry_relay.close()
            self._clear_progressive_sparse_snapshot()
        self.offline_state_changed.emit(state)

    def _on_progressive_sparse_snapshot_received(self, snapshot: object) -> None:
        if not isinstance(snapshot, progressive_sparse_contract.ProgressiveSfmSnapshot):
            self.offline_log_received.emit(
                "Rejected non-contract progressive sparse-SfM snapshot at controller boundary."
            )
            return
        active_session_id = self._offline_controller.session_id
        if not active_session_id or snapshot.session_id != active_session_id:
            self.offline_log_received.emit(
                "Rejected stale/cross-session progressive sparse-SfM snapshot at controller boundary."
            )
            return
        if self._offline_controller.state not in {
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        }:
            self.offline_log_received.emit(
                "Rejected progressive sparse-SfM snapshot outside an active offline run."
            )
            return
        previous = self._latest_progressive_sparse_snapshot
        if previous is not None:
            try:
                _validate_progressive_sparse_transition(previous, snapshot)
            except ReconstructionValidationError as exc:
                self.offline_log_received.emit(
                    f"Rejected stale progressive sparse-SfM controller update: {exc}"
                )
                return
        self._latest_progressive_sparse_snapshot = snapshot
        self.progressive_sparse_snapshot_changed.emit(snapshot)

    def _clear_progressive_sparse_snapshot(self) -> None:
        if self._latest_progressive_sparse_snapshot is None:
            return
        self._latest_progressive_sparse_snapshot = None
        self.progressive_sparse_snapshot_changed.emit(None)

    def _on_offline_event_received(self, payload: object) -> None:
        # Preserve the existing public forwarding contract exactly. I6.1B only
        # retains a mapping copy for passive workspace composition.
        if isinstance(payload, Mapping):
            self._latest_offline_workspace_event = dict(payload)
        self.offline_event_received.emit(payload)
        self._emit_offline_workspace_changed()

    def _emit_offline_workspace_changed(self, *_args: object) -> None:
        snapshot = self.offline_workspace_snapshot
        if snapshot == self._last_emitted_offline_workspace_snapshot:
            return
        self._last_emitted_offline_workspace_snapshot = snapshot
        self.offline_workspace_changed.emit(snapshot)

    def _emit_reconstruction_advisor_changed(self, *_args: object) -> None:
        state = self.reconstruction_advisor_state
        if state == self._last_emitted_reconstruction_advisor_state:
            return
        self._last_emitted_reconstruction_advisor_state = state
        self.reconstruction_advisor_changed.emit(state)

    def _reconcile_reconstruction_provider_selection(self, *_args: object) -> None:
        state = build_reconstruction_provider_selection_state(
            advisor_state=self.reconstruction_advisor_state,
            selected_provider_id=self._manual_reconstruction_provider_id,
            previous_status=self._provider_selection_status,
            previous_reason_codes=self._provider_selection_reason_codes,
        )
        if (
            self._manual_reconstruction_provider_id
            and state.status is ManualProviderSelectionStatus.CLEARED_INVALID
        ):
            self._manual_reconstruction_provider_id = ""
            self._provider_selection_status = ManualProviderSelectionStatus.CLEARED_INVALID
            self._provider_selection_reason_codes = state.reason_codes
        elif state.status is ManualProviderSelectionStatus.SELECTED:
            self._provider_selection_status = ManualProviderSelectionStatus.SELECTED
            self._provider_selection_reason_codes = ()
        self._emit_reconstruction_provider_selection_changed()

    def _emit_reconstruction_frame_view_plan_changed(self, *_args: object) -> None:
        state = self.reconstruction_frame_view_plan_state
        if state == self._last_emitted_reconstruction_frame_view_plan_state:
            return
        self._last_emitted_reconstruction_frame_view_plan_state = state
        self.reconstruction_frame_view_plan_changed.emit(state)

    def _emit_reconstruction_provider_selection_changed(self, *_args: object) -> None:
        state = self.reconstruction_provider_selection_state
        if state == self._last_emitted_reconstruction_provider_selection_state:
            return
        self._last_emitted_reconstruction_provider_selection_state = state
        self.reconstruction_provider_selection_changed.emit(state)

    def _emit_session_model_changed(self, *_args: object) -> None:
        # Cache the public state carried by ``live_state_changed`` before composing
        # the G1 model. This preserves backward compatibility with compact runtime
        # doubles that do not implement the full LiveCaptureSlamRuntime surface.
        if _args and isinstance(_args[0], LiveCaptureRuntimeState):
            self._session_live_state = _args[0]
        self.session_model_changed.emit(self.session_model)

    def _evaluate_start_guard(
        self,
        guard: ReconstructionStartGuard | None,
        *,
        operation: str,
    ) -> str:
        """Evaluate one application resource guard and return a normalized blocker."""

        if guard is None:
            return ""
        try:
            reason = guard()
        except Exception as exc:
            raise ReconstructionValidationError(
                f"{operation.capitalize()} preflight failed: {exc}"
            ) from exc
        if reason is None:
            return ""
        if not isinstance(reason, str):
            raise ReconstructionValidationError(
                f"{operation.capitalize()} preflight returned an invalid result."
            )
        return reason.strip()

    def _run_start_guard(
        self,
        guard: ReconstructionStartGuard | None,
        *,
        operation: str,
    ) -> None:
        reason = self._evaluate_start_guard(guard, operation=operation)
        if reason:
            raise ReconstructionValidationError(reason)

    @staticmethod
    def _raise_if_one_start_preflight_blocked(preflight: OfflineOneStartPreflight) -> None:
        if preflight.ready:
            return
        if OneStartBlockerCode.UNSUPPORTED_WORKFLOW in preflight.blocker_codes:
            raise ReconstructionValidationError(
                "Select Image sequence or Recorded video before starting offline reconstruction."
            )
        detail = preflight.blocker_details or "One Start admission evidence is incomplete."
        raise ReconstructionValidationError(f"Offline One Start preflight blocked: {detail}")

    def _require_offline_start_available(self) -> None:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot start an offline session."
            )
        if self._live_slam_controller.is_active:
            raise ReconstructionValidationError(
                "Live reconstruction is active; stop it before starting offline reconstruction."
            )
        if self._da3_canonical_controller.is_active:
            raise ReconstructionValidationError(
                "DA3 canonical integration is active; cancel it before starting offline reconstruction."
            )
        if self._offline_controller.is_active:
            raise ReconstructionValidationError(
                "An offline reconstruction job is already active or finalizing."
            )
        if self._dense_controller.is_active:
            raise ReconstructionValidationError(
                'Explicit dense reconstruction is active; cancel it before starting another session.'
            )

    def _commit_offline_start(
        self,
        configuration: OfflineReconstructionConfiguration,
        *,
        session_id: str | None = None,
    ) -> str:
        """Commit an already-admitted offline launch without re-running resource guards."""

        self._require_offline_start_available()
        sid = self._offline_controller.start(configuration, session_id=session_id)
        self._emit_session_model_changed()
        return sid

    def _require_idle_for_persistence(self, operation: str) -> None:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                f"Cannot {operation}: reconstruction controller is shut down."
            )
        if self.is_active:
            raise ReconstructionValidationError(
                f"Cannot {operation} while a reconstruction session is active."
            )

    def _on_da3_canonical_result_ready(self, payload: object, output_dir: str) -> None:
        if isinstance(payload, dict):
            try:
                self._persistence_controller.accept_offline_result(payload, output_dir)
            except (OSError, TypeError, ValueError, ReconstructionValidationError) as exc:
                self.error_occurred.emit(
                    f"DA3 canonical persistence bridge rejected result: {exc}"
                )
        self.da3_canonical_result_ready.emit(payload, output_dir)

    def _on_offline_result_ready(self, payload: object, output_dir: str) -> None:
        # The admitted terminal result supersedes every provisional I6.6 snapshot.
        self._clear_progressive_sparse_snapshot()
        if not isinstance(payload, dict):
            self.error_occurred.emit(
                "Offline reconstruction result handoff rejected a non-mapping payload."
            )
            return
        session_id = self._offline_controller.session_id
        admitted_output_dir = self._offline_controller.output_dir
        if not session_id or admitted_output_dir is None:
            self.error_occurred.emit(
                "Offline reconstruction result handoff has no active admitted session/output identity."
            )
            return
        try:
            handoff = build_offline_result_handoff(
                payload,
                output_dir,
                expected_session_id=session_id,
                expected_output_dir=admitted_output_dir,
            )
            self._persistence_controller.accept_offline_result(payload, output_dir)
        except (OSError, TypeError, ValueError, ReconstructionValidationError) as exc:
            self.error_occurred.emit(
                f"Offline reconstruction result handoff rejected result: {exc}"
            )
            return
        self._last_offline_result_handoff = handoff
        self.offline_result_handoff_ready.emit(handoff)
        # Preserve the historical UI/product signal, but only after the result has
        # passed I6.2C identity/provenance admission and persistence acceptance.
        self.offline_result_ready.emit(payload, output_dir)

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._dense_controller.shutdown()
        self._da3_canonical_controller.shutdown()
        self._offline_controller.shutdown()
        self._offline_telemetry_relay.close()
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
