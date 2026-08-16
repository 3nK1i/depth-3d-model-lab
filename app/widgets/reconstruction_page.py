"""Reconstruction workspace through Phase 8.9A2.

C1-C5 established the live viewer. D1-D5 added bounded offline sparse-SfM,
camera/health/coverage inspection, and semantic-object presentation. E1 binds
the active scene/object inspection target. E2 presents already-computed
measurement values. E3 presents source-authored measurement validity and
uncertainty. E4 presents authoritative inspection/QC reports. E5 adds the
reference-comparison workflow around an explicit reference identity and existing
Phase 8.7C comparison evidence without rerunning registration or tolerances. F1
adds metadata-only appearance-representation controls over persisted Gaussian/NeRF
artifacts. F2 adds static visualization of an already-produced backend Gaussian
preview. F3 adds the corresponding backend-authored NeRF preview without
executing its isolated runtime. F4 coordinates user-driven switching between
those qualified appearance views. F5 exposes bounded resource state and an
explicit canonical fallback without changing any reconstruction authority. G1
composes the existing runtime/scene/checkpoint evidence into one generic session
model without adding save/reload, resume, recovery, or domain-specific behavior. G2
adds explicit idle-only persistent-scene load/save/reload interaction on top of
the existing persistence controller. G3 exposes integrity-checked whole-engine checkpoint loading
and exact-context resume preparation without automatically launching resumed execution. G4 adds
explicit failure/recovery guidance, observes bounded live auto-recovery, and offers only a guarded
fresh retry from the current source; checkpoint execution remains outside G4. G5 adds compact
progressive disclosure to the settings sidebar without changing any reconstruction behavior.
Phase 8.9A2 adds a bounded product-export interaction over the already-qualified A1 export
contract. It exports only materialized, session-matched reconstruction evidence and remains
distinct from persistent-scene save/reload. QC report productization remains Phase 8.9B.
Phase 9.3A adds a passive canonical-scene presentation summary over the existing persistent
scene and Phase 9.1H authority sidecar. Phase 9.3B renders the integrity-bound persistent
canonical surface. Phase 9.3C adds the canonical reconstruction world frame plus explicitly
persisted camera frustums and trajectory without inventing missing poses or scale.
Phase 9.3F adds professional presentation-only scene-layer controls over the
already-qualified canonical, camera, semantic and inspection viewer resources.
Phase 9.4E adds backend-free interactive persisted Gaussian appearance, and
Phase 9.4F adds presentation-only Gaussian density/size/opacity plus a scene-layer
visibility control without changing canonical picking or any authority contract.
Phase 9.4G adds adaptive display budgets, coalesced slider updates, explicit CPU
staging/UI timing diagnostics, and deterministic release/reload of interactive
Gaussian viewer resources without launching a Gaussian backend.
Phase 9.5E adds isolated interactive persisted Nerfacto viewing. Phase 9.5F adds
presentation-only NeRF render-detail/opacity/refresh controls plus a NeRF appearance
scene-layer toggle while canonical geometry remains the only picking/measurement authority.
Phase 9.5G bounds interactive render requests to one in-flight plus one latest-view
pending request, surfaces isolated-worker CUDA/render diagnostics, retains at most two
render frames, and releases the external worker on hide/switch/clear/shutdown.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile
from time import monotonic

import numpy as np

from core.measurement.interactive_distance import InteractiveDistanceEstimate

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.camera_calibration_dialog import (
    CameraCalibrationDialog,
    default_camera_calibration_profile_root,
)
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.da3_canonical_integration_controller import DA3CanonicalRunState
from app.offline_reconstruction_visualization import (
    OfflineSparseVisualization,
    load_offline_sparse_visualization,
)
from app.offline_reconstruction_inspection import (
    build_offline_camera_inspection,
    format_offline_structure_summary,
)
from app.offline_reconstruction_quality import (
    OfflineSfmHealthPresentation,
    build_offline_sfm_health_presentation,
    unavailable_offline_sfm_health,
)
from app.offline_reconstruction_spatial import (
    OfflineSparseSpatialCoverage,
    build_offline_sparse_spatial_coverage,
)
from app.offline_reconstruction_semantics import (
    OfflineSemanticPresentation,
    semantic_presentation_from_payload,
    semantic_presentation_from_persistent_scene,
)
from app.reconstruction_diagnostics import LiveTrackingDiagnosticsAccumulator
from app.reconstruction_canonical_scene_presentation import (
    CanonicalScenePresentation,
    build_canonical_scene_presentation,
)
from app.reconstruction_canonical_geometry import (
    CanonicalViewerMesh,
    load_canonical_viewer_mesh,
)
from app.reconstruction_canonical_cameras import (
    CanonicalCameraTrajectoryPresentation,
    load_persistent_canonical_camera_trajectory,
)
from app.reconstruction_canonical_semantics import (
    CanonicalSemanticScenePresentation,
    load_persistent_canonical_semantics,
)
from app.reconstruction_canonical_inspection_overlays import (
    CanonicalObjectInspectionOverlayPresentation,
    build_canonical_object_inspection_overlay,
)
from app.measurement_canonical_registration_presentation import (
    MeasurementCanonicalRegistrationPresentation,
)
from app.reconstruction_scene_layers import build_reconstruction_scene_layer_availability
from app.reconstruction_dense_provider_presentation import (
    DenseProviderEntry,
    DenseProviderPresentation,
    build_dense_provider_presentation,
)
from app.reconstruction_scene_runtime_health import (
    ReconstructionSceneRuntimeHealth,
    build_reconstruction_scene_bind_plan,
    build_reconstruction_scene_runtime_health,
)
from app.reconstruction_inspection_selection import (
    ReconstructionInspectionSelection,
    build_reconstruction_inspection_selection,
)
from app.reconstruction_measurement_presentation import (
    ReconstructionMeasurementPresentation,
    build_reconstruction_measurement_presentation,
    empty_reconstruction_measurement_text,
)
from app.reconstruction_measurement_validity_presentation import (
    ReconstructionMeasurementValidityPresentation,
    build_reconstruction_measurement_validity_presentation,
    empty_reconstruction_measurement_validity_text,
)
from app.reconstruction_inspection_qc_presentation import (
    ReconstructionInspectionQcPresentation,
    build_reconstruction_inspection_qc_presentation,
    empty_reconstruction_inspection_qc_text,
)
from app.reconstruction_reference_comparison import (
    ReconstructionReferenceComparisonPresentation,
    ReconstructionReferenceDescriptor,
    build_reconstruction_reference_comparison_presentation,
    empty_reference_comparison_text,
    load_reference_descriptor_json,
    reference_descriptor_from_payload,
)
from app.reconstruction_appearance_presentation import (
    AppearanceRepresentationEntry,
    ReconstructionAppearancePresentation,
    build_reconstruction_appearance_presentation,
    empty_appearance_representation_text,
)
from app.reconstruction_digital_twin_workspace import (
    DigitalTwinWorkspacePresentation,
    build_digital_twin_workspace_presentation,
)
from app.reconstruction_digital_twin_composition import (
    DigitalTwinSceneComposition,
    build_digital_twin_scene_composition,
)
from app.reconstruction_digital_twin_navigation import (
    DigitalTwinNavigationSync,
    build_digital_twin_navigation_sync,
)
from app.reconstruction_digital_twin_interaction import (
    DigitalTwinInteractionPresentation,
    build_digital_twin_interaction_presentation,
)
from app.reconstruction_digital_twin_resources import (
    DigitalTwinResourceCoordination,
    build_digital_twin_resource_coordination,
)
from app.reconstruction_estimated_metric_measurement import (
    EstimatedMetricScaleHint,
    ReconstructionRulerMetricRecord,
    build_reconstruction_ruler_metric_record,
    estimated_scale_from_ruler_reference,
)
from app.reconstruction_digital_twin_persistence import (
    DigitalTwinProcessingHistoryRecord,
    DigitalTwinWorkspaceState,
    build_digital_twin_workspace_state,
    export_digital_twin_state_bundle,
    read_optional_digital_twin_workspace_state,
    verify_digital_twin_state_bundle,
    write_digital_twin_workspace_state,
)
from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingCommand,
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingLifecycle,
    DigitalTwinProcessingSnapshot,
    idle_digital_twin_processing_snapshot,
)
from app.reconstruction_processing_job_controller import DigitalTwinProcessingJobController
from app.reconstruction_da3_full_scene import DA3_FULL_SCENE_PROTOCOL, build_full_scene_da3_job
from app.reconstruction_nerf_build import (
    NERF_BUILD_PROTOCOL,
    NERF_BUILD_PRESETS,
    NERF_DEFAULT_BUILD_PRESET,
    build_nerf_training_job,
)
from app.reconstruction_gaussian_visualization import (
    GaussianVisualizationPresentation,
    build_gaussian_visualization_presentation,
    empty_gaussian_visualization_text,
)
from app.reconstruction_gaussian_interactive import (
    InteractiveGaussianScene,
    load_interactive_gaussian_scene,
)
from app.reconstruction_gaussian_controls import (
    GAUSSIAN_DENSITY_PERCENT_OPTIONS,
    GAUSSIAN_OPACITY_PERCENT_MAX,
    GAUSSIAN_OPACITY_PERCENT_MIN,
    GAUSSIAN_SIZE_PERCENT_MAX,
    GAUSSIAN_SIZE_PERCENT_MIN,
    GaussianDisplayBuffers,
    GaussianDisplaySettings,
    apply_gaussian_display_settings,
)
from app.reconstruction_gaussian_resources import (
    GaussianViewerRuntimeHealth,
    build_gaussian_interactive_resource_policy,
    build_gaussian_viewer_runtime_health,
)
from app.reconstruction_nerf_visualization import (
    NerfVisualizationPresentation,
    build_nerf_visualization_presentation,
    empty_nerf_visualization_text,
)
from app.reconstruction_nerf_interactive import (
    InteractiveNerfPresentation,
    build_interactive_nerf_presentation,
    verify_interactive_nerf_integrity,
    turntable_state_to_nerf_render_request,
)
from app.reconstruction_nerf_controls import (
    NERF_DEFAULT_RENDER_PROFILE,
    NERF_OPACITY_PERCENT_MAX,
    NERF_OPACITY_PERCENT_MIN,
    NERF_RENDER_PROFILES,
    NerfDisplaySettings,
)
from app.reconstruction_nerf_resources import (
    NerfViewerRuntimeHealth,
    build_nerf_interactive_resource_policy,
    build_nerf_viewer_runtime_health,
)
from app.reconstruction_representation_switching import (
    RepresentationSwitchTarget,
    build_representation_switch_target,
    canonical_view_text,
)
from app.reconstruction_appearance_resources import (
    AppearanceResourceFallbackPresentation,
    build_appearance_resource_fallback_presentation,
    empty_appearance_resource_fallback_text,
)
from app.reconstruction_controller import (
    ReconstructionSourceSelection,
    ReconstructionWorkflow,
    default_reconstruction_output_root,
)
from app.reconstruction_session_model import (
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionModel,
)
from app.reconstruction_session_interaction import (
    ReconstructionSceneInteractionPresentation,
    build_reconstruction_scene_interaction_presentation,
)
from app.reconstruction_checkpoint_resume import (
    ReconstructionCheckpointResumePresentation,
    build_reconstruction_checkpoint_resume_presentation,
)
from app.reconstruction_failure_recovery import (
    ReconstructionFailureRecoveryPresentation,
    ReconstructionFailureRecoveryState,
    build_reconstruction_failure_recovery_presentation,
)
from app.widgets.reconstruction_settings_disclosure import ReconstructionSettingsDisclosureGroup
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.reconstruction_status import (
    ReconstructionStatusPresentation,
    format_elapsed_seconds,
    format_live_telemetry,
    present_live_state,
    present_offline_event,
    present_offline_state,
)
from app.theme import (
    PAGE_MARGIN_BOTTOM,
    PAGE_MARGIN_H,
    PAGE_MARGIN_TOP,
    PAGE_SPACING,
    SPLITTER_HANDLE,
)
from app.widgets.icon_button import IconButton
from app.widgets.toggle_switch import ToggleSwitch
from app.widgets.reconstruction_diagnostics_hud import ReconstructionDiagnosticsHud
from app.widgets.reconstruction_orientation_gizmo import ReconstructionOrientationGizmo
from app.widgets.reconstruction_appearance_preview import ReconstructionAppearancePreview
from app.widgets.reconstruction_nerf_worker_client import ReconstructionNerfWorkerClient
from app.widgets.viewport.vispy_canvas import ReconstructionCanvas
from core.reconstruction import (
    CanonicalSurfaceMeshExportSource,
    InspectionReportExportSource,
    CameraIntrinsicsPolicy,
    ENGINE_CHECKPOINT_MANIFEST,
    LiveCaptureRuntimeState,
    OfflineSparseExportSource,
    PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST,
    PersistentSceneAuthoritySnapshot,
    ReconstructionExportRequest,
    ReconstructionExportSource,
    ReconstructionCamera,
    export_reconstruction_bundle,
    load_live_camera_calibration,
    verify_reconstruction_export_bundle,
)
from core.reconstruction.calibration_application import apply_camera_calibration_profile
from core.reconstruction.calibration_management import CameraCalibrationProfileStore
from core.reconstruction.calibration_profile import (
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
)


class ReconstructionPage(QWidget):
    """Professional Reconstruction workspace shell with passive state binding."""

    inspection_target_changed = pyqtSignal(object)
    reconstruction_measurement_changed = pyqtSignal(object)
    reconstruction_measurement_validity_changed = pyqtSignal(object)
    reconstruction_inspection_qc_changed = pyqtSignal(object)
    reconstruction_reference_comparison_changed = pyqtSignal(object)
    appearance_representation_changed = pyqtSignal(object)
    gaussian_visualization_changed = pyqtSignal(object)
    nerf_visualization_changed = pyqtSignal(object)
    representation_switching_changed = pyqtSignal(object)
    appearance_resource_fallback_changed = pyqtSignal(object)
    digital_twin_workspace_changed = pyqtSignal(object)
    digital_twin_scene_composition_changed = pyqtSignal(object)
    digital_twin_navigation_changed = pyqtSignal(object)
    digital_twin_interaction_changed = pyqtSignal(object)
    estimated_metric_scale_changed = pyqtSignal(object)
    ruler_metric_record_changed = pyqtSignal(object)
    digital_twin_processing_changed = pyqtSignal(object)
    digital_twin_resource_activity_changed = pyqtSignal(bool)
    canonical_scene_presentation_changed = pyqtSignal(object)
    canonical_camera_presentation_changed = pyqtSignal(object)
    canonical_inspection_overlay_changed = pyqtSignal(object)
    canonical_measurement_registration_changed = pyqtSignal(object)
    dense_provider_presentation_changed = pyqtSignal(object)
    failure_recovery_changed = pyqtSignal(object)
    reconstruction_session_model_changed = pyqtSignal(object)
    viewport_fullscreen_requested = pyqtSignal()
    viewport_escape_requested = pyqtSignal()
    camera_calibration_profile_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("reconstructionPage")
        self._controller = None
        self._source_buttons: dict[ReconstructionWorkflow, QPushButton] = {}
        self._live_state = LiveCaptureRuntimeState.IDLE
        self._offline_state = OfflineReconstructionRunState.IDLE
        self._da3_canonical_state = DA3CanonicalRunState.IDLE
        self._live_camera: ReconstructionCamera | None = None
        self._live_calibration_path: Path | None = None
        self._live_camera_mode: str | None = None
        self._calibration_profile_store = CameraCalibrationProfileStore(
            default_camera_calibration_profile_root()
        )
        self._selected_calibration_profile: CameraCalibrationProfile | None = None
        self._live_calibration_application = None
        self._last_live_source_identity: tuple[str, str, str] | None = None
        self._last_workflow = ReconstructionWorkflow.NONE
        self._runtime_started_at: float | None = None
        self._runtime_elapsed_frozen = 0.0
        self._active_runtime_kind: str | None = None
        self._live_pose_count = 0
        self._live_keyframe_count = 0
        self._live_landmark_count: int | None = None
        self._last_runtime_error = ""
        self._scene_availability = ReconstructionSceneAvailability.NONE
        self._persistent_scene = None
        self._persistent_authority_snapshot: PersistentSceneAuthoritySnapshot | None = None
        self._persisted_metric_assessment = None
        self._canonical_scene_presentation: CanonicalScenePresentation | None = None
        self._interactive_measure_points_reconstruction: list[np.ndarray] = []
        self._interactive_distance_estimate: InteractiveDistanceEstimate | None = None
        self._estimated_metric_scale_hint: EstimatedMetricScaleHint | None = None
        self._ruler_metric_record: ReconstructionRulerMetricRecord | None = None
        self._last_ruler_reference_distance_reconstruction_units: float | None = None
        self._canonical_viewer_mesh: CanonicalViewerMesh | None = None
        self._canonical_viewer_error = ""
        self._canonical_camera_presentation: CanonicalCameraTrajectoryPresentation | None = None
        self._canonical_camera_error = ""
        self._canonical_semantic_presentation: CanonicalSemanticScenePresentation | None = None
        self._canonical_semantic_error = ""
        self._canonical_inspection_overlay_presentation: CanonicalObjectInspectionOverlayPresentation | None = None
        self._canonical_inspection_overlay_error = ""
        self._canonical_measurement_registration_presentation: MeasurementCanonicalRegistrationPresentation | None = None
        self._dense_provider_presentation: DenseProviderPresentation | None = None
        self._selected_dense_provider_id = ""
        self._scene_runtime_health: ReconstructionSceneRuntimeHealth | None = None
        self._scene_bind_count = 0
        self._same_scene_reload_count = 0
        self._last_scene_bind_latency_ms = 0.0
        self._last_scene_mesh_visual_reused = False
        self._last_scene_view_preserved = False
        self._last_scene_selection_preserved = False
        self._offline_sparse_export_source: OfflineSparseExportSource | None = None
        self._canonical_surface_export_source: CanonicalSurfaceMeshExportSource | None = None
        self._inspection_qc_export_source: InspectionReportExportSource | None = None
        self._reconstruction_export_source_error = ""
        self._last_reconstruction_export_dir: Path | None = None
        self._inspection_selection = build_reconstruction_inspection_selection()
        self._reconstruction_measurement_presentations: dict[
            tuple[str, str], ReconstructionMeasurementPresentation
        ] = {}
        self._reconstruction_measurement_presentation: ReconstructionMeasurementPresentation | None = None
        self._reconstruction_measurement_validity_presentations: dict[
            tuple[str, str], ReconstructionMeasurementValidityPresentation
        ] = {}
        self._reconstruction_measurement_validity_presentation: ReconstructionMeasurementValidityPresentation | None = None
        self._reconstruction_inspection_qc_reports: dict[str, tuple[object, str]] = {}
        self._reconstruction_inspection_qc_presentation: ReconstructionInspectionQcPresentation | None = None
        self._reconstruction_reference_descriptors: dict[str, ReconstructionReferenceDescriptor] = {}
        self._reconstruction_reference_comparison_presentation: ReconstructionReferenceComparisonPresentation | None = None
        self._appearance_presentation: ReconstructionAppearancePresentation | None = None
        self._selected_appearance_artifact_id = ""
        self._gaussian_visualization_presentation: GaussianVisualizationPresentation | None = None
        self._gaussian_preview_visible = False
        self._interactive_gaussian_scene: InteractiveGaussianScene | None = None
        self._interactive_gaussian_visible = False
        self._gaussian_display_settings = GaussianDisplaySettings()
        self._gaussian_display_buffers: GaussianDisplayBuffers | None = None
        self._gaussian_resource_policy = build_gaussian_interactive_resource_policy()
        self._gaussian_runtime_health: GaussianViewerRuntimeHealth | None = None
        self._gaussian_load_count = 0
        self._gaussian_release_count = 0
        self._last_gaussian_load_latency_ms = 0.0
        self._last_gaussian_apply_latency_ms = 0.0
        self._last_gaussian_total_count = 0
        self._last_gaussian_source_display_count = 0
        self._gaussian_display_update_pending = False
        self._nerf_visualization_presentation: NerfVisualizationPresentation | None = None
        self._nerf_preview_visible = False
        self._interactive_nerf_presentation: InteractiveNerfPresentation | None = None
        self._interactive_nerf_visible = False
        self._nerf_display_settings = NerfDisplaySettings()
        self._interactive_nerf_request_id = 0
        self._interactive_nerf_latest_frame_path = ""
        self._interactive_nerf_last_render_ms = 0.0
        self._interactive_nerf_session_dir: Path | None = None
        self._nerf_resource_policy = build_nerf_interactive_resource_policy()
        self._nerf_runtime_health: NerfViewerRuntimeHealth | None = None
        self._nerf_worker_start_count = 0
        self._nerf_worker_release_count = 0
        self._nerf_release_pending = False
        self._nerf_worker_telemetry: dict[str, object] = {}
        self._nerf_worker_client = ReconstructionNerfWorkerClient(self)
        self._nerf_worker_client.ready.connect(self._on_interactive_nerf_worker_ready)
        self._nerf_worker_client.frame_ready.connect(self._on_interactive_nerf_frame_ready)
        self._nerf_worker_client.failed.connect(self._on_interactive_nerf_worker_failed)
        self._nerf_worker_client.state_changed.connect(self._on_interactive_nerf_worker_state_changed)
        self._representation_switch_target: RepresentationSwitchTarget | None = None
        self._appearance_resource_fallback_presentation: AppearanceResourceFallbackPresentation | None = None
        self._appearance_fallback_reason = ""
        self._digital_twin_workspace_presentation: DigitalTwinWorkspacePresentation | None = None
        self._digital_twin_scene_composition: DigitalTwinSceneComposition | None = None
        self._digital_twin_navigation_sync: DigitalTwinNavigationSync | None = None
        self._digital_twin_interaction_presentation: DigitalTwinInteractionPresentation | None = None
        self._digital_twin_processing_snapshot: DigitalTwinProcessingSnapshot = idle_digital_twin_processing_snapshot()
        self._digital_twin_resource_coordination: DigitalTwinResourceCoordination | None = None
        self._digital_twin_resource_activity_active = False
        # Phase 9.6F persists only scene-bound presentation state and terminal
        # build history. It never resumes a heavyweight job or changes scene authority.
        self._digital_twin_workspace_state: DigitalTwinWorkspaceState | None = None
        self._digital_twin_workspace_manifest: Path | None = None
        self._digital_twin_processing_history: DigitalTwinProcessingHistoryRecord | None = None
        self._last_digital_twin_state_export_dir: Path | None = None
        # Phase 9.6E4 UI focus state only. The first snapshot for each new E1
        # build job may bring the existing BUILD & PROCESSING disclosure into
        # view once; this never changes job/controller/backend state.
        self._digital_twin_processing_focus_job_id = ""
        self._digital_twin_processing_controller = DigitalTwinProcessingJobController(self)
        self._digital_twin_processing_controller.snapshot_changed.connect(
            self._on_digital_twin_processing_snapshot_changed
        )
        self._digital_twin_processing_controller.error_occurred.connect(
            self._on_digital_twin_processing_error
        )
        self._digital_twin_processing_controller.result_ready.connect(
            self._on_digital_twin_processing_result_ready
        )
        self._reconstruction_session_model: ReconstructionSessionModel | None = None
        self._scene_interaction_presentation: ReconstructionSceneInteractionPresentation | None = None
        self._checkpoint_resume_presentation: ReconstructionCheckpointResumePresentation | None = None
        self._failure_recovery_presentation: ReconstructionFailureRecoveryPresentation | None = None
        self._viewport_shutdown = False
        self._pending_visual_pose: np.ndarray | None = None
        self._pending_sparse_landmarks: tuple[np.ndarray, int, int] | None = None
        self._pending_keyframes: tuple[np.ndarray, tuple[str, ...], int, int] | None = None
        self._offline_visualization_path: Path | None = None
        self._offline_visualization_error = ""
        self._offline_visualization: OfflineSparseVisualization | None = None
        self._offline_camera_selection_index: int | None = None
        self._offline_sfm_health: OfflineSfmHealthPresentation | None = None
        self._offline_spatial_coverage: OfflineSparseSpatialCoverage | None = None
        self._offline_semantics: OfflineSemanticPresentation | None = None
        self._semantic_object_selection_index: int | None = None
        self._visualizer_fitted_once = False
        self._viewport_tool_buttons: dict[str, IconButton] = {}
        self._scene_layer_toggles: dict[str, ToggleSwitch] = {}
        self._live_diagnostics = LiveTrackingDiagnosticsAccumulator()
        self._viewport_focus_mode = False
        self._saved_body_splitter_sizes: list[int] | None = None
        self._build_ui()
        self._refresh_reconstruction_interactive_measurement_availability()
        self._visual_pose_timer = QTimer(self)
        self._visual_pose_timer.setSingleShot(True)
        self._visual_pose_timer.setInterval(33)
        self._visual_pose_timer.timeout.connect(self._flush_pending_visual_pose)
        self._gaussian_display_update_timer = QTimer(self)
        self._gaussian_display_update_timer.setSingleShot(True)
        self._gaussian_display_update_timer.setInterval(16)
        self._gaussian_display_update_timer.timeout.connect(self._flush_pending_gaussian_display_update)
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(500)
        self._telemetry_timer.timeout.connect(self._refresh_runtime_telemetry)
        self.set_live_state(LiveCaptureRuntimeState.IDLE)
        self.set_offline_state(OfflineReconstructionRunState.IDLE)
        self.set_scene_availability(ReconstructionSceneAvailability.NONE)
        self.set_checkpoint(None, "")
        self._apply_runtime_presentation(
            ReconstructionStatusPresentation(
                "Waiting for source",
                "Choose a Reconstruction workflow to begin.",
            )
        )

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        self._page_layout = outer
        outer.setContentsMargins(
            PAGE_MARGIN_H,
            PAGE_MARGIN_TOP,
            PAGE_MARGIN_H,
            PAGE_MARGIN_BOTTOM,
        )
        outer.setSpacing(PAGE_SPACING)

        self._header_card = self._build_header_card()
        outer.addWidget(self._header_card)

        body = QSplitter(Qt.Orientation.Horizontal)
        self._body_splitter = body
        body.setObjectName("reconstructionBodySplitter")
        body.setHandleWidth(SPLITTER_HANDLE)
        body.setChildrenCollapsible(False)

        main = QWidget()
        self._main_host = main
        main_layout = QVBoxLayout(main)
        self._main_layout = main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(PAGE_SPACING)

        top = QHBoxLayout()
        top.setSpacing(PAGE_SPACING)
        self._source_card = self._build_source_card()
        self._runtime_card = self._build_runtime_card()
        top.addWidget(self._source_card, stretch=1)
        top.addWidget(self._runtime_card, stretch=1)
        main_layout.addLayout(top)
        self._viewport_card = self._build_viewport_card()
        main_layout.addWidget(self._viewport_card, stretch=1)

        self._settings_panel = self._build_settings_panel()
        body.addWidget(main)
        body.addWidget(self._settings_panel)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        body.setSizes([1010, 320])
        outer.addWidget(body, stretch=1)

    def _build_header_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionHeaderCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 11, 16, 11)
        layout.setSpacing(12)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("Reconstruction")
        title.setObjectName("reconstructionTitle")
        subtitle = QLabel(
            "Live and offline 3D reconstruction workspace. Controller, isolation, persistence, and recovery foundations are ready."
        )
        subtitle.setObjectName("reconstructionSubtitle")
        subtitle.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addLayout(text, stretch=1)

        self._readiness_badge = QLabel("CONTROLLER READY")
        self._readiness_badge.setObjectName("reconstructionReadinessBadge")
        self._readiness_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._readiness_badge)
        return card

    def _build_source_card(self) -> QFrame:
        card = self._card("SOURCE & WORKFLOW")
        layout = card.layout()

        self._source_value = self._value_label("No reconstruction source selected")
        self._workflow_value = self._value_label("Choose a workflow")
        self._source_readiness_value = self._status_value()
        self._add_status_row(layout, "Source", self._source_value)
        self._add_status_row(layout, "Workflow", self._workflow_value)
        self._add_status_row(layout, "Readiness", self._source_readiness_value)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        button_specs = (
            (ReconstructionWorkflow.LIVE_SOURCE, "Live source", self._select_live_source),
            (ReconstructionWorkflow.IMAGE_SEQUENCE, "Image sequence", self._choose_image_sequence),
            (ReconstructionWorkflow.RECORDED_VIDEO, "Recorded video", self._choose_recorded_video),
        )
        for workflow, text, callback in button_specs:
            button = QPushButton(text)
            button.setObjectName("reconstructionWorkflowButton")
            button.setCheckable(True)
            button.setEnabled(False)
            button.setToolTip("Select this Reconstruction input workflow.")
            button.clicked.connect(callback)
            self._source_buttons[workflow] = button
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return card

    def _build_runtime_card(self) -> QFrame:
        card = self._card("RUNTIME STATUS")
        layout = card.layout()

        self._live_value = self._status_value()
        self._offline_value = self._status_value()
        self._da3_runtime_value = self._status_value()
        self._scene_value = self._status_value()
        self._stage_value = self._status_value()
        self._elapsed_value = self._value_label("—")
        self._add_status_row(layout, "Live SLAM", self._live_value)
        self._add_status_row(layout, "Offline", self._offline_value)
        self._add_status_row(layout, "DA3", self._da3_runtime_value)
        self._add_status_row(layout, "Scene", self._scene_value)
        self._add_status_row(layout, "Stage", self._stage_value)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("reconstructionProgressBar")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Waiting")
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setToolTip(
            "Stage-local progress when the backend exposes a meaningful finite fraction. "
            "Live tracking is intentionally shown as ongoing activity rather than a fake percentage."
        )
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)
        progress_row.addWidget(self._progress_bar, stretch=1)
        self._elapsed_value.setMinimumWidth(42)
        self._elapsed_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_row.addWidget(self._elapsed_value)
        layout.addLayout(progress_row)

        self._runtime_message = QLabel("Choose a Reconstruction workflow to begin.")
        self._runtime_message.setObjectName("reconstructionRuntimeMessage")
        self._runtime_message.setWordWrap(True)
        self._runtime_message.setProperty("statusKind", "neutral")
        layout.addWidget(self._runtime_message)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self._start_button = QPushButton("Start")
        self._start_button.setObjectName("reconstructionRuntimeAction")
        self._start_button.setProperty("actionRole", "primary")
        self._start_button.setToolTip("Start the selected Reconstruction workflow.")
        self._start_button.clicked.connect(self._start_runtime)
        controls.addWidget(self._start_button)

        self._pause_button = QPushButton("Pause")
        self._pause_button.setObjectName("reconstructionRuntimeAction")
        self._pause_button.setToolTip("Pause or resume a running live reconstruction session.")
        self._pause_button.clicked.connect(self._pause_or_resume_runtime)
        controls.addWidget(self._pause_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setObjectName("reconstructionRuntimeAction")
        self._stop_button.setToolTip("Stop live reconstruction or cancel an offline run.")
        self._stop_button.clicked.connect(self._stop_runtime)
        controls.addWidget(self._stop_button)
        layout.addLayout(controls)
        return card

    def _build_viewport_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionViewportCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("RECONSTRUCTION VIEWPORT")
        title.setObjectName("reconstructionSectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        self._viewport_state = QLabel("Awaiting scene")
        self._viewport_state.setObjectName("reconstructionViewportState")
        header.addWidget(self._viewport_state)
        layout.addLayout(header)

        host = QFrame()
        host.setObjectName("reconstructionViewportCanvasHost")
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self._reconstruction_canvas = ReconstructionCanvas(self)
        self._reconstruction_canvas.escape_pressed.connect(self._on_viewport_escape)
        self._reconstruction_canvas.measurement_cancel_requested.connect(
            self._cancel_reconstruction_interactive_measurement
        )
        self._reconstruction_canvas.measurement_point_picked.connect(
            self._on_reconstruction_interactive_measurement_point
        )
        self._reconstruction_canvas.interaction_started.connect(
            self._on_reconstruction_view_interaction_started
        )
        self._reconstruction_canvas.interaction_ended.connect(
            self._on_reconstruction_view_interaction_ended
        )
        grid.addWidget(self._reconstruction_canvas.native, 0, 0)

        self._viewport_placeholder = QWidget()
        self._viewport_placeholder.setObjectName("reconstructionViewportPlaceholder")
        self._viewport_placeholder.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        center = QVBoxLayout(self._viewport_placeholder)
        center.setContentsMargins(24, 24, 24, 24)
        center.setSpacing(6)
        # Keep the idle copy just below the world-origin triad: low enough to
        # avoid overlap, but not so low that it feels detached from the scene.
        center.addStretch(4)
        headline = QLabel("Reconstruction viewport")
        headline.setObjectName("reconstructionViewportHeadline")
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(headline)
        detail = QLabel(
            "World origin and reconstruction context share this viewport. Persistent canonical surface geometry is rendered here when an integrity-verified scene resource is available; appearance previews remain presentation-only overlays."
        )
        detail.setObjectName("reconstructionViewportHint")
        detail.setWordWrap(True)
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(detail)
        center.addStretch(3)
        grid.addWidget(self._viewport_placeholder, 0, 0)

        # F2 uses a reusable, backend-free image overlay for the already-produced
        # Gaussian preview render. It does not parse the Gaussian scene payload or
        # execute any Gaussian backend in the UI process. F3 can reuse the same overlay.
        self._appearance_preview = ReconstructionAppearancePreview(host)
        grid.addWidget(self._appearance_preview, 0, 0)

        tools = QFrame()
        tools.setObjectName("reconstructionViewportTools")
        tool_row = QHBoxLayout(tools)
        tool_row.setContentsMargins(5, 4, 5, 4)
        tool_row.setSpacing(2)

        group = QButtonGroup(tools)
        group.setExclusive(True)
        for icon, mode, tip in (
            ("orbit", "orbit", "Orbit reconstruction view"),
            ("pan", "pan", "Pan reconstruction view"),
            ("zoom", "zoom", "Zoom reconstruction view"),
            ("ruler", "measure", "Measure distance — click Point A, then Point B"),
        ):
            button = IconButton(icon, size=28, icon_size=15, tooltip=tip, checkable=True)
            button.clicked.connect(lambda _checked, value=mode: self._set_viewport_tool(value))
            group.addButton(button)
            self._viewport_tool_buttons[mode] = button
            tool_row.addWidget(button)
        self._viewport_tool_buttons["orbit"].setChecked(True)

        fit_button = IconButton("fit", size=28, icon_size=15, tooltip="Fit available scene geometry and camera context to view")
        fit_button.clicked.connect(self._fit_reconstruction_view)
        tool_row.addWidget(fit_button)
        focus_button = IconButton("camera", size=28, icon_size=15, tooltip="Focus current camera")
        focus_button.clicked.connect(self._focus_reconstruction_camera)
        tool_row.addWidget(focus_button)
        reset_button = IconButton("reset", size=28, icon_size=15, tooltip="Reset reconstruction view")
        reset_button.clicked.connect(self._reset_reconstruction_view)
        tool_row.addWidget(reset_button)
        fullscreen_button = IconButton(
            "fullscreen",
            size=28,
            icon_size=15,
            tooltip="Maximize reconstruction viewport",
            checkable=True,
        )
        fullscreen_button.clicked.connect(
            lambda _checked: self.viewport_fullscreen_requested.emit()
        )
        self._viewport_tool_buttons["fullscreen"] = fullscreen_button
        tool_row.addWidget(fullscreen_button)
        grid_button = IconButton("grid", size=28, icon_size=15, tooltip="Show or hide world grid", checkable=True)
        grid_button.setChecked(True)
        grid_button.toggled.connect(self._reconstruction_canvas.set_grid_visible)
        self._viewport_tool_buttons["grid"] = grid_button
        tool_row.addWidget(grid_button)
        trajectory_button = IconButton(
            "trajectory",
            size=28,
            icon_size=15,
            tooltip="Show or hide live camera trajectory",
            checkable=True,
        )
        trajectory_button.setChecked(True)
        trajectory_button.toggled.connect(self._reconstruction_canvas.set_trajectory_visible)
        self._viewport_tool_buttons["trajectory"] = trajectory_button
        tool_row.addWidget(trajectory_button)
        landmarks_button = IconButton(
            "landmarks",
            size=28,
            icon_size=15,
            tooltip="Show or hide live sparse SLAM landmarks",
            checkable=True,
        )
        landmarks_button.setChecked(True)
        landmarks_button.toggled.connect(self._reconstruction_canvas.set_sparse_landmarks_visible)
        self._viewport_tool_buttons["landmarks"] = landmarks_button
        tool_row.addWidget(landmarks_button)
        keyframes_button = IconButton(
            "keyframes",
            size=28,
            icon_size=15,
            tooltip="Show or hide historical SLAM keyframe cameras and persistent camera frustums",
            checkable=True,
        )
        keyframes_button.setChecked(True)
        keyframes_button.toggled.connect(self._reconstruction_canvas.set_keyframes_visible)
        self._viewport_tool_buttons["keyframes"] = keyframes_button
        tool_row.addWidget(keyframes_button)
        envelope_button = IconButton(
            "cube",
            size=28,
            icon_size=15,
            tooltip="Show or hide offline sparse presentation envelope",
            checkable=True,
        )
        envelope_button.setChecked(True)
        envelope_button.toggled.connect(self._reconstruction_canvas.set_offline_sparse_envelope_visible)
        self._viewport_tool_buttons["offline_envelope"] = envelope_button
        tool_row.addWidget(envelope_button)
        semantic_button = IconButton(
            "semantics",
            size=28,
            icon_size=15,
            tooltip="Show or hide semantic object context",
            checkable=True,
        )
        semantic_button.setChecked(True)
        semantic_button.setEnabled(False)
        semantic_button.toggled.connect(self._reconstruction_canvas.set_semantic_objects_visible)
        self._viewport_tool_buttons["semantic_objects"] = semantic_button
        tool_row.addWidget(semantic_button)
        diagnostics_button = IconButton(
            "diagnostics",
            size=28,
            icon_size=15,
            tooltip="Show or hide live tracking diagnostics",
            checkable=True,
        )
        diagnostics_button.setChecked(True)
        self._viewport_tool_buttons["diagnostics"] = diagnostics_button
        tool_row.addWidget(diagnostics_button)

        grid.addWidget(
            tools,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )

        self._interactive_measure_hud = QFrame(host)
        self._interactive_measure_hud.setObjectName("reconstructionInteractiveMeasureHud")
        self._interactive_measure_hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        measure_hud_layout = QVBoxLayout(self._interactive_measure_hud)
        measure_hud_layout.setContentsMargins(11, 7, 11, 7)
        measure_hud_layout.setSpacing(1)
        self._interactive_measure_primary = QLabel("Click Point A")
        self._interactive_measure_primary.setObjectName("reconstructionInteractiveMeasurePrimary")
        self._interactive_measure_primary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._interactive_measure_secondary = QLabel(
            "Canonical surface ruler • metres require qualified/provisional scale or an explicit estimate"
        )
        self._interactive_measure_secondary.setObjectName("reconstructionInteractiveMeasureSecondary")
        self._interactive_measure_secondary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        measure_hud_layout.addWidget(self._interactive_measure_primary)
        measure_hud_layout.addWidget(self._interactive_measure_secondary)
        self._interactive_measure_hud.hide()
        grid.addWidget(
            self._interactive_measure_hud,
            0,
            0,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        )

        self._diagnostics_hud = ReconstructionDiagnosticsHud(host)
        diagnostics_button.toggled.connect(self._sync_diagnostics_visibility)
        self._sync_diagnostics_visibility()
        grid.addWidget(
            self._diagnostics_hud,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self._viewport_safety = QLabel("VIEWER Z-UP • ARBITRARY GAUGE")
        self._viewport_safety.setObjectName("reconstructionViewportSafety")
        self._viewport_safety.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        grid.addWidget(
            self._viewport_safety,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )

        self._orientation_gizmo = ReconstructionOrientationGizmo(
            self._reconstruction_canvas.orientation_gizmo_axes,
            host,
        )
        grid.addWidget(
            self._orientation_gizmo,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        )
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)
        layout.addWidget(host, stretch=1)
        return card

    def _set_viewport_tool(self, mode: str) -> None:
        if mode == "measure":
            # Static F2/F3 preview images are not synchronized picking surfaces.
            # Fail closed to the canonical viewport before ruler interaction.
            if self._gaussian_preview_visible:
                self._hide_gaussian_preview()
            if self._nerf_preview_visible:
                self._hide_nerf_preview()
            if self._interactive_nerf_visible:
                self._nerf_state_value.setText(
                    "Canonical measurement active • NeRF appearance paused (frozen + dimmed)"
                )
            self._begin_reconstruction_interactive_measurement()
            return
        if self._reconstruction_canvas.interaction_mode == "measure":
            self._cancel_reconstruction_interactive_measurement(switch_to_orbit=False)
        self._reconstruction_canvas.set_interaction_mode(mode)
        if self._interactive_nerf_visible:
            self._request_interactive_nerf_render()
        self._refresh_digital_twin_interaction_presentation()

    def _fit_reconstruction_view(self) -> None:
        self._reconstruction_canvas.fit_scene()
        if self._interactive_nerf_visible:
            self._request_interactive_nerf_render()
        self._refresh_digital_twin_navigation_sync()

    def _focus_reconstruction_camera(self) -> None:
        self._reconstruction_canvas.focus_current_camera()
        if self._interactive_nerf_visible:
            self._request_interactive_nerf_render()
        self._refresh_digital_twin_navigation_sync()

    def _reset_reconstruction_view(self) -> None:
        self._reconstruction_canvas.reset_view()
        if self._interactive_nerf_visible:
            self._request_interactive_nerf_render()
        self._refresh_digital_twin_navigation_sync()

    def _on_reconstruction_view_interaction_started(self) -> None:
        if not self._interactive_nerf_visible:
            return
        if self._reconstruction_canvas.interaction_mode == "measure":
            return
        # During drag, reveal the canonical guide underneath. This keeps viewport
        # interaction immediate while the isolated renderer is sampled on release.
        self._appearance_preview.hide()
        refresh_text = (
            "NeRF updates on release"
            if self._nerf_display_settings.auto_refresh
            else "use Refresh view for NeRF update"
        )
        self._viewport_state.setText(f"Canonical navigation guide • {refresh_text}")

    def _on_reconstruction_view_interaction_ended(self) -> None:
        if self._interactive_nerf_visible:
            self._request_interactive_nerf_render()
            if not self._nerf_display_settings.auto_refresh:
                self._appearance_preview.show()
                self._viewport_state.setText("Interactive Nerfacto • manual refresh")
        # Phase 9.6C snapshots the settled shared viewport camera after navigation.
        # Gaussian already shares this scene camera; NeRF requests are derived from
        # it above. This update is presentation-only and creates no camera authority.
        self._refresh_digital_twin_navigation_sync()

    def _reconstruction_interactive_measurement_available(self) -> bool:
        return bool(self._reconstruction_canvas.canonical_mesh_loaded)

    def _refresh_reconstruction_interactive_measurement_availability(self) -> None:
        button = self._viewport_tool_buttons.get("measure")
        if button is None:
            return
        available = self._reconstruction_interactive_measurement_available()
        button.setEnabled(available)
        button.setToolTip(
            "Measure distance — click Point A, then Point B"
            if available
            else "Load an integrity-verified canonical surface to measure distance"
        )
        if not available and self._reconstruction_canvas.interaction_mode == "measure":
            self._cancel_reconstruction_interactive_measurement(switch_to_orbit=True)

    def _interactive_measurement_scene_id(self) -> str:
        if self._persistent_scene is not None:
            scene_id = str(getattr(self._persistent_scene, "scene_id", "")).strip()
            if scene_id:
                return scene_id
        presentation = self._canonical_scene_presentation
        if presentation is not None and presentation.available and presentation.scene_id:
            return presentation.scene_id
        mesh = self._canonical_viewer_mesh
        return "" if mesh is None else str(mesh.scene_id).strip()

    def _begin_reconstruction_interactive_measurement(self) -> None:
        if not self._reconstruction_interactive_measurement_available():
            self._refresh_reconstruction_interactive_measurement_availability()
            return
        self._interactive_measure_points_reconstruction = []
        self._interactive_distance_estimate = None
        self._ruler_metric_record = None
        self._reconstruction_canvas.clear_interactive_measurement_overlay()
        self._reconstruction_canvas.set_interaction_mode("measure")
        self._interactive_measure_primary.setText("Click Point A")
        self._interactive_measure_secondary.setText(
            "Click two canonical surface points • metres use scene scale or an explicit estimated-scale hint"
        )
        self._interactive_measure_primary.setToolTip("")
        self._interactive_measure_secondary.setToolTip("")
        self._interactive_measure_hud.show()
        self._refresh_estimated_metric_scale_presentation()
        self._refresh_digital_twin_interaction_presentation()

    def _apply_ruler_metric_record(self, record: ReconstructionRulerMetricRecord) -> None:
        previous = self._ruler_metric_record
        self._ruler_metric_record = record
        self._interactive_distance_estimate = record.current
        self._interactive_measure_primary.setText(record.primary_text)
        self._interactive_measure_secondary.setText(record.secondary_text)
        self._interactive_measure_primary.setToolTip(record.provenance_tooltip)
        self._interactive_measure_secondary.setToolTip(record.provenance_tooltip)
        if previous != record:
            self.ruler_metric_record_changed.emit(record)

    def _refresh_active_ruler_metric_record(self) -> None:
        if len(self._interactive_measure_points_reconstruction) != 2:
            return
        scene_id = self._interactive_measurement_scene_id()
        if not scene_id:
            return
        record = build_reconstruction_ruler_metric_record(
            scene_id=scene_id,
            point_a_reconstruction=self._interactive_measure_points_reconstruction[0],
            point_b_reconstruction=self._interactive_measure_points_reconstruction[1],
            canonical_scene=self._canonical_scene_presentation,
            estimated_scale_hint=self._estimated_metric_scale_hint,
            previous=self._ruler_metric_record,
        )
        self._apply_ruler_metric_record(record)

    def _on_reconstruction_interactive_measurement_point(self, point) -> None:
        if self._reconstruction_canvas.interaction_mode != "measure":
            return
        value = np.asarray(point, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            return
        if len(self._interactive_measure_points_reconstruction) >= 2:
            self._interactive_measure_points_reconstruction = []
            self._interactive_distance_estimate = None
            self._ruler_metric_record = None
        self._interactive_measure_points_reconstruction.append(np.array(value, copy=True))
        self._reconstruction_canvas.set_interactive_measurement_overlay(
            np.asarray(self._interactive_measure_points_reconstruction, dtype=np.float64)
        )
        if len(self._interactive_measure_points_reconstruction) == 1:
            self._interactive_measure_primary.setText("Point A set • click Point B")
            self._interactive_measure_secondary.setText(
                "Surface point locked to canonical geometry • Esc or right-click cancels"
            )
            self._interactive_measure_primary.setToolTip("")
            self._interactive_measure_secondary.setToolTip("")
            self._refresh_estimated_metric_scale_presentation()
            return

        scene_id = self._interactive_measurement_scene_id()
        if not scene_id:
            return
        record = build_reconstruction_ruler_metric_record(
            scene_id=scene_id,
            point_a_reconstruction=self._interactive_measure_points_reconstruction[0],
            point_b_reconstruction=self._interactive_measure_points_reconstruction[1],
            canonical_scene=self._canonical_scene_presentation,
            estimated_scale_hint=self._estimated_metric_scale_hint,
        )
        self._last_ruler_reference_distance_reconstruction_units = (
            record.current.distance_source_units
        )
        self._apply_ruler_metric_record(record)
        self._refresh_estimated_metric_scale_presentation()

    @property
    def estimated_metric_scale_hint(self) -> EstimatedMetricScaleHint | None:
        return self._estimated_metric_scale_hint

    @property
    def ruler_metric_record(self) -> ReconstructionRulerMetricRecord | None:
        return self._ruler_metric_record

    def set_estimated_metric_scale_hint(self, hint: EstimatedMetricScaleHint) -> None:
        """Bind one non-authoritative estimated scale to the active scene.

        Future automatic metric-depth scale estimators may call this same API; the
        Phase 9.6E0 UI currently creates a hint from an approximate canonical ruler
        reference. The hint never mutates persistent scene authority.
        """

        if not isinstance(hint, EstimatedMetricScaleHint):
            raise TypeError("hint must be EstimatedMetricScaleHint.")
        scene_id = self._interactive_measurement_scene_id()
        if not scene_id or hint.scene_id != scene_id:
            raise ValueError("Estimated metric scale hint must match the active canonical scene.")
        changed = hint != self._estimated_metric_scale_hint
        self._estimated_metric_scale_hint = hint
        self._refresh_estimated_metric_scale_presentation()
        self._refresh_active_ruler_metric_record()
        if changed:
            self.estimated_metric_scale_changed.emit(hint)

    def clear_estimated_metric_scale_hint(self) -> None:
        previous = self._estimated_metric_scale_hint
        self._estimated_metric_scale_hint = None
        self._refresh_estimated_metric_scale_presentation()
        self._refresh_active_ruler_metric_record()
        if previous is not None:
            self.estimated_metric_scale_changed.emit(None)

    def _set_estimated_metric_scale_from_last_ruler_interaction(self) -> None:
        native = self._last_ruler_reference_distance_reconstruction_units
        scene_id = self._interactive_measurement_scene_id()
        if native is None or native <= 0.0 or not scene_id:
            return
        approximate_m, accepted = QInputDialog.getDouble(
            self,
            "Estimated Metric Scale",
            (
                f"Last canonical ruler: {native:.4f} reconstruction units\n"
                "Approximate real-world distance (m):"
            ),
            1.0,
            0.001,
            1000000.0,
            3,
        )
        if not accepted:
            return
        uncertainty_percent, accepted = QInputDialog.getDouble(
            self,
            "Estimated Metric Scale",
            (
                "Estimated uncertainty of that physical reference (%):\n"
                "Use a conservative value when the reference is approximate."
            ),
            20.0,
            0.1,
            99.0,
            1,
        )
        if not accepted:
            return
        hint = estimated_scale_from_ruler_reference(
            scene_id=scene_id,
            distance_reconstruction_units=native,
            approximate_distance_m=approximate_m,
            reference_relative_uncertainty=uncertainty_percent / 100.0,
        )
        self.set_estimated_metric_scale_hint(hint)

    def _refresh_estimated_metric_scale_presentation(self) -> None:
        if not hasattr(self, "_estimated_ruler_scale_state_value"):
            return
        hint = self._estimated_metric_scale_hint
        canonical = self._canonical_scene_presentation
        canonical_scale_available = bool(
            canonical is not None and canonical.scale_unit_to_metre is not None
        )
        if hint is None:
            self._estimated_ruler_scale_state_value.setText(
                "Scene scale available • estimate not required"
                if canonical_scale_available
                else "No estimated scale"
            )
            self._estimated_ruler_scale_value.setText("—")
            self._estimated_ruler_reference_value.setText(
                "Use qualified/provisional scene scale"
                if canonical_scale_available
                else (
                    f"Last ruler: {self._last_ruler_reference_distance_reconstruction_units:.4f} reconstruction units"
                    if self._last_ruler_reference_distance_reconstruction_units is not None
                    else "Measure two canonical points first"
                )
            )
            self._estimated_ruler_scale_authority_value.setToolTip(
                "No session-side estimated scale is active. Canonical scene scale authority is unchanged."
            )
        else:
            self._estimated_ruler_scale_state_value.setText(
                "Estimate retained • scene scale takes precedence"
                if canonical_scale_available
                else "Estimated • uncalibrated • session-only"
            )
            self._estimated_ruler_scale_value.setText(hint.scale_text)
            self._estimated_ruler_reference_value.setText(hint.reference_text)
            self._estimated_ruler_scale_authority_value.setToolTip(
                f"{hint.provenance}\n"
                "Estimate only; canonical unit_to_metre authority is unchanged; metrology is not certified."
            )
        can_set = bool(
            self._interactive_measurement_scene_id()
            and self._last_ruler_reference_distance_reconstruction_units is not None
            and self._last_ruler_reference_distance_reconstruction_units > 0.0
            and not canonical_scale_available
        )
        self._estimated_ruler_scale_set_button.setEnabled(can_set)
        self._estimated_ruler_scale_clear_button.setEnabled(hint is not None)

    def _cancel_reconstruction_interactive_measurement(self, *, switch_to_orbit: bool = True) -> None:
        self._interactive_measure_points_reconstruction = []
        self._interactive_distance_estimate = None
        self._ruler_metric_record = None
        self._reconstruction_canvas.clear_interactive_measurement_overlay()
        self._interactive_measure_hud.hide()
        if switch_to_orbit and self._reconstruction_canvas.interaction_mode == "measure":
            orbit = self._viewport_tool_buttons.get("orbit")
            if orbit is not None:
                orbit.setChecked(True)
            self._reconstruction_canvas.set_interaction_mode("orbit")
        self._refresh_estimated_metric_scale_presentation()
        self._refresh_digital_twin_interaction_presentation()
        if self._interactive_nerf_visible and self._reconstruction_canvas.interaction_mode != "measure":
            self._request_interactive_nerf_render(force=True)

    def _on_viewport_escape(self) -> None:
        if self._reconstruction_canvas.interaction_mode == "measure":
            self._cancel_reconstruction_interactive_measurement(switch_to_orbit=True)
            return
        self.viewport_escape_requested.emit()

    @property
    def interactive_distance_estimate(self) -> InteractiveDistanceEstimate | None:
        return self._interactive_distance_estimate

    def set_viewport_focus_mode(self, enabled: bool) -> None:
        """Expand only the Reconstruction viewport without recreating its GL surface."""

        enabled = bool(enabled)
        if enabled == self._viewport_focus_mode:
            self.set_viewport_fullscreen_active(enabled)
            return

        if enabled:
            self._saved_body_splitter_sizes = self._body_splitter.sizes()
            self._header_card.hide()
            self._source_card.hide()
            self._runtime_card.hide()
            self._settings_panel.hide()
            self._body_splitter.setSizes([1, 0])
            self._page_layout.setContentsMargins(0, 0, 0, 0)
            self._page_layout.setSpacing(0)
            self._main_layout.setSpacing(0)
        else:
            self._header_card.show()
            self._source_card.show()
            self._runtime_card.show()
            self._settings_panel.show()
            if self._saved_body_splitter_sizes:
                self._body_splitter.setSizes(self._saved_body_splitter_sizes)
            self._page_layout.setContentsMargins(
                PAGE_MARGIN_H,
                PAGE_MARGIN_TOP,
                PAGE_MARGIN_H,
                PAGE_MARGIN_BOTTOM,
            )
            self._page_layout.setSpacing(PAGE_SPACING)
            self._main_layout.setSpacing(PAGE_SPACING)

        self._viewport_focus_mode = enabled
        self.set_viewport_fullscreen_active(enabled)

    def set_viewport_fullscreen_active(self, active: bool) -> None:
        button = self._viewport_tool_buttons.get("fullscreen")
        if button is None:
            return
        active = bool(active)
        button.setChecked(active)
        button.setToolTip(
            "Restore Reconstruction workspace"
            if active
            else "Maximize reconstruction viewport"
        )

    @property
    def viewport_focus_mode(self) -> bool:
        return self._viewport_focus_mode

    def _sync_diagnostics_visibility(self, _checked: bool | None = None) -> None:
        if not hasattr(self, "_diagnostics_hud"):
            return
        button = self._viewport_tool_buttons.get("diagnostics")
        requested = bool(button is not None and button.isChecked())
        live_context = (
            self._last_workflow is ReconstructionWorkflow.LIVE_SOURCE
            or self._live_state in {
                LiveCaptureRuntimeState.STARTING,
                LiveCaptureRuntimeState.RUNNING,
                LiveCaptureRuntimeState.PAUSED,
                LiveCaptureRuntimeState.RECOVERING,
                LiveCaptureRuntimeState.STOPPING,
                LiveCaptureRuntimeState.STOPPED,
                LiveCaptureRuntimeState.FAILED,
            }
        )
        self._diagnostics_hud.setVisible(requested and live_context)

    def _apply_offline_sfm_health(self, presentation: OfflineSfmHealthPresentation) -> None:
        self._offline_sfm_health = presentation
        self._offline_quality_status_value.setText(presentation.status_label)
        self._offline_quality_status_value.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._offline_quality_status_value)
        self._offline_quality_status_value.setToolTip(presentation.gate_tooltip)
        self._offline_quality_registration_value.setText(presentation.registration_text)
        self._offline_quality_reprojection_value.setText(presentation.reprojection_text)
        self._offline_quality_support_value.setText(presentation.support_text)
        self._offline_quality_gate_value.setText(presentation.gate_text)
        self._offline_quality_gate_value.setToolTip(presentation.gate_tooltip)
        self._offline_quality_registration_value.setToolTip(
            "Registered sparse-SfM cameras versus input keyframes. This is coverage, not metric accuracy."
        )
        self._offline_quality_reprojection_value.setToolTip(
            "Mean sparse feature reprojection residual in image pixels, as reported by the qualified SfM backend."
        )
        self._offline_quality_support_value.setToolTip(
            "Mean landmark track support and sparse landmark density. Values are operational SfM health telemetry."
        )
        self._offline_inspection_group.setVisible(True)

    def _reset_offline_sfm_health(self) -> None:
        self._offline_sfm_health = None
        if not hasattr(self, "_offline_quality_status_value"):
            return
        self._offline_quality_status_value.setText("—")
        self._offline_quality_status_value.setProperty("statusKind", "")
        self._refresh_style(self._offline_quality_status_value)
        self._offline_quality_status_value.setToolTip("")
        self._offline_quality_registration_value.setText("—")
        self._offline_quality_reprojection_value.setText("—")
        self._offline_quality_support_value.setText("—")
        self._offline_quality_gate_value.setText("Operational SfM health only • not metrology")
        self._offline_quality_gate_value.setToolTip(
            "D3 never promotes sparse-SfM health to metric accuracy or certified metrology."
        )

    def _apply_offline_spatial_coverage(self, presentation: OfflineSparseSpatialCoverage) -> None:
        self._offline_spatial_coverage = presentation
        self._offline_spatial_envelope_value.setText(presentation.envelope_text)
        self._offline_spatial_envelope_value.setToolTip(presentation.envelope_tooltip)
        self._offline_spatial_camera_span_value.setText(presentation.camera_span_text)
        self._offline_spatial_camera_span_value.setToolTip(
            "Axis-aligned span of registered camera centers in arbitrary reconstruction gauge units."
        )
        self._offline_spatial_standoff_value.setText(presentation.standoff_text)
        self._offline_spatial_standoff_value.setToolTip(
            "Camera-center distance to the robust sparse-SfM sample center. Values are gauge units, not metres."
        )
        self._offline_spatial_basis_value.setText(presentation.basis_text)
        self._offline_spatial_basis_value.setToolTip(
            "D4 uses the bounded D1 visualization sample only. The envelope is a presentation aid, not canonical geometry."
        )
        if presentation.has_envelope:
            self._reconstruction_canvas.set_offline_sparse_envelope(
                presentation.robust_min_reconstruction,
                presentation.robust_max_reconstruction,
            )
        else:
            self._reconstruction_canvas.clear_offline_sparse_envelope()
        self._offline_inspection_group.setVisible(True)

    def _reset_offline_spatial_coverage(self) -> None:
        self._offline_spatial_coverage = None
        if hasattr(self, "_offline_spatial_envelope_value"):
            self._offline_spatial_envelope_value.setText("—")
            self._offline_spatial_envelope_value.setToolTip("")
            self._offline_spatial_camera_span_value.setText("—")
            self._offline_spatial_standoff_value.setText("—")
            self._offline_spatial_basis_value.setText(
                "Sparse presentation envelope only • not object dimensions"
            )
        if hasattr(self, "_reconstruction_canvas"):
            self._reconstruction_canvas.clear_offline_sparse_envelope()

    def _activate_offline_inspection(self, visual: OfflineSparseVisualization) -> None:
        self._offline_visualization = visual
        self._offline_camera_selection_index = None
        self._offline_camera_combo.blockSignals(True)
        self._offline_camera_combo.clear()
        self._offline_camera_combo.addItem("Select camera…", -1)
        for index, camera_id in enumerate(visual.camera_ids):
            self._offline_camera_combo.addItem(f"{index + 1:02d} • {camera_id}", index)
        self._offline_camera_combo.setCurrentIndex(0)
        self._offline_camera_combo.blockSignals(False)
        self._offline_camera_combo.setEnabled(bool(visual.camera_ids))
        self._offline_structure_value.setText(format_offline_structure_summary(visual))
        self._offline_structure_value.setToolTip(
            "Bounded D1 visualization sample versus authoritative full sparse-SfM counts."
        )
        self._offline_camera_position_value.setText("—")
        self._offline_camera_delta_value.setText("—")
        self._offline_inspection_group.setVisible(True)
        self._refresh_offline_camera_navigation()
        self._refresh_inspection_selection()

    def _reset_offline_inspection(self, *, hide: bool = True) -> None:
        self._offline_visualization = None
        self._offline_camera_selection_index = None
        if hasattr(self, "_offline_camera_combo"):
            self._offline_camera_combo.blockSignals(True)
            self._offline_camera_combo.clear()
            self._offline_camera_combo.addItem("Select camera…", -1)
            self._offline_camera_combo.setCurrentIndex(0)
            self._offline_camera_combo.setEnabled(False)
            self._offline_camera_combo.blockSignals(False)
            self._offline_camera_position_value.setText("—")
            self._offline_camera_delta_value.setText("—")
            self._offline_structure_value.setText("No offline sparse SfM")
            self._offline_inspection_group.setVisible(
                (not hide)
                or self._offline_sfm_health is not None
                or self._offline_spatial_coverage is not None
            )
            self._refresh_offline_camera_navigation()
        if hasattr(self, "_reconstruction_canvas"):
            self._reconstruction_canvas.clear_offline_camera_selection()
        self._refresh_inspection_selection()

    def _on_offline_camera_selected(self, combo_index: int) -> None:
        visual = self._offline_visualization
        if visual is None:
            return
        item_data = self._offline_camera_combo.itemData(combo_index)
        camera_index = -1 if item_data is None else int(item_data)
        if camera_index < 0:
            self._offline_camera_selection_index = None
            self._reconstruction_canvas.clear_offline_camera_selection()
            self._offline_camera_position_value.setText("—")
            self._offline_camera_delta_value.setText("—")
            self._refresh_offline_camera_navigation()
            return
        try:
            inspection = build_offline_camera_inspection(visual, camera_index)
            self._reconstruction_canvas.select_offline_camera(camera_index)
        except (IndexError, TypeError, ValueError) as exc:
            self._offline_camera_selection_index = None
            self._reconstruction_canvas.clear_offline_camera_selection()
            self._offline_camera_position_value.setText("Unavailable")
            self._offline_camera_delta_value.setText("Unavailable")
            self._offline_camera_position_value.setToolTip(str(exc))
            self._refresh_offline_camera_navigation()
            return
        self._offline_camera_selection_index = camera_index
        self._offline_camera_position_value.setText(inspection.position_text)
        self._offline_camera_position_value.setToolTip(
            "Camera translation in the arbitrary reconstruction gauge; values are not metres."
        )
        self._offline_camera_delta_value.setText(inspection.delta_previous_text)
        self._offline_camera_delta_value.setToolTip(
            "Euclidean displacement from the previous registered camera in arbitrary SfM gauge units."
        )
        self._viewport_state.setToolTip(
            f"Selected registered camera {inspection.ordinal_text}: {inspection.camera_id}"
        )
        self._refresh_offline_camera_navigation()

    def _refresh_offline_camera_navigation(self) -> None:
        visual = self._offline_visualization
        selected = self._offline_camera_selection_index
        count = 0 if visual is None else len(visual.camera_ids)
        has_selection = selected is not None and 0 <= selected < count
        self._offline_focus_camera_button.setEnabled(has_selection)
        self._offline_clear_camera_button.setEnabled(has_selection)
        self._offline_prev_camera_button.setEnabled(has_selection and selected > 0)
        self._offline_next_camera_button.setEnabled(has_selection and selected + 1 < count)

    def _select_previous_offline_camera(self) -> None:
        if self._offline_camera_selection_index is None:
            return
        target = self._offline_camera_selection_index - 1
        if target >= 0:
            self._offline_camera_combo.setCurrentIndex(target + 1)

    def _select_next_offline_camera(self) -> None:
        visual = self._offline_visualization
        if visual is None or self._offline_camera_selection_index is None:
            return
        target = self._offline_camera_selection_index + 1
        if target < len(visual.camera_ids):
            self._offline_camera_combo.setCurrentIndex(target + 1)

    def _focus_selected_offline_camera(self) -> None:
        if self._offline_camera_selection_index is None:
            return
        self._reconstruction_canvas.focus_selected_offline_camera()

    def _clear_offline_camera_selection(self) -> None:
        if self._offline_camera_combo.count():
            self._offline_camera_combo.setCurrentIndex(0)
        else:
            self._offline_camera_selection_index = None
            self._reconstruction_canvas.clear_offline_camera_selection()
            self._refresh_offline_camera_navigation()

    def set_semantic_geometry(self, payload: object, *, source: str = "semantic_geometry") -> None:
        """Present an already-built semantic result without recomputing semantics."""

        # A direct/in-memory semantic payload is not automatically the persisted
        # semantic artifact bound to the active canonical scene.  Keep the D5
        # presentation path, but clear any prior 9.3D canonical-surface binding
        # rather than silently associating unrelated semantic evidence.
        self._reconstruction_canvas.clear_canonical_semantic_objects()
        self._canonical_semantic_presentation = None
        self._canonical_semantic_error = ""
        presentation = semantic_presentation_from_payload(payload, source=source)
        self._apply_semantic_presentation(presentation)

    def _apply_semantic_presentation(self, presentation: OfflineSemanticPresentation) -> None:
        if not isinstance(presentation, OfflineSemanticPresentation):
            raise TypeError("presentation must be OfflineSemanticPresentation.")
        self._offline_semantics = presentation
        self._semantic_object_selection_index = None
        self._semantic_object_combo.blockSignals(True)
        self._semantic_object_combo.clear()
        self._semantic_object_combo.addItem("Select object…", -1)
        for index, item in enumerate(presentation.objects):
            self._semantic_object_combo.addItem(
                f"{index + 1:02d} • {item.display_name}", index
            )
        self._semantic_object_combo.setCurrentIndex(0)
        self._semantic_object_combo.setEnabled(bool(presentation.objects))
        self._semantic_object_combo.blockSignals(False)
        self._semantic_structure_value.setText(presentation.structure_text)
        self._semantic_structure_value.setToolTip(
            "Bounded semantic-object presentation from existing Phase 8.7 semantic context."
        )
        self._semantic_role_value.setText("—")
        self._semantic_support_value.setText("—")
        self._semantic_quality_value.setText("—")
        if presentation.metric_geometry_status == "arbitrary":
            self._semantic_scale_value.setText("Arbitrary reconstruction gauge")
        elif presentation.metric_geometry_status == "provisional_metric":
            self._semantic_scale_value.setText("Provisional metric geometry • not certified")
        elif presentation.metric_geometry_status == "validated_metric":
            self._semantic_scale_value.setText("Validated metric geometry • metrology not certified")
        else:
            self._semantic_scale_value.setText("Scale semantics unavailable")
        self._semantic_authority_value.setText(presentation.authority_text)
        self._semantic_authority_value.setToolTip(
            "Semantic context labels geometry; inspection/QC and measurement retain their own authority."
        )
        if presentation.objects:
            self._viewport_placeholder.setVisible(False)
            lower = np.asarray(
                [item.bounds_min_reconstruction for item in presentation.objects], dtype=np.float64
            )
            upper = np.asarray(
                [item.bounds_max_reconstruction for item in presentation.objects], dtype=np.float64
            )
            centroids = np.asarray(
                [item.centroid_reconstruction for item in presentation.objects], dtype=np.float64
            )
            self._reconstruction_canvas.set_semantic_objects(
                bounds_min_reconstruction=lower,
                bounds_max_reconstruction=upper,
                centroids_reconstruction=centroids,
                object_ids=tuple(item.object_id for item in presentation.objects),
                total_count=presentation.total_object_count,
            )
        else:
            self._reconstruction_canvas.clear_semantic_objects()
        semantic_button = self._viewport_tool_buttons.get("semantic_objects")
        if semantic_button is not None:
            semantic_button.setEnabled(bool(presentation.objects))
            self._reconstruction_canvas.set_semantic_objects_visible(semantic_button.isChecked())
        self._semantic_objects_group.setVisible(True)
        self._refresh_scene_layer_controls()
        self._refresh_semantic_navigation()
        self._refresh_inspection_selection()
        self._refresh_viewport_state()

    def _reset_semantic_presentation(self, *, hide: bool = True) -> None:
        self._offline_semantics = None
        self._semantic_object_selection_index = None
        if hasattr(self, "_semantic_object_combo"):
            self._semantic_object_combo.blockSignals(True)
            self._semantic_object_combo.clear()
            self._semantic_object_combo.addItem("Select object…", -1)
            self._semantic_object_combo.setCurrentIndex(0)
            self._semantic_object_combo.setEnabled(False)
            self._semantic_object_combo.blockSignals(False)
            self._semantic_structure_value.setText("No semantic context")
            self._semantic_role_value.setText("—")
            self._semantic_support_value.setText("—")
            self._semantic_quality_value.setText("—")
            self._semantic_scale_value.setText("—")
            self._semantic_authority_value.setText(
                "Semantic context only • not inspection/QC decision"
            )
            self._semantic_objects_group.setVisible(not hide)
            self._refresh_semantic_navigation()
        if hasattr(self, "_reconstruction_canvas"):
            self._reconstruction_canvas.clear_semantic_objects()
            self._reconstruction_canvas.clear_canonical_semantic_objects()
        self._canonical_semantic_presentation = None
        self._canonical_semantic_error = ""
        self._canonical_inspection_overlay_presentation = None
        self._canonical_inspection_overlay_error = ""
        button = getattr(self, "_viewport_tool_buttons", {}).get("semantic_objects")
        if button is not None:
            button.setEnabled(False)
        self._refresh_scene_layer_controls()
        self._refresh_inspection_selection()

    def _on_semantic_object_selected(self, combo_index: int) -> None:
        presentation = self._offline_semantics
        if presentation is None:
            return
        data = self._semantic_object_combo.itemData(combo_index)
        index = -1 if data is None else int(data)
        if index < 0:
            self._semantic_object_selection_index = None
            self._reconstruction_canvas.clear_semantic_object_selection()
            self._semantic_role_value.setText("—")
            self._semantic_support_value.setText("—")
            self._semantic_quality_value.setText("—")
            if presentation.metric_geometry_status == "arbitrary":
                self._semantic_scale_value.setText("Arbitrary reconstruction gauge")
            elif presentation.metric_geometry_status == "provisional_metric":
                self._semantic_scale_value.setText("Provisional metric geometry • not certified")
            elif presentation.metric_geometry_status == "validated_metric":
                self._semantic_scale_value.setText("Validated metric geometry • metrology not certified")
            else:
                self._semantic_scale_value.setText("Scale semantics unavailable")
            self._semantic_authority_value.setText(presentation.authority_text)
            self._refresh_semantic_navigation()
            self._refresh_inspection_selection()
            self._refresh_digital_twin_interaction_presentation()
            return
        if index >= len(presentation.objects):
            return
        # Static backend previews are not camera-synchronized interaction surfaces.
        # Reveal the canonical viewport before presenting exact semantic membership.
        if self._gaussian_preview_visible:
            self._hide_gaussian_preview()
        if self._nerf_preview_visible:
            self._hide_nerf_preview()
        item = presentation.objects[index]
        self._semantic_object_selection_index = index
        self._reconstruction_canvas.select_semantic_object(index)
        self._semantic_role_value.setText(item.role_text)
        self._semantic_support_value.setText(item.support_text)
        self._semantic_quality_value.setText(item.quality_text)
        self._semantic_scale_value.setText(item.scale_text)
        self._semantic_authority_value.setText(item.authority_text)
        self._semantic_quality_value.setToolTip(
            "Semantic/model confidence and mesh-vertex quality are context signals, not measurement accuracy."
        )
        canonical_semantics = self._canonical_semantic_presentation
        if (
            canonical_semantics is not None
            and index < len(canonical_semantics.objects)
            and canonical_semantics.objects[index].object_id == item.object_id
        ):
            bound = canonical_semantics.objects[index]
            membership = (
                "Exact canonical-surface membership persisted and validated."
                if bound.surface_membership_available
                else "Legacy 3D semantic snapshot: exact surface membership was not persisted."
            )
            self._semantic_support_value.setToolTip(
                f"{membership}\n{bound.provenance_text}\n"
                "Semantic membership remains context only; canonical geometry, measurement, and QC authority are unchanged."
            )
        else:
            self._semantic_support_value.setToolTip(
                "Semantic support counts are context only; no canonical-surface association is currently validated."
            )
        self._viewport_state.setToolTip(
            f"Selected semantic object {item.object_id}: {item.class_name} ({item.semantic_role})."
        )
        self._refresh_semantic_navigation()
        self._refresh_inspection_selection()
        self._refresh_digital_twin_interaction_presentation()

    def _refresh_semantic_navigation(self) -> None:
        presentation = self._offline_semantics
        selected = self._semantic_object_selection_index
        count = 0 if presentation is None else len(presentation.objects)
        valid = selected is not None and 0 <= selected < count
        if hasattr(self, "_semantic_focus_button"):
            self._semantic_focus_button.setEnabled(valid)
            self._semantic_clear_button.setEnabled(valid)
            self._semantic_prev_button.setEnabled(valid and selected > 0)
            self._semantic_next_button.setEnabled(valid and selected + 1 < count)

    def _select_previous_semantic_object(self) -> None:
        if self._semantic_object_selection_index is None:
            return
        target = self._semantic_object_selection_index - 1
        if target >= 0:
            self._semantic_object_combo.setCurrentIndex(target + 1)

    def _select_next_semantic_object(self) -> None:
        presentation = self._offline_semantics
        if presentation is None or self._semantic_object_selection_index is None:
            return
        target = self._semantic_object_selection_index + 1
        if target < len(presentation.objects):
            self._semantic_object_combo.setCurrentIndex(target + 1)

    def _focus_selected_semantic_object(self) -> None:
        if self._semantic_object_selection_index is None:
            return
        self._reconstruction_canvas.focus_selected_semantic_object()
        if self._interactive_nerf_visible:
            self._request_interactive_nerf_render(force=True)
        self._refresh_digital_twin_navigation_sync()
        self._refresh_digital_twin_interaction_presentation()

    def _clear_semantic_object_selection(self) -> None:
        if self._semantic_object_combo.count():
            self._semantic_object_combo.setCurrentIndex(0)
        else:
            self._semantic_object_selection_index = None
            self._reconstruction_canvas.clear_semantic_object_selection()
            self._refresh_semantic_navigation()

    @property
    def inspection_selection(self) -> ReconstructionInspectionSelection:
        return self._inspection_selection

    def _refresh_inspection_selection(self) -> None:
        """Bind the active scene + existing D5 selection into the E1 target."""

        scene = self._persistent_scene
        scene_id = str(getattr(scene, "scene_id", "")).strip() if scene is not None else ""
        session_id = str(getattr(scene, "session_id", "")).strip() if scene is not None else ""
        scene_source = "persistent_scene" if scene is not None else ""
        canonical = bool(
            getattr(scene, "canonical_geometry_authoritative", False)
        ) if scene is not None else False

        if not session_id and self._offline_semantics is not None:
            session_id = self._offline_semantics.session_id
            scene_source = self._offline_semantics.source
        if not session_id and self._offline_visualization is not None:
            session_id = self._offline_visualization.session_id
            scene_source = "offline_sparse_reconstruction"

        selection = build_reconstruction_inspection_selection(
            scene_id=scene_id,
            session_id=session_id,
            scene_source=scene_source,
            scene_availability=getattr(self._scene_availability, "value", str(self._scene_availability)),
            canonical_geometry_authoritative=canonical,
            semantic_presentation=self._offline_semantics,
            selected_object_index=self._semantic_object_selection_index,
        )
        changed = selection != self._inspection_selection
        self._inspection_selection = selection

        if hasattr(self, "_inspection_scope_value"):
            self._inspection_scope_value.setText(selection.scope_text)
            self._inspection_scene_value.setText(selection.scene_text)
            self._inspection_scene_value.setToolTip(selection.scene_tooltip)
            self._inspection_object_value.setText(selection.object_text)
            self._inspection_object_value.setToolTip(selection.object_tooltip)
            self._inspection_geometry_value.setText(selection.geometry_text)
            self._inspection_state_value.setText(selection.state_text)
            self._inspection_authority_value.setText(selection.authority_text)

        self._refresh_reconstruction_measurement_presentation()
        self._refresh_canonical_measurement_registration_presentation()
        if changed:
            self.inspection_target_changed.emit(selection)

    @property
    def canonical_measurement_registration_presentation(
        self,
    ) -> MeasurementCanonicalRegistrationPresentation | None:
        return self._canonical_measurement_registration_presentation

    def set_canonical_measurement_registration_presentation(
        self,
        presentation: MeasurementCanonicalRegistrationPresentation | None,
    ) -> None:
        if presentation is not None and not isinstance(
            presentation, MeasurementCanonicalRegistrationPresentation
        ):
            raise TypeError(
                "presentation must be MeasurementCanonicalRegistrationPresentation or None."
            )
        changed = presentation != self._canonical_measurement_registration_presentation
        self._canonical_measurement_registration_presentation = presentation
        self._refresh_canonical_measurement_registration_presentation()
        if changed:
            self.canonical_measurement_registration_changed.emit(presentation)

    def clear_canonical_measurement_registration(self) -> None:
        self.set_canonical_measurement_registration_presentation(None)

    def _refresh_canonical_measurement_registration_presentation(self) -> None:
        presentation = self._canonical_measurement_registration_presentation
        if hasattr(self, "_canonical_registration_state_value"):
            if presentation is None:
                self._canonical_registration_state_value.setText("Unavailable")
                self._canonical_registration_binding_value.setText("No exact canonical binding")
                self._canonical_registration_dimensions_value.setText("No canonical dimension geometry")
                self._canonical_registration_uncertainty_value.setText("No placement uncertainty")
                self._canonical_registration_reason_value.setText(
                    "No qualified measurement→canonical registration"
                )
                self._canonical_registration_authority_value.setText(
                    "Original measurement remains source truth • canonical placement is additive"
                )
            else:
                self._canonical_registration_state_value.setText(presentation.status_text)
                self._canonical_registration_binding_value.setText(presentation.binding_text)
                self._canonical_registration_dimensions_value.setText(presentation.dimensions_text)
                self._canonical_registration_uncertainty_value.setText(presentation.uncertainty_text)
                self._canonical_registration_reason_value.setText(presentation.reason_text)
                self._canonical_registration_authority_value.setText(presentation.authority_text)

        canvas = getattr(self, "_reconstruction_canvas", None)
        if canvas is None:
            return
        canvas.clear_canonical_registered_dimensions(update_canvas=False)
        selection = self._inspection_selection
        if (
            presentation is not None
            and presentation.dimensions
            and selection.has_object
            and selection.object_id == presentation.canonical_object_id
            and self._persistent_scene is not None
            and presentation.scene_id == self._persistent_scene.scene_id
        ):
            starts = np.asarray(
                [item.start_reconstruction for item in presentation.dimensions], dtype=np.float64
            )
            ends = np.asarray(
                [item.end_reconstruction for item in presentation.dimensions], dtype=np.float64
            )
            canvas.set_canonical_registered_dimensions(
                scene_id=presentation.scene_id,
                object_id=presentation.canonical_object_id,
                starts_reconstruction=starts,
                ends_reconstruction=ends,
                labels=[item.value_text for item in presentation.dimensions],
            )
        self._refresh_scene_layer_controls()

    def select_semantic_object_by_id(self, object_id: str) -> bool:
        target = str(object_id).strip()
        presentation = self._offline_semantics
        if not target or presentation is None:
            return False
        for index, item in enumerate(presentation.objects):
            if item.object_id == target:
                self._semantic_object_combo.setCurrentIndex(index + 1)
                return True
        return False

    @property
    def reconstruction_measurement_presentation(self) -> ReconstructionMeasurementPresentation | None:
        return self._reconstruction_measurement_presentation

    def set_reconstruction_measurement_evidence(
        self,
        measurement_object: object,
        *,
        source: str = "phase7_5_measurement",
    ) -> None:
        """Attach already-computed measurement evidence to the active E1 object.

        The caller owns correspondence between the selected persistent semantic
        object and the Phase 7.5 measurement observation. E2 never guesses that
        association from class names, roles, or geometry.
        """

        presentation = build_reconstruction_measurement_presentation(
            self._inspection_selection,
            measurement_object,
            source=source,
        )
        key = (presentation.session_id, presentation.target_object_id)
        self._reconstruction_measurement_presentations[key] = presentation
        # Replacing E2 evidence invalidates any E3 qualification previously
        # attached to this exact target/provenance. The caller must reattach
        # source-authored validity evidence explicitly.
        self._reconstruction_measurement_validity_presentations.pop(key, None)
        self._refresh_reconstruction_measurement_presentation()

    def clear_reconstruction_measurement_evidence(self) -> None:
        self._reconstruction_measurement_presentations.clear()
        self._reconstruction_measurement_validity_presentations.clear()
        self._refresh_reconstruction_measurement_presentation()

    def _refresh_reconstruction_measurement_presentation(self) -> None:
        selection = self._inspection_selection
        session_key = selection.session_id or selection.scene_id
        key = (session_key, selection.object_id) if selection.has_object else None
        presentation = (
            self._reconstruction_measurement_presentations.get(key)
            if key is not None
            else None
        )
        changed = presentation != self._reconstruction_measurement_presentation
        self._reconstruction_measurement_presentation = presentation

        if not hasattr(self, "_measurement_target_value"):
            return
        if presentation is None:
            target, source, values, state, authority = empty_reconstruction_measurement_text(selection)
            self._measurement_target_value.setText(target)
            self._measurement_source_value.setText(source)
            self._measurement_source_value.setToolTip("")
            self._measurement_values_value.setText(values)
            self._measurement_state_value.setText(state)
            self._measurement_authority_value.setText(authority)
        else:
            self._measurement_target_value.setText(presentation.target_label)
            self._measurement_source_value.setText(presentation.source)
            self._measurement_source_value.setToolTip(presentation.provenance_tooltip)
            self._measurement_values_value.setText(presentation.values_text)
            self._measurement_state_value.setText(presentation.state_text)
            self._measurement_authority_value.setText(presentation.authority_text)

        if changed:
            self.reconstruction_measurement_changed.emit(presentation)
        self._refresh_reconstruction_measurement_validity_presentation()

    @property
    def reconstruction_measurement_validity_presentation(self) -> ReconstructionMeasurementValidityPresentation | None:
        return self._reconstruction_measurement_validity_presentation

    def set_reconstruction_measurement_validity_evidence(
        self,
        measurement_object: object,
        *,
        source: str = "phase7_5_measurement",
        context: object | None = None,
    ) -> None:
        """Attach E3 qualification to the exact E2-bound measurement evidence.

        E3 requires an existing E2 presentation for the same target and exact
        source observation. It never guesses correspondence, computes new
        uncertainty/reliability, runs QC, or changes reconstruction scale.
        """

        selection = self._inspection_selection
        session_key = selection.session_id or selection.scene_id
        key = (session_key, selection.object_id) if selection.has_object else None
        e2 = self._reconstruction_measurement_presentations.get(key) if key is not None else None
        if e2 is None:
            raise ValueError("E3 validity evidence requires existing E2 measurement evidence for the active target.")
        presentation = build_reconstruction_measurement_validity_presentation(
            selection,
            measurement_object,
            source=source,
            context=context,
        )
        if presentation.source != e2.source:
            raise ValueError("E3 validity source must match the exact E2 measurement source.")
        if presentation.source_observation_id != e2.source_observation_id:
            raise ValueError("E3 validity observation must match the exact E2 measurement observation.")
        self._reconstruction_measurement_validity_presentations[key] = presentation
        self._refresh_reconstruction_measurement_validity_presentation()

    def clear_reconstruction_measurement_validity_evidence(self) -> None:
        self._reconstruction_measurement_validity_presentations.clear()
        self._refresh_reconstruction_measurement_validity_presentation()

    def _refresh_reconstruction_measurement_validity_presentation(self) -> None:
        selection = self._inspection_selection
        session_key = selection.session_id or selection.scene_id
        key = (session_key, selection.object_id) if selection.has_object else None
        e2 = self._reconstruction_measurement_presentations.get(key) if key is not None else None
        presentation = (
            self._reconstruction_measurement_validity_presentations.get(key)
            if key is not None and e2 is not None
            else None
        )
        changed = presentation != self._reconstruction_measurement_validity_presentation
        self._reconstruction_measurement_validity_presentation = presentation

        if not hasattr(self, "_measurement_validity_qualified_value"):
            return
        if presentation is None:
            qualified, validity, reliability, uncertainty, scale, calibration, support, authority = (
                empty_reconstruction_measurement_validity_text(
                    selection,
                    has_e2_evidence=e2 is not None,
                )
            )
            self._measurement_validity_qualified_value.setText(qualified)
            self._measurement_validity_qualified_value.setToolTip("")
            self._measurement_validity_status_value.setText(validity)
            self._measurement_validity_reliability_value.setText(reliability)
            self._measurement_validity_uncertainty_value.setText(uncertainty)
            self._measurement_validity_scale_value.setText(scale)
            self._measurement_validity_calibration_value.setText(calibration)
            self._measurement_validity_support_value.setText(support)
            self._measurement_validity_reasons_value.setText("—")
            self._measurement_validity_authority_value.setText(authority)
        else:
            self._measurement_validity_qualified_value.setText(presentation.qualified_values_text)
            self._measurement_validity_qualified_value.setToolTip(presentation.provenance_tooltip)
            self._measurement_validity_status_value.setText(presentation.validity_text)
            self._measurement_validity_reliability_value.setText(presentation.reliability_text)
            self._measurement_validity_uncertainty_value.setText(presentation.uncertainty_text)
            self._measurement_validity_scale_value.setText(presentation.scale_text)
            self._measurement_validity_calibration_value.setText(presentation.calibration_text)
            self._measurement_validity_support_value.setText(presentation.support_text)
            self._measurement_validity_reasons_value.setText(presentation.reasons_text)
            self._measurement_validity_authority_value.setText(presentation.authority_text)

        if changed:
            self.reconstruction_measurement_validity_changed.emit(presentation)
        self._refresh_reconstruction_inspection_qc_presentation()

    @property
    def reconstruction_inspection_qc_presentation(self) -> ReconstructionInspectionQcPresentation | None:
        return self._reconstruction_inspection_qc_presentation

    def set_reconstruction_inspection_qc_evidence(
        self,
        inspection_report: object,
        *,
        source: str = "phase8_7c.inspection_report",
    ) -> None:
        """Attach an existing authoritative inspection/QC report to the active scene.

        E4 consumes the serialized Phase 8.7C contract only. It does not run
        registration, apply tolerance rules, infer PASS/FAIL from semantic or
        measurement confidence, or load/select references.
        """

        presentation = build_reconstruction_inspection_qc_presentation(
            self._inspection_selection, inspection_report, source=source
        )
        export_source = self._inspection_qc_export_source
        if export_source is not None and export_source.report is not inspection_report:
            # Display-only E4 evidence must never silently replace the exact report bound
            # to inspection-decision export authority.  Drop export qualification and
            # require the caller to bind a fresh A1 InspectionReportExportSource.
            self._inspection_qc_export_source = None
        self._reconstruction_inspection_qc_reports[presentation.session_id] = (
            inspection_report, str(source).strip()
        )
        self._refresh_reconstruction_inspection_qc_presentation()
        self._refresh_reconstruction_export_controls()

    def set_reconstruction_inspection_qc_export_source(
        self,
        source: InspectionReportExportSource | None,
        *,
        presentation_source: str = "phase8_7c.inspection_report",
    ) -> None:
        """Bind exact inspection-decision evidence to the existing product export.

        B reuses the A1 authority contract and the existing E4 presentation path.
        A plain mapping/report may still be displayed by E4, but it is not product-export
        authority.  Product QC export therefore requires the typed A1 source carrying the
        exact INSPECTION_REPORT pipeline artifact and its immutable InspectionReport.
        """

        if source is None:
            self._inspection_qc_export_source = None
            self._refresh_reconstruction_export_controls()
            return
        if not isinstance(source, InspectionReportExportSource):
            raise TypeError("source must be InspectionReportExportSource or None.")
        # Validate/bind the same report through the established E4 presentation contract
        # before exposing it to product export.  Session mismatch therefore fails before
        # mutating the export source.
        self.set_reconstruction_inspection_qc_evidence(
            source.report,
            source=presentation_source,
        )
        self._inspection_qc_export_source = source
        self._refresh_reconstruction_export_controls()

    def clear_reconstruction_inspection_qc_evidence(self) -> None:
        self._reconstruction_inspection_qc_reports.clear()
        self._inspection_qc_export_source = None
        self._refresh_reconstruction_inspection_qc_presentation()
        self._refresh_reconstruction_export_controls()

    def _refresh_reconstruction_inspection_qc_presentation(self) -> None:
        selection = self._inspection_selection
        session_key = (selection.session_id or selection.scene_id).strip() if selection.has_scene else ""
        attached = self._reconstruction_inspection_qc_reports.get(session_key) if session_key else None
        presentation = None
        if attached is not None:
            report, source = attached
            presentation = build_reconstruction_inspection_qc_presentation(
                selection, report, source=source
            )
        changed = presentation != self._reconstruction_inspection_qc_presentation
        self._reconstruction_inspection_qc_presentation = presentation
        self._refresh_canonical_inspection_overlay()

        if not hasattr(self, "_inspection_qc_target_value"):
            return
        if presentation is None:
            target, decision, findings, coverage, registration, comparison, authority = (
                empty_reconstruction_inspection_qc_text(selection)
            )
            self._inspection_qc_target_value.setText(target)
            self._inspection_qc_target_value.setToolTip("")
            self._inspection_qc_decision_value.setText(decision)
            self._inspection_qc_findings_value.setText(findings)
            self._inspection_qc_coverage_value.setText(coverage)
            self._inspection_qc_decisions_value.setText("—")
            self._inspection_qc_registration_value.setText(registration)
            self._inspection_qc_comparison_value.setText(comparison)
            self._inspection_qc_authority_value.setText(authority)
        else:
            self._inspection_qc_target_value.setText(presentation.target_label)
            self._inspection_qc_target_value.setToolTip(presentation.provenance_tooltip)
            self._inspection_qc_decision_value.setText(presentation.decision_text)
            self._inspection_qc_findings_value.setText(presentation.findings_text)
            self._inspection_qc_coverage_value.setText(presentation.coverage_text)
            self._inspection_qc_decisions_value.setText(presentation.decisions_text)
            self._inspection_qc_registration_value.setText(presentation.registration_text)
            self._inspection_qc_comparison_value.setText(presentation.comparison_text)
            self._inspection_qc_authority_value.setText(presentation.authority_text)

        if changed:
            self.reconstruction_inspection_qc_changed.emit(presentation)
        self._refresh_reconstruction_reference_comparison_presentation()

    @property
    def canonical_inspection_overlay_presentation(
        self,
    ) -> CanonicalObjectInspectionOverlayPresentation | None:
        return self._canonical_inspection_overlay_presentation

    def _refresh_canonical_inspection_overlay(self) -> None:
        """Bind existing E3/E4 evidence to the selected Phase 9.3D object.

        9.3E is presentation-only. Failures here never revoke canonical geometry,
        semantic membership, the E2/E3 sidebar evidence, or source QC authority.
        """

        previous = self._canonical_inspection_overlay_presentation
        self._reconstruction_canvas.clear_canonical_inspection_overlay(update_canvas=False)
        self._canonical_inspection_overlay_presentation = None
        self._canonical_inspection_overlay_error = ""
        try:
            presentation = build_canonical_object_inspection_overlay(
                selection=self._inspection_selection,
                canonical_semantics=self._canonical_semantic_presentation,
                measurement=self._reconstruction_measurement_presentation,
                validity=self._reconstruction_measurement_validity_presentation,
                inspection_qc=self._reconstruction_inspection_qc_presentation,
            )
            if presentation is not None:
                self._reconstruction_canvas.set_canonical_inspection_overlay(
                    scene_id=presentation.scene_id,
                    object_id=presentation.object_id,
                    anchor_viewer=presentation.anchor_viewer,
                    text=presentation.overlay_text,
                )
        except (TypeError, ValueError) as exc:
            self._canonical_inspection_overlay_error = str(exc)
            presentation = None

        self._canonical_inspection_overlay_presentation = presentation
        if previous != presentation:
            self.canonical_inspection_overlay_changed.emit(presentation)
        self._refresh_scene_layer_controls()

    @property
    def reconstruction_reference_comparison_presentation(self) -> ReconstructionReferenceComparisonPresentation | None:
        return self._reconstruction_reference_comparison_presentation

    def set_reconstruction_reference_comparison_reference(
        self,
        reference: object,
        *,
        source: str = "phase8_7c.reference_assembly",
    ) -> None:
        """Attach an explicit E5 reference identity to the active E1 scene.

        The reference is only validated/presented here. E5 does not build a
        reference assembly, parse CAD geometry, run registration, or apply
        tolerances. Existing Phase 8.7C report evidence is paired only when its
        session and reference_id match exactly.
        """

        selection = self._inspection_selection
        if not selection.has_scene:
            raise ValueError("E5 reference comparison requires an active reconstruction scene.")
        session_key = str(selection.session_id or selection.scene_id).strip()
        descriptor = (
            reference
            if isinstance(reference, ReconstructionReferenceDescriptor)
            else reference_descriptor_from_payload(reference, provenance=source)
        )
        self._reconstruction_reference_descriptors[session_key] = descriptor
        self._refresh_reconstruction_reference_comparison_presentation()

    def load_reconstruction_reference_comparison_json(self, path: str | Path) -> None:
        """Attach metadata from an existing Phase 8.7C ReferenceAssembly JSON artifact."""

        selection = self._inspection_selection
        if not selection.has_scene:
            raise ValueError("E5 reference comparison requires an active reconstruction scene.")
        descriptor = load_reference_descriptor_json(path)
        session_key = str(selection.session_id or selection.scene_id).strip()
        self._reconstruction_reference_descriptors[session_key] = descriptor
        self._refresh_reconstruction_reference_comparison_presentation()

    def clear_reconstruction_reference_comparison(self) -> None:
        """Clear all E5 reference bindings without deleting E4 source reports."""

        self._reconstruction_reference_descriptors.clear()
        self._refresh_reconstruction_reference_comparison_presentation()

    def _choose_reconstruction_reference_json(self) -> None:
        selection = self._inspection_selection
        if not selection.has_scene:
            QMessageBox.information(
                self,
                "Reference Comparison",
                "Select or load a reconstruction scene before attaching a reference.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Phase 8.7C Reference Assembly",
            str(Path.home()),
            "Reference JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            self.load_reconstruction_reference_comparison_json(path)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Reference Comparison", str(exc))

    def _clear_active_reconstruction_reference(self) -> None:
        selection = self._inspection_selection
        session_key = str(selection.session_id or selection.scene_id).strip() if selection.has_scene else ""
        if session_key:
            self._reconstruction_reference_descriptors.pop(session_key, None)
        self._refresh_reconstruction_reference_comparison_presentation()

    def _refresh_reconstruction_reference_comparison_presentation(self) -> None:
        selection = self._inspection_selection
        session_key = str(selection.session_id or selection.scene_id).strip() if selection.has_scene else ""
        reference = self._reconstruction_reference_descriptors.get(session_key) if session_key else None
        attached_report = self._reconstruction_inspection_qc_reports.get(session_key) if session_key else None
        presentation = None
        if reference is not None:
            report = attached_report[0] if attached_report is not None else None
            report_source = attached_report[1] if attached_report is not None else "phase8_7c.inspection_report"
            presentation = build_reconstruction_reference_comparison_presentation(
                selection, reference, report, report_source=report_source
            )
        changed = presentation != self._reconstruction_reference_comparison_presentation
        self._reconstruction_reference_comparison_presentation = presentation

        if not hasattr(self, "_reference_comparison_target_value"):
            return
        if presentation is None:
            target, reference_text, structure, registration, deviations, scale, state, authority = (
                empty_reference_comparison_text(selection)
            )
            self._reference_comparison_target_value.setText(target)
            self._reference_comparison_reference_value.setText(reference_text)
            self._reference_comparison_reference_value.setToolTip("")
            self._reference_comparison_structure_value.setText(structure)
            self._reference_comparison_registration_value.setText(registration)
            self._reference_comparison_deviations_value.setText(deviations)
            self._reference_comparison_evidence_value.setText("—")
            self._reference_comparison_scale_value.setText(scale)
            self._reference_comparison_state_value.setText(state)
            self._reference_comparison_authority_value.setText(authority)
        else:
            self._reference_comparison_target_value.setText(presentation.target_text)
            self._reference_comparison_reference_value.setText(presentation.reference.reference_text)
            self._reference_comparison_reference_value.setToolTip(presentation.provenance_tooltip)
            self._reference_comparison_structure_value.setText(presentation.reference.structure_text)
            self._reference_comparison_registration_value.setText(presentation.registration_text)
            self._reference_comparison_deviations_value.setText(presentation.deviations_text)
            self._reference_comparison_evidence_value.setText(presentation.evidence_text)
            self._reference_comparison_scale_value.setText(presentation.scale_text)
            self._reference_comparison_state_value.setText(presentation.state_text)
            self._reference_comparison_authority_value.setText(presentation.authority_text)

        self._reference_comparison_clear_button.setEnabled(reference is not None)
        self._reference_comparison_load_button.setEnabled(selection.has_scene)
        if changed:
            self.reconstruction_reference_comparison_changed.emit(presentation)

    @property
    def digital_twin_workspace_presentation(self) -> DigitalTwinWorkspacePresentation | None:
        return self._digital_twin_workspace_presentation

    @property
    def digital_twin_scene_composition(self) -> DigitalTwinSceneComposition | None:
        return self._digital_twin_scene_composition

    def _refresh_digital_twin_scene_composition(self) -> None:
        canvas = getattr(self, "_reconstruction_canvas", None)
        previous = self._digital_twin_scene_composition
        try:
            composition = build_digital_twin_scene_composition(
                self._canonical_scene_presentation,
                canonical_surface_loaded=bool(canvas and canvas.canonical_mesh_loaded),
                canonical_surface_visible=bool(canvas and canvas.canonical_mesh_visible),
                semantic_presentation=self._canonical_semantic_presentation,
                semantic_visible=bool(canvas and canvas.semantic_objects_visible),
                camera_presentation=self._canonical_camera_presentation,
                camera_visible=bool(
                    canvas
                    and (
                        canvas.canonical_camera_frustums_visible
                        or canvas.canonical_camera_trajectory_visible
                    )
                ),
                appearance_presentation=self._appearance_presentation,
                active_view_representation=self.active_view_representation,
                gaussian_visible=bool(self.interactive_gaussian_visible or self.gaussian_preview_visible),
                nerf_visible=bool(self.interactive_nerf_visible or self.nerf_preview_visible),
            )
        except (TypeError, ValueError) as exc:
            self._digital_twin_scene_composition = None
            if hasattr(self, "_digital_twin_composition_value"):
                self._digital_twin_composition_value.setText(
                    "Composition unavailable • canonical authority retained"
                )
                self._digital_twin_composition_value.setToolTip(str(exc))
            if previous is not None:
                self.digital_twin_scene_composition_changed.emit(None)
            self._refresh_digital_twin_navigation_sync()
            return

        self._digital_twin_scene_composition = composition
        if hasattr(self, "_digital_twin_composition_value"):
            self._digital_twin_composition_value.setText(composition.composition_text)
            self._digital_twin_composition_value.setToolTip(
                composition.visibility_text
                + "\n"
                + composition.authority_text
            )
        if composition != previous:
            self.digital_twin_scene_composition_changed.emit(composition)
        self._refresh_digital_twin_navigation_sync()

    @property
    def digital_twin_navigation_sync(self) -> DigitalTwinNavigationSync | None:
        return self._digital_twin_navigation_sync

    def _refresh_digital_twin_navigation_sync(self) -> None:
        previous = self._digital_twin_navigation_sync
        composition = self._digital_twin_scene_composition
        canvas = getattr(self, "_reconstruction_canvas", None)
        if composition is None:
            self._digital_twin_navigation_sync = None
            if hasattr(self, "_digital_twin_navigation_value"):
                self._digital_twin_navigation_value.setText(
                    "Navigation unavailable • canonical controls retained"
                )
                self._digital_twin_navigation_value.setToolTip(
                    "Unified scene composition is unavailable. No appearance representation is granted "
                    "a separate navigation or spatial authority."
                )
            if previous is not None:
                self.digital_twin_navigation_changed.emit(None)
            self._refresh_digital_twin_interaction_presentation()
            return

        try:
            camera_state_getter = (
                getattr(canvas, "interactive_render_camera_state", None)
                if canvas is not None
                else None
            )
            camera_state = (
                camera_state_getter()
                if bool(composition.scene_id) and callable(camera_state_getter)
                else None
            )
            sync = build_digital_twin_navigation_sync(
                composition,
                camera_state=camera_state,
                gaussian_interactive=self.interactive_gaussian_visible,
                nerf_interactive=self.interactive_nerf_visible,
                gaussian_preview_visible=self.gaussian_preview_visible,
                nerf_preview_visible=self.nerf_preview_visible,
                view_preserved_on_same_scene_reload=bool(self._last_scene_view_preserved),
            )
        except (TypeError, ValueError) as exc:
            self._digital_twin_navigation_sync = None
            if hasattr(self, "_digital_twin_navigation_value"):
                self._digital_twin_navigation_value.setText(
                    "Navigation synchronization unavailable • canonical controls retained"
                )
                self._digital_twin_navigation_value.setToolTip(str(exc))
            if previous is not None:
                self.digital_twin_navigation_changed.emit(None)
            self._refresh_digital_twin_interaction_presentation()
            return

        self._digital_twin_navigation_sync = sync
        if hasattr(self, "_digital_twin_navigation_value"):
            self._digital_twin_navigation_value.setText(sync.navigation_text)
            self._digital_twin_navigation_value.setToolTip(
                sync.camera_text + "\n" + sync.scene_switch_text + "\n" + sync.authority_text
            )
        if sync != previous:
            self.digital_twin_navigation_changed.emit(sync)
        self._refresh_digital_twin_interaction_presentation()

    @property
    def digital_twin_interaction_presentation(self) -> DigitalTwinInteractionPresentation | None:
        return self._digital_twin_interaction_presentation

    def _apply_digital_twin_interaction_appearance_policy(self) -> None:
        """Apply transient NeRF legibility assistance without mutating user settings."""

        presentation = self._digital_twin_interaction_presentation
        if presentation is None or not self._interactive_nerf_visible:
            return
        cap = presentation.effective_nerf_opacity_cap_percent
        effective = self._nerf_display_settings.opacity_percent
        if cap is not None:
            effective = min(int(effective), int(cap))
        self._appearance_preview.set_image_opacity(effective)
        self._appearance_preview.set_mouse_passthrough(True)
        if presentation.measurement_active:
            self._appearance_preview.set_footer(
                "NeRF frozen/dimmed for canonical ruler • clicks and measurement remain canonical"
            )
        elif presentation.semantic_selection_active:
            self._appearance_preview.set_footer(
                "NeRF dimmed for semantic inspection • canonical membership/inspection overlays remain authoritative"
            )
        else:
            self._appearance_preview.set_footer(
                "External Nerfacto appearance • canonical geometry remains picking/measurement authority"
            )
        if self._appearance_preview.has_image and not bool(
            getattr(self._reconstruction_canvas, "is_interacting", False)
        ):
            self._appearance_preview.show()

    def _refresh_digital_twin_interaction_presentation(self) -> None:
        previous = self._digital_twin_interaction_presentation
        composition = self._digital_twin_scene_composition
        canvas = getattr(self, "_reconstruction_canvas", None)
        if composition is None:
            self._digital_twin_interaction_presentation = None
            if hasattr(self, "_digital_twin_interaction_value"):
                self._digital_twin_interaction_value.setText(
                    "Interaction unavailable • canonical authority retained"
                )
                self._digital_twin_interaction_value.setToolTip(
                    "Unified scene composition is unavailable. No appearance layer is promoted to picking, measurement, scale, or QC authority."
                )
            if previous is not None:
                self.digital_twin_interaction_changed.emit(None)
            return

        try:
            semantic_available = bool(
                (canvas and getattr(canvas, "semantic_objects_available", False))
                or self._canonical_semantic_presentation is not None
                or self._offline_semantics is not None
            )
            semantic_selected = bool(
                self._semantic_object_selection_index is not None
                or (canvas and getattr(canvas, "selected_semantic_object_id", None))
            )
            interaction = build_digital_twin_interaction_presentation(
                composition,
                canonical_surface_loaded=bool(
                    canvas and getattr(canvas, "canonical_mesh_loaded", False)
                ),
                semantic_context_available=semantic_available,
                semantic_selection_active=semantic_selected,
                measurement_active=bool(
                    canvas is not None
                    and getattr(canvas, "interaction_mode", "") == "measure"
                ),
                gaussian_interactive=self.interactive_gaussian_visible,
                nerf_interactive=self.interactive_nerf_visible,
                gaussian_preview_visible=self.gaussian_preview_visible,
                nerf_preview_visible=self.nerf_preview_visible,
            )
        except (TypeError, ValueError) as exc:
            self._digital_twin_interaction_presentation = None
            if hasattr(self, "_digital_twin_interaction_value"):
                self._digital_twin_interaction_value.setText(
                    "Interaction policy unavailable • canonical authority retained"
                )
                self._digital_twin_interaction_value.setToolTip(str(exc))
            if previous is not None:
                self.digital_twin_interaction_changed.emit(None)
            return

        self._digital_twin_interaction_presentation = interaction
        if hasattr(self, "_digital_twin_interaction_value"):
            self._digital_twin_interaction_value.setText(interaction.interaction_text)
            self._digital_twin_interaction_value.setToolTip(
                interaction.semantic_text
                + "\n"
                + interaction.measurement_text
                + "\n"
                + interaction.appearance_text
                + "\n"
                + interaction.authority_text
            )
        self._apply_digital_twin_interaction_appearance_policy()
        if interaction != previous:
            self.digital_twin_interaction_changed.emit(interaction)

    def _refresh_digital_twin_workspace_presentation(self) -> None:
        canvas = getattr(self, "_reconstruction_canvas", None)
        try:
            presentation = build_digital_twin_workspace_presentation(
                self._canonical_scene_presentation,
                canonical_surface_loaded=bool(canvas and canvas.canonical_mesh_loaded),
                semantic_context_available=bool(
                    (canvas and canvas.semantic_objects_available)
                    or self._canonical_semantic_presentation is not None
                    or self._offline_semantics is not None
                ),
                appearance_presentation=self._appearance_presentation,
                active_view_representation=self.active_view_representation,
                gaussian_runtime_active=self.interactive_gaussian_visible,
                nerf_runtime_active=self.interactive_nerf_visible,
            )
        except (TypeError, ValueError) as exc:
            previous = self._digital_twin_workspace_presentation
            self._digital_twin_workspace_presentation = None
            if hasattr(self, "_digital_twin_workspace_value"):
                self._digital_twin_workspace_value.setText("Digital Twin composition unavailable")
                self._digital_twin_workspace_value.setToolTip(str(exc))
                self._digital_twin_active_view_value.setText("Canonical geometry")
                self._digital_twin_composition_value.setText("Composition rejected")
                self._digital_twin_navigation_value.setText("Canonical navigation retained")
                if hasattr(self, "_digital_twin_interaction_value"):
                    self._digital_twin_interaction_value.setText("Canonical interaction retained")
                self._digital_twin_scale_value.setText("Not qualified")
                self._digital_twin_authority_value.setText("Canonical authority preserved outside summary")
                self._digital_twin_resource_value.setText("Neural appearance not promoted")
                self._digital_twin_safety_value.setText(
                    "Fail closed • QC/reference separate • metrology not certified"
                )
            if previous is not None:
                self.digital_twin_workspace_changed.emit(None)
            self._refresh_digital_twin_resource_coordination()
            self._refresh_digital_twin_scene_composition()
            return

        changed = presentation != self._digital_twin_workspace_presentation
        self._digital_twin_workspace_presentation = presentation
        if hasattr(self, "_digital_twin_workspace_value"):
            self._digital_twin_workspace_value.setText(presentation.workspace_text)
            self._digital_twin_workspace_value.setToolTip(presentation.scene_id)
            self._digital_twin_active_view_value.setText(presentation.active_view_text)
            self._digital_twin_composition_value.setText(presentation.composition_text)
            self._digital_twin_scale_value.setText(presentation.scale_text)
            self._digital_twin_scale_value.setToolTip(
                "Scale wording is inherited from the existing canonical-scene authority presentation; "
                "camera calibration alone does not establish metric scale."
            )
            self._digital_twin_authority_value.setText(presentation.authority_text)
            self._digital_twin_resource_value.setText(presentation.resource_text)
            self._digital_twin_safety_value.setText(presentation.safety_text)
        if changed:
            self.digital_twin_workspace_changed.emit(presentation)
        self._refresh_digital_twin_resource_coordination()
        self._refresh_digital_twin_scene_composition()

    @property
    def digital_twin_processing_snapshot(self) -> DigitalTwinProcessingSnapshot:
        return self._digital_twin_processing_snapshot

    @property
    def digital_twin_resource_coordination(self) -> DigitalTwinResourceCoordination | None:
        return self._digital_twin_resource_coordination

    @property
    def digital_twin_resource_active(self) -> bool:
        """Whether Reconstruction owns dynamic resources beyond its controller.

        MainWindow already observes ``ReconstructionController.is_active``.  G adds
        the E1 build supervisor, isolated interactive NeRF process and bounded
        interactive Gaussian residency to that application-wide ownership boundary.
        Static previews intentionally do not count.
        """

        coordination = self._digital_twin_resource_coordination
        if coordination is None:
            coordination = self._refresh_digital_twin_resource_coordination()
        return bool(coordination and coordination.page_resource_active)

    def _refresh_digital_twin_resource_coordination(self) -> DigitalTwinResourceCoordination:
        controller = self._controller
        snapshot = self._digital_twin_processing_snapshot
        coordination = build_digital_twin_resource_coordination(
            processing_active=bool(snapshot.is_active or self._digital_twin_processing_controller.is_running),
            reconstruction_active=bool(controller is not None and controller.is_active),
            interactive_nerf_active=bool(
                self._interactive_nerf_visible or self._nerf_worker_client.is_running
            ),
            interactive_gaussian_resident=bool(
                self._interactive_gaussian_visible
                or self._interactive_gaussian_scene is not None
                or self._gaussian_display_buffers is not None
            ),
            static_preview_active=bool(self._gaussian_preview_visible or self._nerf_preview_visible),
            build_label=(snapshot.job_text if snapshot.spec is not None else ""),
        )
        previous_active = bool(self._digital_twin_resource_activity_active)
        self._digital_twin_resource_coordination = coordination
        current_active = bool(coordination.page_resource_active)
        self._digital_twin_resource_activity_active = current_active

        if hasattr(self, "_digital_twin_processing_resource_value"):
            self._digital_twin_processing_resource_value.setText(coordination.owner_text)
            self._digital_twin_processing_resource_value.setToolTip(coordination.details_text)
            self._digital_twin_processing_gpu_value.setText(coordination.startup_resources_text)
            self._digital_twin_processing_gpu_value.setToolTip(
                "Startup hardware snapshot only. Live per-process CUDA telemetry remains owned by the qualified external runtime diagnostics."
            )
        if hasattr(self, "_digital_twin_resource_value"):
            self._digital_twin_resource_value.setText(coordination.owner_text)
            self._digital_twin_resource_value.setToolTip(coordination.details_text)

        if current_active != previous_active:
            self.digital_twin_resource_activity_changed.emit(current_active)
        return coordination

    def _release_appearance_resources_for_heavy_handoff(self, purpose: str) -> None:
        """Release dynamic appearance residency before a heavyweight handoff.

        G centralizes the local E2/E3 guards.  The canonical scene remains loaded
        and static preview artifacts remain available to re-open later.
        """

        if self._interactive_gaussian_visible or self._interactive_gaussian_scene is not None:
            self._hide_interactive_gaussian()
        if self._gaussian_preview_visible:
            self._hide_gaussian_preview()
        if self._interactive_nerf_visible or self._nerf_worker_client.is_running:
            self._hide_interactive_nerf(blocking=True)
        if self._nerf_preview_visible:
            self._hide_nerf_preview()
        coordination = self._refresh_digital_twin_resource_coordination()
        if self._nerf_worker_client.is_running or coordination.interactive_nerf_active:
            label = str(purpose).strip() or "heavyweight operation"
            raise RuntimeError(
                f"Interactive NeRF resources did not release before starting {label}."
            )
        if coordination.interactive_gaussian_resident:
            label = str(purpose).strip() or "heavyweight operation"
            raise RuntimeError(
                f"Interactive Gaussian resources did not release before starting {label}."
            )

    def _interactive_resource_admission_failure(self, label: str) -> str:
        coordination = self._refresh_digital_twin_resource_coordination()
        return coordination.interactive_admission_failure(label)

    def start_digital_twin_processing_job(
        self,
        spec: DigitalTwinProcessingJobSpec,
        command: DigitalTwinProcessingCommand,
    ) -> str:
        """Start one future E2/E3 build job through the generic E1 supervisor.

        E1 intentionally exposes this adapter seam without wiring any current
        product button to it.  Heavy DA3/Nerfstudio imports remain outside the
        ``depth`` process and a second heavyweight build cannot overlap.
        """

        if not isinstance(spec, DigitalTwinProcessingJobSpec):
            raise TypeError("spec must be DigitalTwinProcessingJobSpec.")
        if not isinstance(command, DigitalTwinProcessingCommand):
            raise TypeError("command must be DigitalTwinProcessingCommand.")
        active_scene_id = str(getattr(self._persistent_scene, "scene_id", "")).strip()
        if spec.scene_id and active_scene_id and spec.scene_id != active_scene_id:
            raise ValueError(
                "Digital Twin processing job scene_id does not match the active canonical scene."
            )
        coordination = self._refresh_digital_twin_resource_coordination()
        if coordination.reconstruction_active:
            raise RuntimeError("Stop active Reconstruction before starting a Digital Twin build job.")
        self._release_appearance_resources_for_heavy_handoff(spec.title)
        coordination = self._refresh_digital_twin_resource_coordination()
        if coordination.conflict or coordination.interactive_nerf_active:
            raise RuntimeError(coordination.conflict_reason or "Digital Twin resources are not ready for build handoff.")
        return self._digital_twin_processing_controller.start(spec, command)

    def cancel_digital_twin_processing_job(self) -> bool:
        return self._digital_twin_processing_controller.cancel()

    def _on_digital_twin_processing_snapshot_changed(
        self, snapshot: DigitalTwinProcessingSnapshot
    ) -> None:
        if not isinstance(snapshot, DigitalTwinProcessingSnapshot):
            return
        changed = snapshot != self._digital_twin_processing_snapshot
        self._digital_twin_processing_snapshot = snapshot
        if snapshot.lifecycle.is_terminal and snapshot.spec is not None:
            try:
                self._digital_twin_processing_history = (
                    DigitalTwinProcessingHistoryRecord.from_snapshot(snapshot)
                )
            except (TypeError, ValueError):
                # History is presentation-only; never disturb the live E1 job
                # lifecycle if a terminal summary itself cannot be persisted.
                pass
        self._refresh_digital_twin_processing_presentation()
        self._refresh_digital_twin_processing_history_presentation()
        self._refresh_digital_twin_resource_coordination()
        self._refresh_digital_twin_workspace_persistence_controls()
        self._refresh_da3_full_scene_controls()
        self._refresh_nerf_build_controls()
        self._refresh_da3_canonical_controls()
        self._refresh_source_controls()
        self._refresh_runtime_controls()
        self._focus_digital_twin_processing_for_new_job(snapshot)
        if changed:
            self.digital_twin_processing_changed.emit(snapshot)

    def _focus_digital_twin_processing_for_new_job(
        self, snapshot: DigitalTwinProcessingSnapshot
    ) -> None:
        """Reveal the existing processing disclosure once for each new build job.

        Phase 9.6E4 makes heavyweight lifecycle state visible without adding a
        second job surface. User disclosure choices remain untouched while idle
        and after the one-time focus transition.
        """

        spec = snapshot.spec
        if spec is None or snapshot.lifecycle is DigitalTwinProcessingLifecycle.IDLE:
            return
        job_id = str(spec.job_id).strip()
        if not job_id or job_id == self._digital_twin_processing_focus_job_id:
            return
        if not hasattr(self, "_digital_twin_processing_group"):
            return
        self._digital_twin_processing_focus_job_id = job_id
        self._digital_twin_processing_group.set_expanded(True)
        if hasattr(self, "_settings_scroll"):
            QTimer.singleShot(
                0,
                lambda: self._settings_scroll.ensureWidgetVisible(
                    self._digital_twin_processing_group, 8, 24
                ),
            )

    def _on_digital_twin_processing_error(self, message: str) -> None:
        text = str(message).strip()
        if text and hasattr(self, "_digital_twin_processing_state_value"):
            self._digital_twin_processing_state_value.setToolTip(text)

    def _start_da3_full_scene_build(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown:
            return
        if self._digital_twin_processing_snapshot.is_active:
            self.set_error("A Digital Twin build job is already active.")
            self._refresh_runtime_controls()
            return
        selection = controller.source_selection
        if (
            selection.workflow is not ReconstructionWorkflow.RECORDED_VIDEO
            or not selection.ready
            or selection.path is None
        ):
            self.set_error("Select a ready Recorded video before building Full-Scene DA3 geometry.")
            self._refresh_da3_full_scene_controls()
            return
        if controller.is_active or self._digital_twin_processing_snapshot.is_active:
            self.set_error("Stop the active reconstruction/build job before starting Full-Scene DA3.")
            self._refresh_da3_full_scene_controls()
            return
        try:
            # Phase 9.6G centralizes the prior E2 local handoff in
            # start_digital_twin_processing_job().
            spec, command = build_full_scene_da3_job(
                video_path=selection.path,
                output_root=default_reconstruction_output_root(),
            )
            self.start_digital_twin_processing_job(spec, command)
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        self._refresh_da3_full_scene_controls()
        self._refresh_da3_canonical_controls()
        self._refresh_source_controls()
        self._refresh_runtime_controls()

    def _start_nerf_build(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown:
            return
        if self._digital_twin_processing_snapshot.is_active:
            self.set_error("A Digital Twin build job is already active.")
            return
        scene = controller.persistent_scene
        manifest = controller.persistent_scene_manifest
        if scene is None or manifest is None:
            self.set_error("Load or build a persistent canonical scene before training NeRF appearance.")
            self._refresh_nerf_build_controls()
            return
        if controller.is_active:
            self.set_error("Stop the active reconstruction job before training NeRF appearance.")
            self._refresh_nerf_build_controls()
            return
        preset_key = str(self._nerf_build_profile_combo.currentData() or NERF_DEFAULT_BUILD_PRESET)
        try:
            # Phase 9.6G centralizes the prior E3 local handoff in
            # start_digital_twin_processing_job().
            spec, command = build_nerf_training_job(
                scene=scene,
                scene_manifest=manifest,
                output_root=default_reconstruction_output_root(),
                preset_key=preset_key,
            )
            self.start_digital_twin_processing_job(spec, command)
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        self._refresh_nerf_build_controls()
        self._refresh_source_controls()
        self._refresh_runtime_controls()

    def _on_digital_twin_processing_result_ready(
        self, snapshot: DigitalTwinProcessingSnapshot
    ) -> None:
        if not isinstance(snapshot, DigitalTwinProcessingSnapshot) or snapshot.spec is None:
            return
        if not snapshot.result_path:
            return
        if snapshot.spec.kind.value == "nerf_appearance" and snapshot.spec.expected_protocol == NERF_BUILD_PROTOCOL:
            try:
                summary_path = Path(snapshot.result_path).expanduser().resolve()
                summary_path.relative_to(snapshot.spec.output_dir)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if str(summary.get("job_id", "")).strip() != snapshot.spec.job_id:
                    raise ValueError("NeRF result job identity does not match the completed build.")
                if str(summary.get("phase", "")).strip() != "9.6E3" or not bool(summary.get("pass", False)):
                    raise ValueError("Completed NeRF result is not a valid Phase 9.6E3 result.")
                if any(bool(summary.get(key, False)) for key in (
                    "canonical_geometry_authoritative", "metric_scale_authoritative",
                    "measurement_authoritative", "qc_authoritative", "certified_metrology",
                )):
                    raise ValueError("NeRF build result contains an unsafe authority claim.")
                manifest = Path(str(summary.get("persistent_scene_manifest", ""))).expanduser().resolve()
                manifest.relative_to(snapshot.spec.output_dir)
                if self._controller is None:
                    raise RuntimeError("Reconstruction controller became unavailable before NeRF result admission.")
                active_scene_id = str(getattr(self._controller.persistent_scene, "scene_id", "")).strip()
                if active_scene_id != snapshot.spec.scene_id:
                    raise ValueError("Active canonical scene changed while NeRF was training; stale appearance result was not admitted.")
                self._controller.load_persistent_scene(manifest, verify_file_references=True)
                self._nerf_build_value.setText(
                    f"Ready • {int(summary.get('input_frame_count', 0))} views • {int(summary.get('training_steps', 0))} steps"
                )
                self._nerf_build_value.setToolTip(
                    "Trained Nerfacto appearance persisted as an immutable scene revision. Canonical geometry remains authoritative; quality is not certified."
                )
            except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.set_error(f"NeRF build result could not be admitted: {exc}")
            finally:
                self._refresh_nerf_build_controls()
                self._refresh_source_controls()
                self._refresh_runtime_controls()
            return
        if snapshot.spec.kind.value != "dense_geometry" or snapshot.spec.expected_protocol != DA3_FULL_SCENE_PROTOCOL:
            return
        try:
            summary_path = Path(snapshot.result_path).expanduser().resolve()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if str(summary.get("job_id", "")).strip() != snapshot.spec.job_id:
                raise ValueError("Full-scene DA3 result job identity does not match the completed build.")
            if str(summary.get("phase", "")).strip() != "9.6E2" or not bool(summary.get("pass", False)):
                raise ValueError("Completed dense-geometry result is not a valid Phase 9.6E2 result.")
            if any(bool(summary.get(key, False)) for key in (
                "canonical_geometry_authoritative",
                "metric_scale_authoritative",
                "measurement_authoritative",
                "qc_authoritative",
                "certified_metrology",
            )):
                raise ValueError("Full-scene DA3 result contains an unsafe authority claim.")
            manifest = Path(str(summary.get("persistent_scene_manifest", ""))).expanduser().resolve()
            manifest.relative_to(snapshot.spec.output_dir)
            if self._controller is None:
                raise RuntimeError("Reconstruction controller became unavailable before DA3 result admission.")
            self._controller.load_persistent_scene(manifest, verify_file_references=True)
            self._selected_dense_provider_id = "depth-anything-3-external"
            self._refresh_dense_provider_controls()
            self._da3_full_scene_value.setText(
                f"Ready • {int(summary.get('selected_keyframes', 0))} keyframes • arbitrary scale"
            )
            self._da3_full_scene_value.setToolTip(
                "Full-scene temporal coverage uses sequential overlapping DA3 windows. Loop closure and metric scale are not claimed."
            )
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.set_error(f"Full-scene DA3 result could not be admitted: {exc}")
        finally:
            self._refresh_da3_full_scene_controls()
            self._refresh_da3_canonical_controls()
            self._refresh_source_controls()
            self._refresh_runtime_controls()

    def _refresh_da3_full_scene_controls(self) -> None:
        if not hasattr(self, "_da3_full_scene_build_button"):
            return
        controller = self._controller
        snapshot = self._digital_twin_processing_snapshot
        if controller is None:
            self._da3_full_scene_build_button.setEnabled(False)
            self._da3_full_scene_value.setText("Controller unavailable")
            return
        selection = controller.source_selection
        ready_video = bool(
            selection.ready
            and selection.path is not None
            and selection.workflow is ReconstructionWorkflow.RECORDED_VIDEO
        )
        can_start = bool(
            not controller.is_shutdown
            and ready_video
            and not controller.is_active
            and not snapshot.is_active
        )
        self._da3_full_scene_build_button.setEnabled(can_start)
        if snapshot.spec is not None and snapshot.spec.kind.value == "dense_geometry":
            if snapshot.is_active:
                self._da3_full_scene_value.setText(
                    f"{snapshot.stage_text} • {snapshot.progress_text}"
                )
            elif snapshot.lifecycle is DigitalTwinProcessingLifecycle.FAILED:
                self._da3_full_scene_value.setText("Failed • previous canonical scene preserved")
            elif snapshot.lifecycle is DigitalTwinProcessingLifecycle.CANCELLED:
                self._da3_full_scene_value.setText("Cancelled • previous canonical scene preserved")
            elif snapshot.lifecycle is DigitalTwinProcessingLifecycle.COMPLETED and not self._da3_full_scene_value.text().startswith("Ready •"):
                self._da3_full_scene_value.setText("Completed • admitting persistent full-scene geometry")
        elif ready_video:
            self._da3_full_scene_value.setText("Ready • Full Scene • click Build Full Scene below")
        else:
            self._da3_full_scene_value.setText("Select a ready Recorded video for Full Scene")

    def _refresh_nerf_build_controls(self) -> None:
        if not hasattr(self, "_nerf_build_button"):
            return
        controller = self._controller
        snapshot = self._digital_twin_processing_snapshot
        scene = controller.persistent_scene if controller is not None else None
        has_scene = bool(scene is not None and getattr(scene, "canonical_geometry_authoritative", False))
        has_nerf = bool(has_scene and getattr(scene, "nerf_artifact", None) is not None)
        can_start = bool(
            controller is not None
            and not controller.is_shutdown
            and has_scene
            and controller.persistent_scene_manifest is not None
            and not controller.is_active
            and not snapshot.is_active
        )
        self._nerf_build_button.setEnabled(can_start)
        self._nerf_build_profile_combo.setEnabled(can_start)
        self._nerf_build_button.setText("Rebuild NeRF" if has_nerf else "Build & Train NeRF")
        if snapshot.spec is not None and snapshot.spec.kind.value == "nerf_appearance":
            if snapshot.is_active:
                self._nerf_build_value.setText(f"{snapshot.stage_text} • {snapshot.progress_text}")
            elif snapshot.lifecycle is DigitalTwinProcessingLifecycle.FAILED:
                self._nerf_build_value.setText("Failed • existing canonical/NeRF scene preserved")
            elif snapshot.lifecycle is DigitalTwinProcessingLifecycle.CANCELLED:
                self._nerf_build_value.setText("Cancelled • existing scene preserved")
            elif snapshot.lifecycle is DigitalTwinProcessingLifecycle.COMPLETED and not self._nerf_build_value.text().startswith("Ready •"):
                self._nerf_build_value.setText("Completed • admitting persistent NeRF appearance")
        elif has_nerf:
            self._nerf_build_value.setText("Ready • persisted NeRF available • Rebuild creates a new immutable revision")
        elif has_scene:
            self._nerf_build_value.setText("Ready • canonical views available for NeRF training")
        else:
            self._nerf_build_value.setText("Load/build a persistent canonical scene first")

    def _refresh_digital_twin_processing_presentation(self) -> None:
        if not hasattr(self, "_digital_twin_processing_job_value"):
            return
        snapshot = self._digital_twin_processing_snapshot
        self._digital_twin_processing_job_value.setText(snapshot.job_text)
        self._digital_twin_processing_state_value.setText(snapshot.lifecycle_text)
        self._digital_twin_processing_stage_value.setText(snapshot.stage_text)
        self._digital_twin_processing_stage_value.setToolTip(snapshot.message or snapshot.error)
        self._digital_twin_processing_backend_value.setText(snapshot.backend_text)
        self._digital_twin_processing_output_value.setText(snapshot.output_text)
        if snapshot.spec is not None:
            self._digital_twin_processing_output_value.setToolTip(
                snapshot.result_path or str(snapshot.spec.output_dir)
            )
        else:
            self._digital_twin_processing_output_value.setToolTip("")
        self._digital_twin_processing_authority_value.setText(snapshot.authority_text)
        self._digital_twin_processing_cancel_button.setEnabled(snapshot.can_cancel)
        self._refresh_da3_full_scene_controls()
        self._refresh_nerf_build_controls()

        progress = self._digital_twin_processing_progress
        if snapshot.lifecycle is DigitalTwinProcessingLifecycle.IDLE:
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setFormat("Idle")
        elif snapshot.progress_fraction is None and snapshot.lifecycle.is_active:
            progress.setRange(0, 0)
            progress.setFormat("Active")
        else:
            progress.setRange(0, 100)
            value = (
                round(snapshot.progress_fraction * 100.0)
                if snapshot.progress_fraction is not None
                else (100 if snapshot.lifecycle is DigitalTwinProcessingLifecycle.COMPLETED else 0)
            )
            progress.setValue(max(0, min(100, int(value))))
            progress.setFormat(snapshot.progress_text)
        progress.setProperty(
            "statusKind",
            "error"
            if snapshot.lifecycle is DigitalTwinProcessingLifecycle.FAILED
            else "warning"
            if snapshot.lifecycle in {
                DigitalTwinProcessingLifecycle.CANCELLING,
                DigitalTwinProcessingLifecycle.CANCELLED,
            }
            else "ready"
            if snapshot.lifecycle is DigitalTwinProcessingLifecycle.COMPLETED
            else "running"
            if snapshot.lifecycle.is_active
            else "neutral",
        )
        self._refresh_style(progress)
        elapsed = float(snapshot.elapsed_seconds)
        if snapshot.lifecycle is DigitalTwinProcessingLifecycle.IDLE:
            self._digital_twin_processing_elapsed_value.setText("—")
        else:
            minutes = int(elapsed) // 60
            seconds = int(elapsed) % 60
            self._digital_twin_processing_elapsed_value.setText(f"{minutes:02d}:{seconds:02d}")

    def _refresh_digital_twin_processing_history_presentation(self) -> None:
        if not hasattr(self, "_digital_twin_processing_history_value"):
            return
        history = self._digital_twin_processing_history
        if history is None:
            self._digital_twin_processing_history_value.setText("No persisted build history")
            self._digital_twin_processing_history_value.setToolTip(
                "Phase 9.6F persists terminal build history only; active jobs are never resumable."
            )
            return
        self._digital_twin_processing_history_value.setText(
            f"{history.title} • {history.lifecycle.value.title()}"
        )
        details = [
            f"Job: {history.job_id}",
            f"Backend: {history.backend_label}",
            f"Runtime: {history.runtime_label}",
            f"Stage: {history.stage or '—'}",
            f"Profile: {history.profile_label or '—'}",
            "Historical record only • not resumable • no authority granted",
        ]
        if history.result_path:
            details.append(f"Result: {history.result_path}")
        self._digital_twin_processing_history_value.setToolTip("\n".join(details))

    @property
    def canonical_scene_presentation(self) -> CanonicalScenePresentation | None:
        return self._canonical_scene_presentation

    def _refresh_canonical_scene_presentation(self) -> None:
        try:
            presentation = build_canonical_scene_presentation(
                self._persistent_scene,
                self._persistent_authority_snapshot,
            )
        except (TypeError, ValueError) as exc:
            self._canonical_scene_presentation = None
            if hasattr(self, "_canonical_scene_state_value"):
                self._canonical_scene_state_value.setText("Canonical scene unavailable")
                self._canonical_scene_state_value.setToolTip(str(exc))
                self._canonical_scene_geometry_value.setText("—")
                self._canonical_scene_provider_value.setText("—")
                self._canonical_scene_resource_value.setText("—")
                if hasattr(self, "_canonical_scene_world_value"):
                    self._canonical_scene_world_value.setText("Canonical frame unavailable")
                    self._canonical_scene_cameras_value.setText("Camera context unavailable")
                self._canonical_scene_scale_value.setText("Not qualified")
                self._canonical_scene_authority_value.setText(
                    "Canonical authority unavailable • metrology not certified"
                )
                self._canonical_scene_status_value.setText("Persistent scene presentation rejected")
                self._canonical_scene_status_value.setToolTip(str(exc))
            if hasattr(self, "_viewport_safety"):
                self._viewport_safety.setText("VIEWER Z-UP • ARBITRARY GAUGE")
                self._viewport_safety.setToolTip(str(exc))
            self._refresh_reconstruction_interactive_measurement_availability()
            self._refresh_estimated_metric_scale_presentation()
            self._refresh_active_ruler_metric_record()
            self._refresh_digital_twin_workspace_presentation()
            self.canonical_scene_presentation_changed.emit(None)
            return

        changed = presentation != self._canonical_scene_presentation
        self._canonical_scene_presentation = presentation
        if hasattr(self, "_canonical_scene_state_value"):
            self._canonical_scene_state_value.setText(presentation.scene_text)
            self._canonical_scene_state_value.setToolTip(presentation.scene_id)
            self._canonical_scene_geometry_value.setText(presentation.geometry_text)
            self._canonical_scene_geometry_value.setToolTip(presentation.artifact_id)
            self._canonical_scene_provider_value.setText(presentation.provider_id or "—")
            self._canonical_scene_resource_value.setText(presentation.resource_text)
            self._canonical_scene_resource_value.setToolTip(presentation.resource_path)
            self._canonical_scene_scale_value.setText(presentation.scale_text)
            scale_tip = []
            if presentation.scale_unit_to_metre is not None:
                scale_tip.append(
                    f"Scale factor: {presentation.scale_unit_to_metre:.9g} m per reconstruction unit"
                )
            if presentation.metric_evidence_id:
                scale_tip.append(f"Metric evidence: {presentation.metric_evidence_id}")
            if presentation.metric_assessment_state:
                scale_tip.append(f"Assessment: {presentation.metric_assessment_state}")
            if presentation.calibration_available:
                scale_tip.append("Camera calibration evidence is present but remains separate from scale authority.")
            elif presentation.available:
                scale_tip.append("No persisted camera-calibration profile is bound to this scene.")
            self._canonical_scene_scale_value.setToolTip("\n".join(scale_tip))
            self._canonical_scene_authority_value.setText(presentation.authority_text)
            self._canonical_scene_status_value.setText(presentation.state_text)
            self._canonical_scene_status_value.setToolTip(
                "Phase 9.3A is presentation-only. Canonical surface parsing/rendering is deferred to Phase 9.3B."
            )
        if hasattr(self, "_viewport_safety"):
            self._viewport_safety.setText(presentation.viewer_scale_banner_text)
            self._viewport_safety.setToolTip(
                "Viewport axis conversion is presentation-only. Metric wording is shown only from "
                "the scene-bound Phase 9.1H metric qualification assessment; metrology is not certified."
            )
        self._refresh_reconstruction_interactive_measurement_availability()
        self._refresh_estimated_metric_scale_presentation()
        self._refresh_active_ruler_metric_record()
        self._refresh_digital_twin_workspace_presentation()
        if changed:
            self.canonical_scene_presentation_changed.emit(presentation)

    @property
    def canonical_viewer_mesh(self) -> CanonicalViewerMesh | None:
        return self._canonical_viewer_mesh

    def _apply_canonical_viewer_status(self) -> None:
        mesh = self._canonical_viewer_mesh
        if not hasattr(self, "_canonical_scene_status_value"):
            return
        if mesh is not None and self._reconstruction_canvas.canonical_mesh_loaded:
            self._canonical_scene_status_value.setText(
                f"Interactive canonical surface • {mesh.vertex_count:,} vertices • {mesh.face_count:,} faces"
            )
            self._canonical_scene_status_value.setToolTip(
                "Phase 9.3B renders the integrity-verified persisted canonical OBJ in the existing "
                "ReconstructionCanvas. Rendering is presentation-only; canonical authority stays with "
                "the persistent scene and metrology is not certified."
            )
            return
        if self._canonical_viewer_error:
            self._canonical_scene_status_value.setText(
                "Canonical authority retained • interactive surface unavailable"
            )
            self._canonical_scene_status_value.setToolTip(self._canonical_viewer_error)
            return
        presentation = self._canonical_scene_presentation
        if presentation is not None and presentation.available and not presentation.viewer_resource_ready:
            self._canonical_scene_status_value.setText(
                "Canonical authority ready • no materialized interactive surface"
            )
            self._canonical_scene_status_value.setToolTip(
                "Phase 9.3B renders only an integrity-verified persistent surface_obj_path resource. "
                "Canonical authority remains valid even when no materialized viewer surface is available; "
                "metrology is not certified."
            )

    def _refresh_canonical_geometry_viewer(self, *, fit_scene: bool = True) -> bool:
        """Load/replace the persisted canonical surface without changing scene authority.

        9.3I validates/materializes the replacement before clearing the active non-empty
        MeshVisual. Successful scene-to-scene replacement can therefore reuse the existing
        VisPy node and avoid one detach/recreate allocation cycle.
        """

        was_loaded = self._reconstruction_canvas.canonical_mesh_loaded
        if self._reconstruction_canvas.interaction_mode == "measure":
            self._cancel_reconstruction_interactive_measurement(switch_to_orbit=True)
        self._canonical_viewer_mesh = None
        self._canonical_viewer_error = ""
        scene = self._persistent_scene
        if scene is None:
            self._reconstruction_canvas.clear_canonical_mesh()
            self._refresh_reconstruction_interactive_measurement_availability()
            self._refresh_viewport_state()
            return False

        presentation = self._canonical_scene_presentation
        if presentation is None or not presentation.viewer_resource_ready:
            self._reconstruction_canvas.clear_canonical_mesh()
            self._refresh_reconstruction_interactive_measurement_availability()
            self._apply_canonical_viewer_status()
            self._refresh_viewport_state()
            return False

        try:
            mesh = load_canonical_viewer_mesh(scene)
            if mesh is None:
                self._reconstruction_canvas.clear_canonical_mesh()
                self._refresh_reconstruction_interactive_measurement_availability()
                self._apply_canonical_viewer_status()
                self._refresh_viewport_state()
                return False
            self._reconstruction_canvas.set_canonical_mesh(
                vertices_viewer=mesh.vertices_viewer,
                faces=mesh.faces,
                vertex_colors_rgba=mesh.vertex_colors_rgba,
                scene_id=mesh.scene_id,
                artifact_id=mesh.artifact_id,
                source_path=mesh.source_path,
            )
        except (OSError, TypeError, ValueError) as exc:
            # A viewer-resource failure cannot revoke already-established canonical
            # authority, but stale geometry must never remain on screen.
            self._reconstruction_canvas.clear_canonical_mesh()
            self._refresh_reconstruction_interactive_measurement_availability()
            self._canonical_viewer_error = str(exc)
            self._viewport_state.setText("Canonical surface unavailable")
            self._viewport_state.setToolTip(str(exc))
            self._apply_canonical_viewer_status()
            return False

        self._canonical_viewer_mesh = mesh
        self._viewport_placeholder.setVisible(False)
        self._viewport_state.setText("Canonical surface mesh")
        self._viewport_state.setToolTip(
            f"Persistent canonical surface: {mesh.vertex_count:,} vertices, {mesh.face_count:,} faces. "
            "Viewer coordinates are presentation-only; persistent canonical geometry remains authoritative."
        )
        if fit_scene:
            self._reconstruction_canvas.fit_scene()
            self._visualizer_fitted_once = True
        self._apply_canonical_viewer_status()
        self._refresh_reconstruction_interactive_measurement_availability()
        return bool(was_loaded and self._reconstruction_canvas.canonical_mesh_loaded)

    @property
    def persistent_scene(self):
        return self._persistent_scene

    @property
    def persistent_authority_snapshot(self) -> PersistentSceneAuthoritySnapshot | None:
        return self._persistent_authority_snapshot

    @property
    def canonical_camera_presentation(self) -> CanonicalCameraTrajectoryPresentation | None:
        return self._canonical_camera_presentation

    def _refresh_canonical_camera_viewer(self, *, fit_scene: bool = True) -> None:
        """Bind only explicitly persisted camera poses to the canonical scene viewer."""

        self._reconstruction_canvas.clear_canonical_camera_trajectory()
        self._canonical_camera_presentation = None
        self._canonical_camera_error = ""
        scene = self._persistent_scene

        if hasattr(self, "_canonical_scene_world_value"):
            if scene is None:
                self._canonical_scene_world_value.setText("No canonical scene")
                self._canonical_scene_world_value.setToolTip("")
                self._canonical_scene_cameras_value.setText("No persistent camera trajectory")
                self._canonical_scene_cameras_value.setToolTip("")
            else:
                self._canonical_scene_world_value.setText("Canonical reconstruction origin • viewer Z-up")
                self._canonical_scene_world_value.setToolTip(
                    "World axes are rooted at the canonical reconstruction origin. The viewer-only "
                    "axis conversion keeps Z visually up and does not establish metric scale or georeferencing."
                )
                self._canonical_scene_cameras_value.setText("Camera trajectory not persisted")
                self._canonical_scene_cameras_value.setToolTip(
                    "Legacy persistent scenes remain valid. Phase 9.3C renders cameras only when the "
                    "camera_trajectory artifact explicitly stores T_reconstruction_from_camera matrices."
                )

        if scene is None:
            self.canonical_camera_presentation_changed.emit(None)
            self._refresh_viewport_state()
            return

        try:
            presentation = load_persistent_canonical_camera_trajectory(scene)
        except (TypeError, ValueError) as exc:
            self._canonical_camera_error = str(exc)
            if hasattr(self, "_canonical_scene_cameras_value"):
                self._canonical_scene_cameras_value.setText("Persistent camera context rejected")
                self._canonical_scene_cameras_value.setToolTip(str(exc))
            # Camera presentation failure cannot revoke canonical geometry authority
            # and must never leave stale frustums/trajectory on screen.
            self.canonical_camera_presentation_changed.emit(None)
            self._refresh_viewport_state()
            return

        if presentation is None:
            self.canonical_camera_presentation_changed.emit(None)
            self._refresh_viewport_state()
            return

        self._reconstruction_canvas.set_canonical_camera_trajectory(
            matrices_reconstruction=presentation.matrices_reconstruction,
            camera_ids=presentation.camera_ids,
            scene_id=presentation.scene_id,
            artifact_id=presentation.artifact_id,
            provider_id=presentation.provider_id,
            total_count=presentation.total_pose_count,
        )
        self._canonical_camera_presentation = presentation
        if hasattr(self, "_canonical_scene_cameras_value"):
            if presentation.trajectory_available:
                text = (
                    f"{presentation.persisted_pose_count:,} persisted poses • "
                    "frustums + trajectory"
                )
            else:
                text = f"{presentation.persisted_pose_count:,} persisted pose • frustum"
            if presentation.total_pose_count != presentation.persisted_pose_count:
                text += f" • {presentation.total_pose_count:,} total"
            self._canonical_scene_cameras_value.setText(text)
            frame = presentation.reconstruction_frame_id or "canonical reconstruction frame"
            self._canonical_scene_cameras_value.setToolTip(
                f"Provider: {presentation.provider_id}\n"
                f"Frame: {frame}\n"
                f"Pose convention: {presentation.pose_convention}\n"
                "Camera poses remain supporting geometry/presentation context; canonical geometry "
                "authority and metric qualification remain separate."
            )
        self._viewport_placeholder.setVisible(False)
        if self._canonical_viewer_mesh is not None:
            self._viewport_state.setText("Canonical scene • cameras + trajectory")
        else:
            self._viewport_state.setText("Canonical camera context")
        self._viewport_state.setToolTip(
            "Phase 9.3C camera frustums and trajectory are loaded from explicitly persisted "
            "T_reconstruction_from_camera evidence. No camera pose or metric scale is invented."
        )
        if fit_scene:
            self._reconstruction_canvas.fit_scene()
            self._visualizer_fitted_once = True
        self.canonical_camera_presentation_changed.emit(presentation)

    @property
    def canonical_semantic_presentation(self) -> CanonicalSemanticScenePresentation | None:
        return self._canonical_semantic_presentation

    def _refresh_canonical_semantic_viewer(
        self, *, preserve_selected_mesh_visual: bool = False
    ) -> None:
        """Bind persisted 3D semantic membership to the loaded canonical mesh."""

        self._reconstruction_canvas.clear_canonical_semantic_objects(
            preserve_selected_mesh_visual=preserve_selected_mesh_visual
        )
        self._canonical_semantic_presentation = None
        self._canonical_semantic_error = ""
        self._canonical_inspection_overlay_presentation = None
        self._canonical_inspection_overlay_error = ""
        scene = self._persistent_scene
        mesh = self._canonical_viewer_mesh
        if scene is None or mesh is None:
            return

        try:
            presentation = load_persistent_canonical_semantics(scene, mesh)
        except (TypeError, ValueError) as exc:
            # Semantic binding failure never revokes the persistent canonical
            # surface or the already-normalized D5 semantic context.  It only
            # disables the 9.3D surface highlight/labels and exposes the reason.
            self._canonical_semantic_error = str(exc)
            if hasattr(self, "_semantic_structure_value"):
                self._semantic_structure_value.setToolTip(
                    "Canonical semantic surface association rejected.\n" + str(exc)
                )
            return
        if presentation is None:
            if self._offline_semantics is not None and hasattr(self, "_semantic_structure_value"):
                self._semantic_structure_value.setToolTip(
                    "Persistent semantic context is available, but no detailed semantic payload snapshot "
                    "is persisted for canonical-surface binding. No 3D membership is fabricated."
                )
            return

        offline = self._offline_semantics
        if offline is None:
            self._canonical_semantic_error = (
                "Canonical semantic binding requires the normalized semantic presentation."
            )
            return
        offline_ids = tuple(item.object_id for item in offline.objects)
        if presentation.object_ids != offline_ids:
            self._canonical_semantic_error = (
                "Canonical semantic object ordering does not match the active semantic presentation."
            )
            if hasattr(self, "_semantic_structure_value"):
                self._semantic_structure_value.setToolTip(
                    "Canonical semantic surface association rejected.\n"
                    + self._canonical_semantic_error
                )
            return

        try:
            self._reconstruction_canvas.set_canonical_semantic_objects(
                object_ids=presentation.object_ids,
                labels=tuple(item.label_text for item in presentation.objects),
                label_positions_viewer=np.asarray(
                    [item.centroid_viewer for item in presentation.objects], dtype=np.float32
                ).reshape((-1, 3)),
                surface_vertices_viewer=tuple(
                    item.surface_vertices_viewer for item in presentation.objects
                ),
                surface_faces=tuple(item.surface_faces for item in presentation.objects),
                scene_id=presentation.scene_id,
                semantic_artifact_id=presentation.semantic_artifact_id,
                canonical_artifact_id=presentation.canonical_artifact_id,
                preserve_selected_mesh_visual=preserve_selected_mesh_visual,
            )
        except (TypeError, ValueError) as exc:
            self._reconstruction_canvas.clear_canonical_semantic_objects()
            self._canonical_semantic_error = str(exc)
            if hasattr(self, "_semantic_structure_value"):
                self._semantic_structure_value.setToolTip(
                    "Canonical semantic viewer presentation rejected.\n" + str(exc)
                )
            return
        self._canonical_semantic_presentation = presentation
        self._semantic_structure_value.setToolTip(
            f"{presentation.binding_text}\n"
            f"Semantic provider: {presentation.semantic_provider_id}\n"
            f"{presentation.authority_text}\n"
            "No YOLO/SAM2 inference or 2D-to-3D promotion is performed while loading this scene."
        )
        if presentation.objects:
            self._viewport_placeholder.setVisible(False)
            if self._canonical_camera_presentation is not None:
                self._viewport_state.setText("Canonical scene • cameras + semantic objects")
            else:
                self._viewport_state.setText("Canonical scene • semantic objects")
            self._viewport_state.setToolTip(
                f"{presentation.binding_text}. Semantic surfaces are supporting membership only; "
                "canonical geometry authority remains with the persistent surface mesh."
            )
        self._refresh_canonical_inspection_overlay()

    @property
    def appearance_representation_presentation(self) -> ReconstructionAppearancePresentation | None:
        return self._appearance_presentation

    @property
    def selected_appearance_representation(self) -> AppearanceRepresentationEntry | None:
        presentation = self._appearance_presentation
        if presentation is None or not self._selected_appearance_artifact_id:
            return None
        return presentation.entry_by_artifact_id(self._selected_appearance_artifact_id)

    def _on_appearance_representation_selected(self, combo_index: int) -> None:
        artifact_id = str(self._appearance_representation_combo.itemData(combo_index) or "").strip()
        self._selected_appearance_artifact_id = artifact_id
        self._appearance_fallback_reason = ""
        self._refresh_appearance_representation_values()
        # F2/F3 still clear stale imagery on metadata changes. F4 performs the
        # actual user-driven switch only from QComboBox.activated, so programmatic
        # scene refresh/selection restoration never launches a view implicitly.
        selected = self.selected_appearance_representation
        if self._interactive_gaussian_visible and (
            selected is None or selected.representation_kind != "gaussian_scene"
        ):
            self._hide_interactive_gaussian()
        if self._gaussian_preview_visible and (
            selected is None or selected.representation_kind != "gaussian_scene"
        ):
            self._hide_gaussian_preview()
        if (self._interactive_nerf_visible or self._nerf_preview_visible) and (
            selected is None or selected.representation_kind != "nerf_scene"
        ):
            self._hide_nerf_visualization()
        self._refresh_gaussian_visualization_controls()
        self._refresh_nerf_visualization_controls()
        self._refresh_nerf_build_controls()
        self._refresh_representation_switching_controls()
        self.appearance_representation_changed.emit(self.selected_appearance_representation)

    def _refresh_appearance_representation_controls(self) -> None:
        previous = self._selected_appearance_artifact_id
        try:
            presentation = build_reconstruction_appearance_presentation(self._persistent_scene)
        except (TypeError, ValueError) as exc:
            self._appearance_presentation = None
            self._selected_appearance_artifact_id = ""
            self._appearance_representation_combo.blockSignals(True)
            self._appearance_representation_combo.clear()
            self._appearance_representation_combo.addItem("Appearance unavailable", "")
            self._appearance_representation_combo.setEnabled(False)
            self._appearance_representation_combo.blockSignals(False)
            values = empty_appearance_representation_text()
            self._appearance_available_value.setText("Appearance metadata rejected")
            self._appearance_available_value.setToolTip(str(exc))
            self._appearance_provider_value.setText(values["provider"])
            self._appearance_artifact_value.setText(values["artifact"])
            self._appearance_render_value.setText(values["render"])
            self._appearance_scale_value.setText(values["scale"])
            self._appearance_state_value.setText("Appearance metadata unavailable")
            self._appearance_state_value.setToolTip(str(exc))
            self._appearance_authority_value.setText(values["authority"])
            self._refresh_digital_twin_workspace_presentation()
            self.appearance_representation_changed.emit(None)
            return

        self._appearance_presentation = presentation
        self._appearance_representation_combo.blockSignals(True)
        self._appearance_representation_combo.clear()
        if presentation is None:
            self._selected_appearance_artifact_id = ""
            self._appearance_representation_combo.addItem("No appearance scene", "")
            self._appearance_representation_combo.setEnabled(False)
        else:
            for entry in presentation.entries:
                self._appearance_representation_combo.addItem(entry.label, entry.artifact_id)
            self._appearance_representation_combo.setEnabled(bool(presentation.entries))
            available_ids = {entry.artifact_id for entry in presentation.entries}
            if previous in available_ids:
                self._selected_appearance_artifact_id = previous
            elif presentation.entries:
                self._selected_appearance_artifact_id = presentation.entries[0].artifact_id
            else:
                self._selected_appearance_artifact_id = ""
            if self._selected_appearance_artifact_id:
                index = self._appearance_representation_combo.findData(self._selected_appearance_artifact_id)
                if index >= 0:
                    self._appearance_representation_combo.setCurrentIndex(index)
            elif not presentation.entries:
                self._appearance_representation_combo.addItem("No neural appearance artifacts", "")
        self._appearance_representation_combo.blockSignals(False)
        self._refresh_appearance_representation_values()
        self._refresh_digital_twin_workspace_presentation()
        self.appearance_representation_changed.emit(self.selected_appearance_representation)

    def _refresh_appearance_representation_values(self) -> None:
        values = empty_appearance_representation_text()
        presentation = self._appearance_presentation
        selected = self.selected_appearance_representation
        self._appearance_available_value.setText(
            presentation.available_text if presentation is not None else values["available"]
        )
        self._appearance_available_value.setToolTip(
            "Persisted Phase 8.7 Gaussian/NeRF appearance artifacts only; this is not backend runtime probing."
        )
        if selected is None:
            self._appearance_provider_value.setText(values["provider"])
            self._appearance_artifact_value.setText(values["artifact"])
            self._appearance_render_value.setText(values["render"])
            self._appearance_scale_value.setText(values["scale"])
        else:
            self._appearance_provider_value.setText(selected.provider_text)
            self._appearance_artifact_value.setText(selected.artifact_id)
            self._appearance_artifact_value.setToolTip(selected.artifact_id)
            self._appearance_render_value.setText(selected.render_text)
            self._appearance_scale_value.setText(selected.scale_text)
        self._appearance_state_value.setText(
            presentation.state_text if presentation is not None else values["state"]
        )
        self._appearance_authority_value.setText(
            presentation.authority_text if presentation is not None else values["authority"]
        )
        self._appearance_authority_value.setToolTip(
            "F1 inspects persisted appearance metadata only. Gaussian/NeRF do not become canonical geometry, measurement, inspection/QC, reference, or certified-metrology authority."
        )

    @property
    def representation_switch_target(self) -> RepresentationSwitchTarget | None:
        return self._representation_switch_target

    @property
    def active_view_representation(self) -> str:
        if self.interactive_gaussian_visible or self.gaussian_preview_visible:
            return "gaussian_scene"
        if self.interactive_nerf_visible or self.nerf_preview_visible:
            return "nerf_scene"
        return "canonical_geometry"

    def _refresh_representation_switching_controls(self) -> None:
        previous = self._representation_switch_target
        target = build_representation_switch_target(
            self.selected_appearance_representation,
            gaussian_presentation=self._gaussian_visualization_presentation,
            nerf_presentation=self._nerf_visualization_presentation,
        )
        self._representation_switch_target = target
        if not hasattr(self, "_appearance_viewport_value"):
            return

        active = self.active_view_representation
        if active == "gaussian_scene":
            if self.interactive_gaussian_visible:
                text = "Interactive Gaussian appearance • appearance only"
                tooltip = "9.4E is showing persisted Gaussian appearance interactively; canonical geometry remains the picking/measurement authority."
            else:
                text = "Gaussian appearance • appearance only"
                tooltip = "F4 is showing the already-qualified F2 Gaussian preview. Canonical geometry remains authoritative."
        elif active == "nerf_scene":
            if self.interactive_nerf_visible:
                text = "Interactive NeRF appearance • appearance only"
                tooltip = "9.5E is showing external Nerfacto renders driven by the canonical viewer camera; canonical geometry remains the picking/measurement authority."
            else:
                text = "NeRF appearance • appearance only"
                tooltip = "The qualified static NeRF preview is visible. Canonical geometry remains authoritative."
        elif target is not None and not target.ready:
            text = target.unavailable_text
            tooltip = (
                "The selected appearance preview is unavailable, so canonical geometry remains visible. "
                "F5 reports the fallback reason and does not substitute another neural representation."
            )
        else:
            text = canonical_view_text()
            tooltip = (
                "Canonical Reconstruction view is active. Re-select/activate Gaussian or NeRF to show an existing "
                "qualified appearance preview; no backend is launched."
            )
        self._appearance_viewport_value.setText(text)
        self._appearance_viewport_value.setToolTip(tooltip)
        if target != previous:
            self.representation_switching_changed.emit(target)
        self._refresh_appearance_resource_fallback_controls()
        self._refresh_digital_twin_workspace_presentation()

    @property
    def appearance_resource_fallback_presentation(self) -> AppearanceResourceFallbackPresentation | None:
        return self._appearance_resource_fallback_presentation

    def _refresh_appearance_resource_fallback_controls(self) -> None:
        if not hasattr(self, "_appearance_resource_state_value"):
            return
        previous = self._appearance_resource_fallback_presentation
        values = empty_appearance_resource_fallback_text()
        try:
            presentation = build_appearance_resource_fallback_presentation(
                self._persistent_scene,
                selected=self.selected_appearance_representation,
                switch_target=self._representation_switch_target,
                active_view=self.active_view_representation,
                failure_reason=self._appearance_fallback_reason,
            )
        except (OSError, TypeError, ValueError) as exc:
            self._appearance_resource_fallback_presentation = None
            self._appearance_resource_device_value.setText(values["device"])
            self._appearance_resource_resources_value.setText(values["resources"])
            self._appearance_resource_presentation_value.setText(values["presentation"])
            self._appearance_resource_fallback_value.setText(values["fallback"])
            self._appearance_resource_pipeline_value.setText(values["pipeline"])
            self._appearance_resource_state_value.setText("Resource/fallback presentation unavailable")
            self._appearance_resource_state_value.setToolTip(str(exc))
            if previous is not None:
                self.appearance_resource_fallback_changed.emit(None)
            return

        self._appearance_resource_fallback_presentation = presentation
        self._appearance_resource_device_value.setText(presentation.device_text)
        self._appearance_resource_device_value.setToolTip(presentation.resource_details)
        self._appearance_resource_resources_value.setText(presentation.resources_text)
        self._appearance_resource_resources_value.setToolTip(
            "Existing F2/F3 static previews are backend-free display artifacts. Startup GPU state is advisory and does not disable an existing preview."
        )
        self._appearance_resource_presentation_value.setText(presentation.presentation_text)
        self._appearance_resource_fallback_value.setText(presentation.fallback_text)
        self._appearance_resource_fallback_value.setToolTip(
            "F5 always fails closed to canonical geometry. It never silently substitutes Gaussian for NeRF or NeRF for Gaussian."
        )
        self._appearance_resource_pipeline_value.setText(presentation.pipeline_text)
        self._appearance_resource_pipeline_value.setToolTip(
            presentation.pipeline_details or "No appearance-stage fallback was recorded in the persistent scene."
        )
        self._appearance_resource_state_value.setText(presentation.state_text)
        self._appearance_resource_state_value.setToolTip(presentation.state_details)
        if presentation != previous:
            self.appearance_resource_fallback_changed.emit(presentation)

    def _on_appearance_representation_activated(self, combo_index: int) -> None:
        # ``activated`` is user-driven, unlike currentIndexChanged during internal
        # state restoration. This keeps F4 switching explicit and deterministic.
        artifact_id = str(self._appearance_representation_combo.itemData(combo_index) or "").strip()
        if artifact_id and artifact_id != self._selected_appearance_artifact_id:
            self._selected_appearance_artifact_id = artifact_id
            self._refresh_appearance_representation_values()
            self._refresh_gaussian_visualization_controls()
            self._refresh_nerf_visualization_controls()
        self._activate_selected_appearance_representation()

    def _activate_selected_appearance_representation(self) -> None:
        self._appearance_fallback_reason = ""
        self._refresh_representation_switching_controls()
        target = self._representation_switch_target
        if target is None:
            self._appearance_fallback_reason = "No qualified appearance target is selected"
            self._hide_active_appearance_preview()
            self._refresh_representation_switching_controls()
            return
        if not target.ready:
            self._appearance_fallback_reason = target.state_text or f"{target.label} preview unavailable"
            self._hide_active_appearance_preview()
            self._refresh_representation_switching_controls()
            return
        if target.representation_kind == "gaussian_scene":
            self._show_gaussian_preview()
        elif target.representation_kind == "nerf_scene":
            self._show_nerf_preview()
        self._refresh_representation_switching_controls()

    def _hide_active_appearance_preview(self) -> None:
        if self._interactive_gaussian_visible:
            self._hide_interactive_gaussian()
        if self._gaussian_preview_visible:
            self._hide_gaussian_preview()
        if self._interactive_nerf_visible or self._nerf_preview_visible:
            self._hide_nerf_visualization()

    def _nerf_settings_from_controls(self) -> NerfDisplaySettings:
        profile = str(self._nerf_render_profile_combo.currentData() or NERF_DEFAULT_RENDER_PROFILE)
        return NerfDisplaySettings(
            render_profile=profile,
            opacity_percent=int(self._nerf_opacity_slider.value()),
            auto_refresh=bool(self._nerf_auto_refresh_toggle.isChecked()),
        )

    def _set_nerf_scene_layer_checked(self, checked: bool) -> None:
        toggle = getattr(self, "_nerf_scene_layer_toggle", None)
        if toggle is None or toggle.isChecked() == bool(checked):
            return
        toggle.blockSignals(True)
        toggle.setChecked(bool(checked))
        toggle.blockSignals(False)

    def _on_nerf_scene_layer_toggled(self, checked: bool) -> None:
        if checked:
            if not self.interactive_nerf_visible:
                self._show_interactive_nerf()
            if not self.interactive_nerf_visible:
                self._set_nerf_scene_layer_checked(False)
        elif self.interactive_nerf_visible:
            self._hide_interactive_nerf()
            self._refresh_nerf_visualization_controls()

    def _on_nerf_display_controls_changed(self, _value: object = None) -> None:
        try:
            settings = self._nerf_settings_from_controls()
        except ValueError:
            return
        previous = self._nerf_display_settings
        self._nerf_display_settings = settings
        self._nerf_opacity_value.setText(f"{settings.opacity_percent}%")
        if self._interactive_nerf_visible:
            self._apply_digital_twin_interaction_appearance_policy()
            if (
                settings.auto_refresh
                and (settings.render_width, settings.render_height)
                != (previous.render_width, previous.render_height)
            ):
                self._request_interactive_nerf_render(force=True)
        self._refresh_nerf_display_summary()

    def _reset_nerf_display_controls(self) -> None:
        self._nerf_render_profile_combo.blockSignals(True)
        self._nerf_opacity_slider.blockSignals(True)
        self._nerf_auto_refresh_toggle.blockSignals(True)
        index = self._nerf_render_profile_combo.findData(NERF_DEFAULT_RENDER_PROFILE)
        if index >= 0:
            self._nerf_render_profile_combo.setCurrentIndex(index)
        self._nerf_opacity_slider.setValue(100)
        self._nerf_auto_refresh_toggle.setChecked(True)
        self._nerf_render_profile_combo.blockSignals(False)
        self._nerf_opacity_slider.blockSignals(False)
        self._nerf_auto_refresh_toggle.blockSignals(False)
        self._nerf_display_settings = NerfDisplaySettings()
        self._nerf_opacity_value.setText("100%")
        if self._interactive_nerf_visible:
            self._apply_digital_twin_interaction_appearance_policy()
            self._request_interactive_nerf_render(force=True)
        self._refresh_nerf_display_summary()

    def _refresh_nerf_display_summary(self) -> None:
        if not hasattr(self, "_nerf_display_value"):
            return
        selected = self.selected_appearance_representation
        if selected is None or selected.representation_kind != "nerf_scene":
            self._nerf_display_value.setText("Interactive display unavailable")
            return
        self._nerf_display_value.setText(self._nerf_display_settings.status_text)
        self._nerf_display_value.setToolTip(
            "Presentation-only settings. They do not modify the persisted NeRF artifact or any scene authority."
        )

    @property
    def nerf_runtime_health(self) -> NerfViewerRuntimeHealth | None:
        return self._nerf_runtime_health

    def _refresh_nerf_runtime_health(self) -> None:
        persistent = self._persistent_scene
        if persistent is None or not bool(getattr(persistent, "canonical_geometry_authoritative", False)):
            self._nerf_runtime_health = None
            if hasattr(self, "_nerf_resources_value"):
                self._nerf_resources_value.setText("External renderer unloaded")
                self._nerf_performance_value.setText("No interactive render measured")
                self._nerf_lifecycle_value.setText("Unloaded • no NeRF worker resident")
            return
        health = build_nerf_viewer_runtime_health(
            policy=self._nerf_resource_policy,
            worker_state=self._nerf_worker_client.state,
            worker_pid=self._nerf_worker_client.worker_pid,
            telemetry=self._nerf_worker_telemetry or self._nerf_worker_client.last_telemetry,
            last_render_ms=self._interactive_nerf_last_render_ms,
            worker_start_count=self._nerf_worker_start_count,
            worker_release_count=self._nerf_worker_release_count,
            queue=self._nerf_worker_client.queue_snapshot,
            canonical_geometry_authoritative=True,
        )
        self._nerf_runtime_health = health
        if not hasattr(self, "_nerf_resources_value"):
            return
        self._nerf_resources_value.setText(health.resources_text)
        self._nerf_performance_value.setText(health.performance_text)
        self._nerf_lifecycle_value.setText(health.lifecycle_text)
        self._nerf_resources_value.setToolTip(
            "CUDA allocation/reservation values come from torch inside the isolated Nerfstudio worker. "
            "They describe that external process only, not total application VRAM usage."
        )
        self._nerf_performance_value.setToolTip(
            "Observed full-image render latency only. Phase 9.5G bounds the request queue; it does not certify NeRF visual quality."
        )
        self._nerf_lifecycle_value.setToolTip(
            "At most one render executes and one latest view waits. Hide, Preview, scene switch/clear and shutdown stop the isolated worker; the real 9.5G target check verifies the worker PID leaves nvidia-smi."
        )

    def _gaussian_settings_from_controls(self) -> GaussianDisplaySettings:
        density = int(self._gaussian_density_combo.currentData() or 100)
        return GaussianDisplaySettings(
            density_percent=density,
            size_percent=int(self._gaussian_size_slider.value()),
            opacity_percent=int(self._gaussian_opacity_slider.value()),
        )

    def _set_gaussian_scene_layer_checked(self, checked: bool) -> None:
        toggle = getattr(self, "_gaussian_scene_layer_toggle", None)
        if toggle is None or toggle.isChecked() == bool(checked):
            return
        toggle.blockSignals(True)
        toggle.setChecked(bool(checked))
        toggle.blockSignals(False)

    def _on_gaussian_scene_layer_toggled(self, checked: bool) -> None:
        if checked:
            if not self.interactive_gaussian_visible:
                self._show_interactive_gaussian()
            if not self.interactive_gaussian_visible:
                self._set_gaussian_scene_layer_checked(False)
        elif self.interactive_gaussian_visible:
            self._hide_interactive_gaussian()

    def _on_gaussian_display_controls_changed(self, _value: object = None) -> None:
        try:
            settings = self._gaussian_settings_from_controls()
        except ValueError:
            return
        self._gaussian_display_settings = settings
        self._gaussian_size_value.setText(f"{settings.size_percent}%")
        self._gaussian_opacity_value.setText(f"{settings.opacity_percent}%")
        if self.interactive_gaussian_visible and self._interactive_gaussian_scene is not None:
            # User slider drags can generate hundreds of valueChanged signals. 9.4G
            # coalesces only active drags to a ~60 Hz presentation cadence; discrete
            # combo/programmatic changes remain immediate for deterministic UX/tests.
            slider_dragging = bool(
                self._gaussian_size_slider.isSliderDown()
                or self._gaussian_opacity_slider.isSliderDown()
            )
            if slider_dragging:
                self._gaussian_display_update_pending = True
                if not self._gaussian_display_update_timer.isActive():
                    self._gaussian_display_update_timer.start()
            else:
                self._gaussian_display_update_pending = False
                self._gaussian_display_update_timer.stop()
                self._apply_interactive_gaussian_display_settings()
        else:
            self._refresh_gaussian_display_summary()
            self._refresh_gaussian_runtime_health()

    def _flush_pending_gaussian_display_update(self) -> None:
        if not self._gaussian_display_update_pending:
            return
        self._gaussian_display_update_pending = False
        if self.interactive_gaussian_visible and self._interactive_gaussian_scene is not None:
            self._apply_interactive_gaussian_display_settings()

    def _reset_gaussian_display_controls(self) -> None:
        self._gaussian_density_combo.blockSignals(True)
        self._gaussian_size_slider.blockSignals(True)
        self._gaussian_opacity_slider.blockSignals(True)
        self._gaussian_density_combo.setCurrentIndex(0)
        self._gaussian_size_slider.setValue(100)
        self._gaussian_opacity_slider.setValue(100)
        self._gaussian_density_combo.blockSignals(False)
        self._gaussian_size_slider.blockSignals(False)
        self._gaussian_opacity_slider.blockSignals(False)
        self._gaussian_display_settings = GaussianDisplaySettings()
        self._gaussian_size_value.setText("100%")
        self._gaussian_opacity_value.setText("100%")
        if self.interactive_gaussian_visible and self._interactive_gaussian_scene is not None:
            self._apply_interactive_gaussian_display_settings()
        else:
            self._refresh_gaussian_display_summary()

    def _apply_interactive_gaussian_display_settings(self) -> None:
        scene = self._interactive_gaussian_scene
        if scene is None:
            self._gaussian_display_buffers = None
            self._refresh_gaussian_display_summary()
            self._refresh_gaussian_runtime_health()
            return
        started_at = monotonic()
        buffers = apply_gaussian_display_settings(scene, self._gaussian_display_settings)
        self._reconstruction_canvas.set_interactive_gaussian_scene(
            buffers.positions_reconstruction,
            buffers.colors_rgba,
            buffers.sizes_px,
            scene_id=scene.scene_id,
            artifact_id=scene.artifact_id,
            total_count=scene.total_gaussian_count,
        )
        self._last_gaussian_apply_latency_ms = max(0.0, (monotonic() - started_at) * 1000.0)
        self._gaussian_display_buffers = buffers
        self._refresh_gaussian_display_summary()
        self._refresh_gaussian_runtime_health()

    @property
    def gaussian_runtime_health(self) -> GaussianViewerRuntimeHealth | None:
        return self._gaussian_runtime_health

    def _refresh_gaussian_runtime_health(self) -> None:
        canvas = getattr(self, "_reconstruction_canvas", None)
        if canvas is None:
            return
        scene = self._interactive_gaussian_scene
        persistent = self._persistent_scene
        health = build_gaussian_viewer_runtime_health(
            policy=self._gaussian_resource_policy,
            scene=scene,
            buffers=self._gaussian_display_buffers,
            resident=bool(self.interactive_gaussian_visible and scene is not None),
            had_release=bool(self._gaussian_release_count),
            load_latency_ms=self._last_gaussian_load_latency_ms,
            apply_latency_ms=self._last_gaussian_apply_latency_ms,
            load_count=self._gaussian_load_count,
            release_count=self._gaussian_release_count,
            canvas_resource_generation=int(getattr(canvas, "interactive_gaussian_resource_generation", 0)),
            canvas_release_count=int(getattr(canvas, "interactive_gaussian_release_count", 0)),
            canonical_geometry_authoritative=bool(
                getattr(persistent, "canonical_geometry_authoritative", False)
            ) if persistent is not None else False,
            last_total_gaussian_count=self._last_gaussian_total_count,
            last_source_display_count=self._last_gaussian_source_display_count,
        )
        self._gaussian_runtime_health = health
        if not hasattr(self, "_gaussian_resources_value"):
            return
        self._gaussian_resources_value.setText(health.resources_text)
        self._gaussian_performance_value.setText(health.performance_text)
        self._gaussian_lifecycle_value.setText(health.lifecycle_text)
        self._gaussian_resources_value.setToolTip(
            "9.4G reports CPU/display staging only. It does not infer OpenGL/CUDA VRAM allocation. "
            + self._gaussian_resource_policy.budget_text
        )
        self._gaussian_performance_value.setToolTip(
            "UI-side safe scene.pt load and VisPy display-apply time on this run only; not Gaussian training/render backend performance."
        )
        self._gaussian_lifecycle_value.setToolTip(
            "Interactive Gaussian CPU arrays and VisPy marker buffers are released on Hide, Preview, scene switch/clear and shutdown; they reload on demand."
        )

    def _refresh_gaussian_display_summary(self) -> None:
        if not hasattr(self, "_gaussian_display_value"):
            return
        buffers = self._gaussian_display_buffers
        if buffers is not None and self.interactive_gaussian_visible:
            self._gaussian_display_value.setText(buffers.status_text)
            self._gaussian_display_value.setToolTip(
                "Presentation-only controls over already-persisted Gaussian display buffers. "
                "Canonical geometry, scale, measurement and QC authority are unchanged."
            )
            return
        presentation = self._gaussian_visualization_presentation
        selected = self.selected_appearance_representation
        if (
            presentation is None
            or selected is None
            or selected.representation_kind != "gaussian_scene"
        ):
            self._gaussian_display_value.setText("Interactive display unavailable")
            self._gaussian_display_value.setToolTip(
                "Select a persisted Gaussian appearance representation to enable interactive display controls."
            )
            return
        settings = self._gaussian_display_settings
        self._gaussian_display_value.setText(
            f"Ready • density {settings.density_percent}% • size {settings.size_percent}% • "
            f"opacity {settings.opacity_percent}%"
        )
        self._gaussian_display_value.setToolTip(
            "These settings apply the next time Interactive Gaussian appearance is shown."
        )

    @property
    def gaussian_visualization_presentation(self) -> GaussianVisualizationPresentation | None:
        return self._gaussian_visualization_presentation

    @property
    def gaussian_preview_visible(self) -> bool:
        return bool(self._gaussian_preview_visible and not self._appearance_preview.isHidden())

    @property
    def interactive_gaussian_visible(self) -> bool:
        return bool(
            self._interactive_gaussian_visible
            and getattr(self._reconstruction_canvas, "interactive_gaussian_visible", False)
        )

    def _refresh_gaussian_visualization_controls(self) -> None:
        values = empty_gaussian_visualization_text()
        previous = self._gaussian_visualization_presentation
        try:
            presentation = build_gaussian_visualization_presentation(self._persistent_scene)
        except (OSError, TypeError, ValueError) as exc:
            presentation = None
            self._gaussian_provider_value.setText(values["provider"])
            self._gaussian_scene_value.setText(values["scene"])
            self._gaussian_preview_value.setText("Preview metadata rejected")
            self._gaussian_preview_value.setToolTip(str(exc))
            self._gaussian_scale_value.setText(values["scale"])
            self._gaussian_state_value.setText("Gaussian visualization unavailable")
            self._gaussian_state_value.setToolTip(str(exc))
            self._gaussian_authority_value.setText(values["authority"])
            self._gaussian_show_button.setEnabled(False)
            self._gaussian_interactive_button.setEnabled(False)
            self._gaussian_hide_button.setEnabled(False)
            for control in (
                self._gaussian_density_combo,
                self._gaussian_size_slider,
                self._gaussian_opacity_slider,
                self._gaussian_reset_display_button,
            ):
                control.setEnabled(False)
            if hasattr(self, "_gaussian_scene_layer_toggle"):
                self._gaussian_scene_layer_toggle.setEnabled(False)
            self._hide_interactive_gaussian()
            self._hide_gaussian_preview()
            self._gaussian_visualization_presentation = None
            if previous is not None:
                self.gaussian_visualization_changed.emit(None)
            return

        self._gaussian_visualization_presentation = presentation
        selected = self.selected_appearance_representation
        gaussian_selected = bool(
            selected is not None and selected.representation_kind == "gaussian_scene"
        )
        resource_blocked = bool(
            self._digital_twin_processing_snapshot.is_active
            or (self._controller is not None and self._controller.is_active)
        )
        if presentation is None:
            self._gaussian_provider_value.setText(values["provider"])
            self._gaussian_scene_value.setText(values["scene"])
            self._gaussian_preview_value.setText(values["preview"])
            self._gaussian_scale_value.setText(values["scale"])
            self._gaussian_state_value.setText(values["state"])
            self._gaussian_authority_value.setText(values["authority"])
            self._gaussian_show_button.setEnabled(False)
            self._gaussian_interactive_button.setEnabled(False)
            self._gaussian_hide_button.setEnabled(False)
            self._hide_interactive_gaussian()
            self._hide_gaussian_preview()
        else:
            self._gaussian_provider_value.setText(presentation.provider_text)
            self._gaussian_scene_value.setText(presentation.scene_text)
            self._gaussian_preview_value.setText(presentation.preview_text)
            self._gaussian_preview_value.setToolTip(presentation.preview_path or presentation.state_text)
            self._gaussian_scale_value.setText(presentation.scale_text)
            if presentation.ready and not gaussian_selected:
                self._gaussian_state_value.setText(
                    "Gaussian preview ready • select Gaussian above to visualize"
                )
            else:
                self._gaussian_state_value.setText(presentation.state_text)
            self._gaussian_state_value.setToolTip(
                "F2 supplies the backend-authored static Gaussian preview. "
                "F4 coordinates switching; F5 reports resource state and canonical fallback without launching a backend."
            )
            self._gaussian_authority_value.setText(presentation.authority_text)
            self._gaussian_show_button.setEnabled(bool(presentation.ready and gaussian_selected))
            self._gaussian_interactive_button.setEnabled(
                bool(gaussian_selected and self._persistent_scene is not None and not resource_blocked)
            )
            self._gaussian_hide_button.setEnabled(bool(self.gaussian_preview_visible or self.interactive_gaussian_visible))
            if not gaussian_selected:
                if self._interactive_gaussian_visible:
                    self._hide_interactive_gaussian()
                if self._gaussian_preview_visible:
                    self._hide_gaussian_preview()

        controls_enabled = bool(
            presentation is not None
            and gaussian_selected
            and self._persistent_scene is not None
        )
        for control in (
            self._gaussian_density_combo,
            self._gaussian_size_slider,
            self._gaussian_opacity_slider,
            self._gaussian_reset_display_button,
        ):
            control.setEnabled(controls_enabled)
        if hasattr(self, "_gaussian_scene_layer_toggle"):
            self._gaussian_scene_layer_toggle.setEnabled(bool(controls_enabled and not resource_blocked))
            self._set_gaussian_scene_layer_checked(self.interactive_gaussian_visible)
        self._refresh_gaussian_display_summary()
        if presentation != previous:
            self.gaussian_visualization_changed.emit(presentation)

    def _show_gaussian_preview(self) -> None:
        presentation = self._gaussian_visualization_presentation
        selected = self.selected_appearance_representation
        if (
            presentation is None
            or not presentation.ready
            or not presentation.preview_path
            or selected is None
            or selected.representation_kind != "gaussian_scene"
            or selected.artifact_id != presentation.artifact_id
        ):
            self._appearance_fallback_reason = "Gaussian preview cannot be shown for the current selection"
            self._gaussian_state_value.setText(self._appearance_fallback_reason)
            self._refresh_appearance_resource_fallback_controls()
            return
        if self._interactive_gaussian_visible:
            self._hide_interactive_gaussian()
        if self._interactive_nerf_visible or self._nerf_preview_visible:
            self._hide_nerf_visualization()
        try:
            self._appearance_preview.set_image_opacity(100)
            self._appearance_preview.set_preview(
                presentation.preview_path,
                title="GAUSSIAN SPLATTING PREVIEW • APPEARANCE ONLY",
            )
            self._appearance_preview.set_footer(
                "Backend-authored static render • canonical geometry remains authoritative"
            )
            self._appearance_preview.set_mouse_passthrough(False)
        except (OSError, ValueError) as exc:
            self._appearance_fallback_reason = "Gaussian preview image could not be displayed"
            self._gaussian_state_value.setText(self._appearance_fallback_reason)
            self._gaussian_state_value.setToolTip(str(exc))
            self._gaussian_show_button.setEnabled(False)
            self._refresh_appearance_resource_fallback_controls()
            return
        # The overlay was inserted after the canvas but before the normal viewport
        # tools/HUD/gizmo, so showing it preserves those controls above the image.
        self._appearance_fallback_reason = ""
        self._appearance_preview.show()
        self._gaussian_preview_visible = True
        self._viewport_placeholder.setVisible(False)
        self._viewport_state.setText("Gaussian Splatting preview • appearance only")
        self._viewport_state.setToolTip(
            "Backend-authored static Gaussian render. Canonical geometry remains authoritative; "
            "F2 performs no Gaussian rendering in the application process."
        )
        self._gaussian_state_value.setText("Gaussian backend preview visible • static F2 visualization")
        self._gaussian_show_button.setEnabled(False)
        self._gaussian_hide_button.setEnabled(True)
        self._refresh_representation_switching_controls()

    def _show_interactive_gaussian(self) -> None:
        failure = self._interactive_resource_admission_failure("interactive Gaussian")
        if failure:
            self.set_error(failure)
            if hasattr(self, "_gaussian_state_value"):
                self._gaussian_state_value.setText(failure)
                self._gaussian_state_value.setToolTip(
                    "Phase 9.6G blocks dynamic appearance residency while Reconstruction or an external build owns resources."
                )
            return
        selected = self.selected_appearance_representation
        if (
            self._persistent_scene is None
            or selected is None
            or selected.representation_kind != "gaussian_scene"
        ):
            self._gaussian_state_value.setText("Interactive Gaussian viewer unavailable for current selection")
            return
        load_started_at = monotonic()
        try:
            interactive = load_interactive_gaussian_scene(
                self._persistent_scene,
                max_gaussians=self._gaussian_resource_policy.max_interactive_gaussians,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._interactive_gaussian_scene = None
            self._interactive_gaussian_visible = False
            self._gaussian_display_buffers = None
            self._set_gaussian_scene_layer_checked(False)
            self._refresh_gaussian_display_summary()
            self._gaussian_state_value.setText("Interactive Gaussian viewer unavailable")
            self._gaussian_state_value.setToolTip(str(exc))
            self._gaussian_interactive_button.setEnabled(False)
            return
        if selected.artifact_id != interactive.artifact_id:
            self._gaussian_state_value.setText("Interactive Gaussian artifact does not match current selection")
            return
        if self._interactive_nerf_visible or self._nerf_preview_visible:
            self._hide_nerf_visualization()
        if self._gaussian_preview_visible:
            self._hide_gaussian_preview()
        self._last_gaussian_load_latency_ms = max(0.0, (monotonic() - load_started_at) * 1000.0)
        self._gaussian_load_count += 1
        self._last_gaussian_total_count = int(interactive.total_gaussian_count)
        self._last_gaussian_source_display_count = int(interactive.display_gaussian_count)
        self._interactive_gaussian_scene = interactive
        self._interactive_gaussian_visible = True
        self._apply_interactive_gaussian_display_settings()
        self._set_gaussian_scene_layer_checked(True)
        self._appearance_preview.hide()
        self._viewport_placeholder.setVisible(False)
        self._viewport_state.setText("Interactive Gaussian appearance • canonical picking")
        self._viewport_state.setToolTip(
            "Persisted Gaussian means/colors/scales are displayed interactively without gsplat imports. "
            "Canonical geometry remains the only picking and measurement authority."
        )
        self._gaussian_state_value.setText(interactive.state_text)
        self._gaussian_state_value.setToolTip(interactive.authority_text)
        self._gaussian_interactive_button.setEnabled(False)
        self._gaussian_show_button.setEnabled(True)
        self._gaussian_hide_button.setEnabled(True)
        self._refresh_digital_twin_resource_coordination()
        self._refresh_representation_switching_controls()

    def _hide_interactive_gaussian(self) -> None:
        if not hasattr(self, "_reconstruction_canvas"):
            return
        had_resident_resources = bool(
            self._interactive_gaussian_scene is not None
            or self._gaussian_display_buffers is not None
            or self._reconstruction_canvas.interactive_gaussian_visible
        )
        self._gaussian_display_update_pending = False
        if hasattr(self, "_gaussian_display_update_timer"):
            self._gaussian_display_update_timer.stop()
        self._reconstruction_canvas.clear_interactive_gaussian_scene()
        self._interactive_gaussian_visible = False
        # 9.4G intentionally drops both the source and derived NumPy staging
        # arrays. Re-enabling the layer performs a fresh integrity-checked CPU load.
        self._interactive_gaussian_scene = None
        self._gaussian_display_buffers = None
        if had_resident_resources:
            self._gaussian_release_count += 1
        self._set_gaussian_scene_layer_checked(False)
        self._refresh_gaussian_display_summary()
        self._refresh_gaussian_runtime_health()
        self._refresh_digital_twin_resource_coordination()
        selected = self.selected_appearance_representation
        presentation = self._gaussian_visualization_presentation
        if hasattr(self, "_gaussian_interactive_button"):
            self._gaussian_interactive_button.setEnabled(
                bool(
                    self._persistent_scene is not None
                    and selected is not None
                    and selected.representation_kind == "gaussian_scene"
                )
            )
        if hasattr(self, "_gaussian_hide_button"):
            self._gaussian_hide_button.setEnabled(self.gaussian_preview_visible)
        if (
            hasattr(self, "_gaussian_state_value")
            and presentation is not None
            and selected is not None
            and selected.representation_kind == "gaussian_scene"
        ):
            self._gaussian_state_value.setText(presentation.state_text)
        self._refresh_viewport_state()
        self._refresh_representation_switching_controls()

    def _hide_gaussian_visualization(self) -> None:
        if self._interactive_gaussian_visible:
            self._hide_interactive_gaussian()
        if self._gaussian_preview_visible:
            self._hide_gaussian_preview()

    def _hide_gaussian_preview(self) -> None:
        if not hasattr(self, "_appearance_preview"):
            return
        presentation = self._gaussian_visualization_presentation
        selected = self.selected_appearance_representation
        if self._gaussian_preview_visible:
            self._appearance_preview.hide()
        self._gaussian_preview_visible = False
        if hasattr(self, "_gaussian_hide_button"):
            self._gaussian_hide_button.setEnabled(self.interactive_gaussian_visible)
        if hasattr(self, "_gaussian_show_button"):
            self._gaussian_show_button.setEnabled(
                bool(
                    presentation is not None
                    and presentation.ready
                    and selected is not None
                    and selected.representation_kind == "gaussian_scene"
                )
            )
        if (
            hasattr(self, "_gaussian_state_value")
            and self._gaussian_state_value.text()
            == "Gaussian backend preview visible • static F2 visualization"
            and presentation is not None
            and presentation.ready
            and selected is not None
            and selected.representation_kind == "gaussian_scene"
        ):
            self._gaussian_state_value.setText(presentation.state_text)
        self._refresh_viewport_state()
        self._refresh_representation_switching_controls()

    @property
    def nerf_visualization_presentation(self) -> NerfVisualizationPresentation | None:
        return self._nerf_visualization_presentation

    @property
    def nerf_preview_visible(self) -> bool:
        return bool(self._nerf_preview_visible and not self._appearance_preview.isHidden())

    @property
    def interactive_nerf_visible(self) -> bool:
        return bool(self._interactive_nerf_visible)

    def _refresh_nerf_visualization_controls(self) -> None:
        values = empty_nerf_visualization_text()
        previous = self._nerf_visualization_presentation
        try:
            presentation = build_nerf_visualization_presentation(self._persistent_scene)
        except (OSError, TypeError, ValueError) as exc:
            presentation = None
            interactive = None
            interactive_error = str(exc)
        else:
            interactive_error = ""
            try:
                interactive = build_interactive_nerf_presentation(self._persistent_scene)
            except (OSError, TypeError, ValueError) as exc:
                interactive = None
                interactive_error = str(exc)

        self._nerf_visualization_presentation = presentation
        self._interactive_nerf_presentation = interactive
        selected = self.selected_appearance_representation
        nerf_selected = bool(
            selected is not None and selected.representation_kind == "nerf_scene"
        )
        resource_blocked = bool(
            self._digital_twin_processing_snapshot.is_active
            or (self._controller is not None and self._controller.is_active)
        )

        if presentation is None:
            self._nerf_provider_value.setText(values["provider"])
            self._nerf_scene_value.setText(values["scene"])
            self._nerf_preview_value.setText(values["preview"])
            self._nerf_scale_value.setText(values["scale"])
            self._nerf_state_value.setText("NeRF visualization unavailable")
            self._nerf_state_value.setToolTip(interactive_error)
            self._nerf_authority_value.setText(values["authority"])
            self._nerf_interactive_button.setEnabled(False)
            self._nerf_show_button.setEnabled(False)
            self._nerf_hide_button.setEnabled(False)
            for control in (
                self._nerf_render_profile_combo,
                self._nerf_opacity_slider,
                self._nerf_auto_refresh_toggle,
                self._nerf_refresh_view_button,
                self._nerf_reset_display_button,
            ):
                control.setEnabled(False)
            if hasattr(self, "_nerf_scene_layer_toggle"):
                self._nerf_scene_layer_toggle.setEnabled(False)
            self._hide_nerf_visualization(refresh=False)
        else:
            self._nerf_provider_value.setText(presentation.provider_text)
            self._nerf_scene_value.setText(presentation.scene_text)
            self._nerf_preview_value.setText(presentation.preview_text)
            self._nerf_preview_value.setToolTip(presentation.preview_path or presentation.state_text)
            self._nerf_scale_value.setText(presentation.scale_text)
            if self._interactive_nerf_visible:
                state = self._nerf_worker_client.state
                if state == "rendering":
                    text = "Interactive Nerfacto • rendering current canonical view"
                elif self._nerf_worker_client.is_ready:
                    text = "Interactive Nerfacto ready • navigate canonical view to update"
                elif state in {"starting", "stopping"}:
                    text = f"Interactive Nerfacto • {state}"
                else:
                    text = "Interactive Nerfacto • external renderer unavailable"
                self._nerf_state_value.setText(text)
            elif presentation.ready and not nerf_selected:
                self._nerf_state_value.setText(
                    "NeRF appearance ready • select NeRF above to visualize"
                )
            elif interactive is None and interactive_error:
                self._nerf_state_value.setText(presentation.state_text)
                self._nerf_state_value.setToolTip(
                    "Static preview remains available. Interactive 9.5E validation failed closed: "
                    + interactive_error
                )
            else:
                self._nerf_state_value.setText(presentation.state_text)
            if not self._nerf_state_value.toolTip():
                self._nerf_state_value.setToolTip(
                    "9.5E launches Nerfstudio only in the isolated external runtime. Navigation is driven by the canonical viewer; ruler picking always uses canonical geometry."
                )
            self._nerf_authority_value.setText(presentation.authority_text)
            self._nerf_interactive_button.setEnabled(
                bool(
                    interactive is not None
                    and interactive.ready
                    and nerf_selected
                    and not self._interactive_nerf_visible
                    and not resource_blocked
                )
            )
            self._nerf_show_button.setEnabled(
                bool(presentation.ready and nerf_selected and not self._nerf_preview_visible)
            )
            self._nerf_hide_button.setEnabled(
                bool(self._interactive_nerf_visible or self._nerf_preview_visible)
            )
            if not nerf_selected and (self._interactive_nerf_visible or self._nerf_preview_visible):
                self._hide_nerf_visualization(refresh=False)

        controls_enabled = bool(
            interactive is not None
            and interactive.ready
            and nerf_selected
            and self._persistent_scene is not None
        )
        for control in (
            self._nerf_render_profile_combo,
            self._nerf_opacity_slider,
            self._nerf_auto_refresh_toggle,
            self._nerf_reset_display_button,
        ):
            control.setEnabled(controls_enabled)
        self._nerf_refresh_view_button.setEnabled(
            bool(controls_enabled and self._interactive_nerf_visible and self._nerf_worker_client.is_ready)
        )
        if hasattr(self, "_nerf_scene_layer_toggle"):
            self._nerf_scene_layer_toggle.setEnabled(bool(controls_enabled and not resource_blocked))
            self._set_nerf_scene_layer_checked(self.interactive_nerf_visible)
        self._refresh_nerf_display_summary()

        if presentation != previous:
            self.nerf_visualization_changed.emit(presentation)

    def _show_nerf_preview(self) -> None:
        presentation = self._nerf_visualization_presentation
        selected = self.selected_appearance_representation
        if (
            presentation is None
            or not presentation.ready
            or not presentation.preview_path
            or selected is None
            or selected.representation_kind != "nerf_scene"
            or selected.artifact_id != presentation.artifact_id
        ):
            self._appearance_fallback_reason = "NeRF preview cannot be shown for the current selection"
            self._nerf_state_value.setText(self._appearance_fallback_reason)
            self._refresh_appearance_resource_fallback_controls()
            return
        if self._interactive_nerf_visible:
            self._hide_interactive_nerf()
        if self._interactive_gaussian_visible:
            self._hide_interactive_gaussian()
        if self._gaussian_preview_visible:
            self._hide_gaussian_preview()
        try:
            self._appearance_preview.set_image_opacity(100)
            self._set_nerf_scene_layer_checked(False)
            self._appearance_preview.set_preview(
                presentation.preview_path,
                title="NERF PREVIEW • APPEARANCE ONLY",
            )
            self._appearance_preview.set_footer(
                "Backend-authored static render • canonical geometry remains authoritative"
            )
            self._appearance_preview.set_mouse_passthrough(False)
        except (OSError, ValueError) as exc:
            self._appearance_fallback_reason = "NeRF preview image could not be displayed"
            self._nerf_state_value.setText(self._appearance_fallback_reason)
            self._nerf_state_value.setToolTip(str(exc))
            self._nerf_show_button.setEnabled(False)
            self._refresh_appearance_resource_fallback_controls()
            return
        self._appearance_fallback_reason = ""
        self._appearance_preview.show()
        self._nerf_preview_visible = True
        self._viewport_placeholder.setVisible(False)
        self._viewport_state.setText("NeRF preview • appearance only")
        self._viewport_state.setToolTip(
            "Backend-authored static NeRF render. Canonical geometry remains authoritative; no NeRF runtime is launched for Preview."
        )
        self._nerf_state_value.setText("NeRF backend preview visible • static")
        self._refresh_nerf_visualization_controls()
        self._refresh_representation_switching_controls()

    def _show_interactive_nerf(self) -> None:
        failure = self._interactive_resource_admission_failure("interactive NeRF")
        if failure:
            self.set_error(failure)
            if hasattr(self, "_nerf_state_value"):
                self._nerf_state_value.setText(failure)
                self._nerf_state_value.setToolTip(
                    "Phase 9.6G permits only one heavyweight Reconstruction/NeRF/build owner at a time."
                )
            return
        presentation = self._interactive_nerf_presentation
        selected = self.selected_appearance_representation
        if (
            presentation is None
            or not presentation.ready
            or selected is None
            or selected.representation_kind != "nerf_scene"
            or selected.artifact_id != presentation.artifact_id
        ):
            self._appearance_fallback_reason = "Interactive NeRF is unavailable for the current selection"
            self._nerf_state_value.setText(self._appearance_fallback_reason)
            self._refresh_appearance_resource_fallback_controls()
            return
        try:
            verify_interactive_nerf_integrity(self._persistent_scene, presentation)
        except (OSError, TypeError, ValueError) as exc:
            self._appearance_fallback_reason = "Interactive NeRF integrity verification failed"
            self._nerf_state_value.setText(self._appearance_fallback_reason)
            self._nerf_state_value.setToolTip(str(exc))
            self._refresh_appearance_resource_fallback_controls()
            return
        if self._interactive_gaussian_visible:
            self._hide_interactive_gaussian()
        if self._gaussian_preview_visible:
            self._hide_gaussian_preview()
        if self._nerf_preview_visible:
            self._hide_nerf_preview()

        self._interactive_nerf_visible = True
        self._nerf_preview_visible = False
        self._appearance_fallback_reason = ""
        self._appearance_preview.set_image_opacity(self._nerf_display_settings.opacity_percent)
        self._appearance_preview.set_mouse_passthrough(True)
        self._set_nerf_scene_layer_checked(True)
        if presentation.preview_path:
            try:
                self._appearance_preview.set_preview(
                    presentation.preview_path,
                    title="INTERACTIVE NERFACTO • APPEARANCE ONLY",
                )
                self._appearance_preview.set_footer(
                    "Starting isolated renderer • canonical geometry remains picking/measurement authority"
                )
                self._appearance_preview.show()
            except (OSError, ValueError):
                self._appearance_preview.hide()
        self._viewport_placeholder.setVisible(False)
        self._viewport_state.setText("Interactive Nerfacto • starting isolated renderer")

        if self._interactive_nerf_session_dir is not None:
            self._schedule_interactive_nerf_session_cleanup(self._interactive_nerf_session_dir)
        self._interactive_nerf_session_dir = Path(
            tempfile.mkdtemp(prefix="eia-phase9-5e-nerf-")
        ).resolve()
        worker = Path(__file__).resolve().parents[2] / "scripts" / "nerf_workers" / "nerfstudio_interactive_worker.py"
        try:
            self._nerf_worker_client.start(
                load_spec_path=presentation.load_spec_path,
                session_dir=self._interactive_nerf_session_dir,
                worker_script=worker,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self._interactive_nerf_visible = False
            self._set_nerf_scene_layer_checked(False)
            self._appearance_preview.hide()
            self._appearance_fallback_reason = "Interactive NeRF renderer could not start"
            self._nerf_state_value.setText(self._appearance_fallback_reason)
            self._nerf_state_value.setToolTip(str(exc))
            self._refresh_nerf_visualization_controls()
            self._refresh_representation_switching_controls()
            return
        self._refresh_digital_twin_resource_coordination()
        self._refresh_nerf_visualization_controls()
        self._refresh_representation_switching_controls()

    def _request_interactive_nerf_render(self, *, force: bool = False) -> None:
        presentation = self._interactive_nerf_presentation
        if (
            not self._interactive_nerf_visible
            or presentation is None
            or not self._nerf_worker_client.is_ready
            or self._reconstruction_canvas.interaction_mode == "measure"
            or (not force and not self._nerf_display_settings.auto_refresh)
        ):
            return
        try:
            state = self._reconstruction_canvas.interactive_render_camera_state()
            self._interactive_nerf_request_id += 1
            request = turntable_state_to_nerf_render_request(
                request_id=self._interactive_nerf_request_id,
                scene_id=presentation.scene_id,
                artifact_id=presentation.artifact_id,
                width=self._nerf_display_settings.render_width,
                height=self._nerf_display_settings.render_height,
                **state,
            )
            queue_result = self._nerf_worker_client.request_render(request.as_protocol_payload())
        except (RuntimeError, TypeError, ValueError) as exc:
            self._on_interactive_nerf_worker_failed(str(exc))
            return
        if queue_result == "coalesced":
            self._nerf_state_value.setText("Interactive Nerfacto • latest view queued (older request coalesced)")
            self._viewport_state.setText("Interactive Nerfacto • rendering latest bounded view…")
        else:
            self._nerf_state_value.setText("Interactive Nerfacto • rendering current canonical view")
            self._viewport_state.setText("Interactive Nerfacto • rendering…")
        self._refresh_nerf_runtime_health()

    def _on_interactive_nerf_worker_ready(self, payload: object) -> None:
        if not self._interactive_nerf_visible or self._interactive_nerf_presentation is None:
            self._nerf_worker_client.stop()
            return
        data = payload if isinstance(payload, dict) else {}
        self._nerf_worker_telemetry = dict(data)
        self._nerf_worker_start_count += 1
        self._refresh_nerf_runtime_health()
        if str(data.get("scene_role", "appearance_only")) != "appearance_only":
            self._on_interactive_nerf_worker_failed("Interactive NeRF worker violated appearance-only authority.")
            return
        self._nerf_state_value.setText("Interactive Nerfacto ready • requesting current view")
        self._request_interactive_nerf_render(force=True)

    def _on_interactive_nerf_frame_ready(self, payload: object) -> None:
        if not self._interactive_nerf_visible or self._interactive_nerf_presentation is None:
            return
        data = payload if isinstance(payload, dict) else {}
        request_id = int(data.get("request_id", -1))
        if request_id < self._interactive_nerf_request_id:
            return
        if str(data.get("scene_id", "")) not in {"", self._interactive_nerf_presentation.scene_id}:
            return
        path = str(data.get("frame_path", "")).strip()
        if not path:
            self._on_interactive_nerf_worker_failed("Interactive NeRF worker returned no frame path.")
            return
        try:
            self._appearance_preview.set_preview(
                path,
                title="INTERACTIVE NERFACTO • APPEARANCE ONLY",
            )
        except (OSError, ValueError) as exc:
            self._on_interactive_nerf_worker_failed(str(exc))
            return
        self._interactive_nerf_latest_frame_path = path
        self._interactive_nerf_last_render_ms = float(data.get("render_ms", 0.0) or 0.0)
        self._nerf_worker_telemetry = dict(data)
        self._appearance_preview.set_mouse_passthrough(True)
        self._refresh_digital_twin_interaction_presentation()
        interaction = self._digital_twin_interaction_presentation
        if interaction is not None and interaction.measurement_active:
            self._viewport_state.setText("Canonical measurement • NeRF appearance frozen + dimmed")
        elif interaction is not None and interaction.semantic_selection_active:
            self._viewport_state.setText("Semantic inspection • NeRF appearance dimmed")
        else:
            self._appearance_preview.set_image_opacity(self._nerf_display_settings.opacity_percent)
            self._appearance_preview.set_footer(
                f"External Nerfacto render • {self._interactive_nerf_last_render_ms:.0f} ms observed • canonical geometry remains picking/measurement authority"
            )
            self._appearance_preview.show()
            self._viewport_state.setText("Interactive Nerfacto • appearance only")
        self._viewport_state.setToolTip(
            "The displayed image comes from the isolated Nerfstudio renderer. Navigation and ruler picking remain bound to the canonical VisPy scene underneath."
        )
        self._nerf_state_value.setText(
            f"Interactive Nerfacto ready • last render {self._interactive_nerf_last_render_ms:.0f} ms"
        )
        self._refresh_nerf_runtime_health()
        self._refresh_representation_switching_controls()

    def _on_interactive_nerf_worker_failed(self, message: str) -> None:
        text = str(message).strip() or "Interactive NeRF renderer failed"
        self._appearance_fallback_reason = text
        self._appearance_preview.hide()
        self._nerf_state_value.setText("Interactive NeRF unavailable • canonical view retained")
        self._nerf_state_value.setToolTip(text)
        self._viewport_state.setText("Canonical scene • interactive NeRF unavailable")
        self._refresh_nerf_runtime_health()
        self._refresh_representation_switching_controls()

    def _on_interactive_nerf_worker_state_changed(self, state: str) -> None:
        normalized = str(state).strip().lower()
        if normalized == "stopped" and self._nerf_release_pending:
            self._nerf_worker_release_count += 1
            self._nerf_release_pending = False
            self._nerf_worker_telemetry = {}
        if self._interactive_nerf_visible and hasattr(self, "_nerf_state_value"):
            self._refresh_nerf_visualization_controls()
        self._refresh_nerf_runtime_health()
        self._refresh_digital_twin_resource_coordination()

    def _hide_nerf_preview(self) -> None:
        if not hasattr(self, "_appearance_preview"):
            return
        if self._nerf_preview_visible:
            self._appearance_preview.hide()
        self._nerf_preview_visible = False
        self._appearance_preview.set_mouse_passthrough(False)
        self._refresh_viewport_state()

    def _hide_interactive_nerf(self, *, blocking: bool = False) -> None:
        if not self._interactive_nerf_visible and not self._nerf_worker_client.is_running:
            return
        self._interactive_nerf_visible = False
        self._set_nerf_scene_layer_checked(False)
        self._appearance_preview.hide()
        self._appearance_preview.set_mouse_passthrough(False)
        if self._nerf_worker_client.is_running:
            self._nerf_release_pending = True
        if blocking:
            self._nerf_worker_client.stop_blocking()
        else:
            self._nerf_worker_client.stop()
        session = self._interactive_nerf_session_dir
        self._interactive_nerf_session_dir = None
        if session is not None:
            if blocking:
                self._cleanup_interactive_nerf_session_dir(session)
            else:
                self._schedule_interactive_nerf_session_cleanup(session)
        self._refresh_viewport_state()
        self._refresh_nerf_runtime_health()
        self._refresh_digital_twin_resource_coordination()

    def _hide_nerf_visualization(self, *, refresh: bool = True) -> None:
        if self._interactive_nerf_visible or self._nerf_worker_client.is_running:
            self._hide_interactive_nerf()
        if self._nerf_preview_visible:
            self._hide_nerf_preview()
        if refresh:
            self._refresh_nerf_visualization_controls()
            self._refresh_representation_switching_controls()

    @staticmethod
    def _cleanup_interactive_nerf_session_dir(path: Path) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass

    def _schedule_interactive_nerf_session_cleanup(self, path: Path) -> None:
        QTimer.singleShot(3000, lambda p=Path(path): self._cleanup_interactive_nerf_session_dir(p))

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("reconstructionSettingsPanel")
        panel.setMinimumWidth(290)
        panel.setMaximumWidth(380)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)

        title = QLabel("RECONSTRUCTION SETTINGS")
        title.setObjectName("reconstructionSectionTitle")
        panel_layout.addWidget(title)

        # D4 makes the offline inspection stack taller than the available
        # sidebar height on common 768p/900p/1080p desktops.  Do not let Qt
        # satisfy that pressure by crushing rows into each other.  Keep the
        # panel width fixed and make only the settings body vertically
        # scrollable.  The title remains pinned and no horizontal scrolling
        # is permitted.
        self._settings_scroll = QScrollArea(panel)
        self._settings_scroll.setObjectName("reconstructionSettingsScroll")
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._settings_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._settings_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._settings_scroll.viewport().setAutoFillBackground(False)

        settings_body = QWidget()
        settings_body.setObjectName("reconstructionSettingsScrollBody")
        settings_body.setAutoFillBackground(False)
        layout = QVBoxLayout(settings_body)
        # Some Linux/Qt styles draw the vertical scrollbar over the scroll
        # viewport instead of reserving its full visual width.  Keep a small
        # right-side gutter so right-aligned settings values wrap before the
        # scrollbar and never render underneath it.
        layout.setContentsMargins(0, 0, 14, 0)
        layout.setSpacing(10)

        workflow = self._settings_group("WORKFLOW")
        self._workflow_group = workflow
        workflow_layout = workflow.content_layout
        self._workflow_input_value = self._settings_value("No source selected")
        self._runtime_prerequisite_value = self._settings_value("Choose a workflow")
        self._runtime_output_value = self._settings_value("Not started")
        self._add_status_row(workflow_layout, "Input", self._workflow_input_value)
        self._add_status_row(workflow_layout, "Runtime", self._runtime_prerequisite_value)
        self._add_status_row(workflow_layout, "Output", self._runtime_output_value)
        self._add_settings_row(workflow_layout, "Profile", "Efficient")
        self._add_settings_row(workflow_layout, "Geometry", "Canonical authority")
        self._add_settings_row(workflow_layout, "Appearance", "Deferred")
        layout.addWidget(workflow)

        session = self._settings_group("SESSION & PERSISTENCE")
        self._session_group = session
        session_layout = session.content_layout
        self._session_value = self._settings_value("No active session")
        self._session_context_value = self._settings_value("No session context")
        self._session_evidence_value = self._settings_value("Runtime/session metadata only")
        self._checkpoint_value = self._settings_value("No checkpoint")
        self._scene_manifest_value = self._settings_value("No persistent scene")
        self._telemetry_value = self._settings_value("No runtime telemetry")
        self._add_status_row(session_layout, "Session", self._session_value)
        self._add_status_row(session_layout, "Context", self._session_context_value)
        self._add_status_row(session_layout, "Evidence", self._session_evidence_value)
        self._add_status_row(session_layout, "Scene", self._scene_manifest_value)
        self._add_status_row(session_layout, "Checkpoint", self._checkpoint_value)
        self._add_status_row(session_layout, "Telemetry", self._telemetry_value)
        self._session_action_state_value = self._settings_value("Load persistent scene")
        self._add_status_row(session_layout, "Actions", self._session_action_state_value)
        session_actions = QHBoxLayout()
        session_actions.setSpacing(6)
        self._session_load_button = QPushButton("Load Scene")
        self._session_load_button.setObjectName("reconstructionRuntimeAction")
        self._session_load_button.setToolTip(
            "Load a persistent reconstruction scene manifest with checksum and referenced-file "
            "verification. A matching Phase 9.6F workspace sidecar is restored when present. "
            "This does not start reconstruction or resume a checkpoint."
        )
        self._session_load_button.clicked.connect(self._load_persistent_scene_interaction)
        self._session_load_button.setEnabled(False)
        session_actions.addWidget(self._session_load_button)
        self._session_save_button = QPushButton("Save Scene")
        self._session_save_button.setObjectName("reconstructionRuntimeAction")
        self._session_save_button.setToolTip(
            "Write the currently loaded persistent scene manifest and checksum to a selected "
            "folder. Phase 9.6F also writes scene-bound workspace/ruler/build-history state. "
            "Referenced artifact payloads remain at their recorded paths."
        )
        self._session_save_button.clicked.connect(self._save_persistent_scene_interaction)
        self._session_save_button.setEnabled(False)
        session_actions.addWidget(self._session_save_button)
        self._session_reload_button = QPushButton("Reload")
        self._session_reload_button.setObjectName("reconstructionRuntimeAction")
        self._session_reload_button.setToolTip(
            "Re-read the active persistent scene from its current manifest, verify integrity, "
            "require the same session identity, and restore a matching Phase 9.6F workspace sidecar. "
            "No checkpoint resume is executed, and no heavyweight build resume is executed."
        )
        self._session_reload_button.clicked.connect(self._reload_persistent_scene_interaction)
        self._session_reload_button.setEnabled(False)
        for button in (
            self._session_load_button,
            self._session_save_button,
            self._session_reload_button,
        ):
            # G6 responsive-sidebar hardening: session action labels must not
            # establish a minimum width wider than the Reconstruction viewport.
            # This changes only layout negotiation; action semantics stay intact.
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )
        session_actions.addWidget(self._session_reload_button)
        session_layout.addLayout(session_actions)

        # Phase 9.6F persists only scene-bound workspace presentation state in a
        # checksum-protected companion sidecar. Heavy payloads remain owned by
        # the existing persistent scene; no build job is resumed from this state.
        self._digital_twin_workspace_state_value = self._settings_value(
            "No Digital Twin workspace sidecar"
        )
        self._add_status_row(
            session_layout, "Twin state", self._digital_twin_workspace_state_value
        )
        twin_state_actions = QHBoxLayout()
        twin_state_actions.setSpacing(6)
        self._digital_twin_state_export_button = QPushButton("Export State")
        self._digital_twin_state_export_button.setObjectName("reconstructionRuntimeAction")
        self._digital_twin_state_export_button.setToolTip(
            "Export the scene/authority/workspace manifests for audit and same-environment restore. "
            "Heavy referenced payloads are not embedded; use Product export for materialized geometry/QC evidence."
        )
        self._digital_twin_state_export_button.clicked.connect(
            self._export_digital_twin_state_interaction
        )
        self._digital_twin_state_export_button.setEnabled(False)
        twin_state_actions.addWidget(self._digital_twin_state_export_button)
        self._digital_twin_state_export_open_button = QPushButton("Open State Export")
        self._digital_twin_state_export_open_button.setObjectName("reconstructionRuntimeAction")
        self._digital_twin_state_export_open_button.setToolTip(
            "Open the most recent Digital Twin state-export folder from this application session."
        )
        self._digital_twin_state_export_open_button.clicked.connect(
            self._open_digital_twin_state_export_interaction
        )
        self._digital_twin_state_export_open_button.setEnabled(False)
        twin_state_actions.addWidget(self._digital_twin_state_export_open_button)
        for button in (
            self._digital_twin_state_export_button,
            self._digital_twin_state_export_open_button,
        ):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        session_layout.addLayout(twin_state_actions)

        # Phase 8.9A2 keeps product export visibly separate from persistent-scene
        # save/reload while preserving the locked Phase 8.8 sidebar information
        # architecture. Export only packages already-materialized A1-qualified
        # reconstruction evidence; it never reconstructs, persists, or promotes it.
        self._reconstruction_export_state_value = self._settings_value(
            "No materialized export evidence"
        )
        self._add_status_row(
            session_layout, "Product export", self._reconstruction_export_state_value
        )
        export_actions = QHBoxLayout()
        export_actions.setSpacing(6)
        self._reconstruction_export_button = QPushButton("Export Bundle")
        self._reconstruction_export_button.setObjectName("reconstructionRuntimeAction")
        self._reconstruction_export_button.setToolTip(
            "Export the currently materialized, session-matched reconstruction evidence through "
            "the authority-aware Phase 8.9 contract. This is separate from Save Scene and does "
            "not rerun reconstruction, copy a persistent-scene manifest, or promote appearance assets."
        )
        self._reconstruction_export_button.clicked.connect(
            self._export_reconstruction_bundle_interaction
        )
        self._reconstruction_export_button.setEnabled(False)
        export_actions.addWidget(self._reconstruction_export_button)
        self._reconstruction_export_open_button = QPushButton("Open Export")
        self._reconstruction_export_open_button.setObjectName("reconstructionRuntimeAction")
        self._reconstruction_export_open_button.setToolTip(
            "Open the most recent reconstruction product-export folder created in this application session."
        )
        self._reconstruction_export_open_button.clicked.connect(
            self._open_reconstruction_export_interaction
        )
        self._reconstruction_export_open_button.setEnabled(False)
        export_actions.addWidget(self._reconstruction_export_open_button)
        for button in (
            self._reconstruction_export_button,
            self._reconstruction_export_open_button,
        ):
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )
        session_layout.addLayout(export_actions)

        self._session_resume_state_value = self._settings_value("Load reconstruction checkpoint")
        self._add_status_row(session_layout, "Resume", self._session_resume_state_value)
        resume_actions = QHBoxLayout()
        resume_actions.setSpacing(6)
        self._session_checkpoint_load_button = QPushButton("Load Checkpoint")
        self._session_checkpoint_load_button.setObjectName("reconstructionRuntimeAction")
        self._session_checkpoint_load_button.setToolTip(
            "Load the existing whole-engine reconstruction checkpoint manifest with SHA-256 "
            "integrity verification. Loading alone does not start or resume reconstruction."
        )
        self._session_checkpoint_load_button.clicked.connect(self._load_engine_checkpoint_interaction)
        self._session_checkpoint_load_button.setEnabled(False)
        resume_actions.addWidget(self._session_checkpoint_load_button)
        self._session_resume_button = QPushButton("Prepare Resume")
        self._session_resume_button.setObjectName("reconstructionRuntimeAction")
        self._session_resume_button.setToolTip(
            "Prepare resume only after the current pipeline plan, providers, and payload references "
            "pass the existing whole-engine compatibility contract. G3 does not launch the resumed engine."
        )
        self._session_resume_button.clicked.connect(self._prepare_engine_resume_interaction)
        self._session_resume_button.setEnabled(False)
        for button in (
            self._session_checkpoint_load_button,
            self._session_resume_button,
        ):
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )
        resume_actions.addWidget(self._session_resume_button)
        session_layout.addLayout(resume_actions)

        self._session_recovery_state_value = self._settings_value("No recovery action required")
        self._add_status_row(session_layout, "Recovery", self._session_recovery_state_value)
        recovery_actions = QHBoxLayout()
        recovery_actions.setSpacing(6)
        self._session_retry_button = QPushButton("Retry From Source")
        self._session_retry_button.setObjectName("reconstructionRuntimeAction")
        self._session_retry_button.setToolTip(
            "Start a fresh reconstruction attempt from the currently selected source after a terminal "
            "failure/cancellation. This creates a new session; it is not checkpoint resume or manual "
            "worker restart."
        )
        self._session_retry_button.clicked.connect(self._retry_failed_runtime_interaction)
        self._session_retry_button.setEnabled(False)
        self._session_retry_button.setMinimumWidth(0)
        self._session_retry_button.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        recovery_actions.addWidget(self._session_retry_button)
        recovery_actions.addStretch(1)
        session_layout.addLayout(recovery_actions)
        layout.addWidget(session)

        # Phase 9.6A adds one compact composition/authority summary over the
        # existing qualified Reconstruction presentation paths. It does not
        # replace, duplicate, or hide the detailed canonical/semantic/measurement/
        # QC/Gaussian/NeRF controls below; later 9.6E owns any UX consolidation.
        self._digital_twin_workspace_group = self._settings_group("DIGITAL TWIN WORKSPACE")
        workspace_layout = self._digital_twin_workspace_group.content_layout
        self._digital_twin_workspace_value = self._settings_value("No persistent Digital Twin scene")
        self._digital_twin_active_view_value = self._settings_value("Canonical geometry")
        self._digital_twin_composition_value = self._settings_value("Waiting for persistent scene")
        self._digital_twin_navigation_value = self._settings_value("Waiting for persistent scene")
        self._digital_twin_interaction_value = self._settings_value("Waiting for canonical interaction")
        self._digital_twin_scale_value = self._settings_value("—")
        self._digital_twin_authority_value = self._settings_value("No canonical scene authority")
        self._digital_twin_resource_value = self._settings_value("No neural appearance resource required")
        self._digital_twin_safety_value = self._settings_value(
            "Appearance only overlays • QC/reference separate • metrology not certified"
        )
        # E4 keeps the decision-critical workspace state visible at a glance and
        # moves supporting navigation/resource diagnostics behind one nested
        # disclosure. The existing value widgets remain live and unchanged.
        self._add_status_row(workspace_layout, "Workspace", self._digital_twin_workspace_value)
        self._add_status_row(workspace_layout, "Active view", self._digital_twin_active_view_value)
        self._add_status_row(workspace_layout, "Composition", self._digital_twin_composition_value)
        self._add_status_row(workspace_layout, "Scale", self._digital_twin_scale_value)
        self._add_status_row(workspace_layout, "Authority", self._digital_twin_authority_value)

        self._digital_twin_workspace_details_button = QPushButton("Show details")
        self._digital_twin_workspace_details_button.setObjectName("reconstructionRuntimeAction")
        self._digital_twin_workspace_details_button.setToolTip(
            "Show navigation, interaction, neural-resource and safety details for the unified workspace."
        )
        self._digital_twin_workspace_details_button.setMinimumWidth(0)
        self._digital_twin_workspace_details_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._digital_twin_workspace_details_button.clicked.connect(
            self._toggle_digital_twin_workspace_details
        )
        workspace_layout.addWidget(self._digital_twin_workspace_details_button)

        self._digital_twin_workspace_details = QWidget()
        self._digital_twin_workspace_details.setObjectName("reconstructionDigitalTwinWorkspaceDetails")
        workspace_details_layout = QVBoxLayout(self._digital_twin_workspace_details)
        workspace_details_layout.setContentsMargins(0, 0, 0, 0)
        workspace_details_layout.setSpacing(5)
        self._add_status_row(
            workspace_details_layout, "Navigation", self._digital_twin_navigation_value
        )
        self._add_status_row(
            workspace_details_layout, "Interaction", self._digital_twin_interaction_value
        )
        self._add_status_row(
            workspace_details_layout, "Resources", self._digital_twin_resource_value
        )
        self._add_status_row(
            workspace_details_layout, "Safety", self._digital_twin_safety_value
        )
        self._digital_twin_workspace_details.hide()
        workspace_layout.addWidget(self._digital_twin_workspace_details)
        self._digital_twin_authority_value.setToolTip(
            "Canonical geometry remains the only spatial, picking and measurement authority. "
            "Semantics are context only; Gaussian/NeRF are appearance only; QC/reference stays separate."
        )
        self._digital_twin_safety_value.setToolTip(
            "Phase 9.6A composes existing authority state only. It cannot create metric scale, "
            "promote appearance, infer QC decisions, or certify metrology."
        )
        layout.addWidget(self._digital_twin_workspace_group)
        self._refresh_digital_twin_workspace_presentation()

        # Phase 9.6E1 established the generic single-active-job surface; E2/E3
        # populated it with DA3 and NeRF build actions. Phase 9.6G now adds the
        # shared dynamic resource owner/GPU policy rows without moving heavy
        # runtime imports into ``depth`` or changing canonical authority.
        self._digital_twin_processing_group = self._settings_group("BUILD & PROCESSING")
        self._digital_twin_processing_group.set_expanded(False)
        processing_layout = self._digital_twin_processing_group.content_layout
        processing_hint = QLabel(
            "One isolated build job at a time • external runtimes only • canonical workspace stays authoritative."
        )
        processing_hint.setObjectName("reconstructionSettingsHint")
        processing_hint.setWordWrap(True)
        processing_layout.addWidget(processing_hint)
        self._digital_twin_processing_job_value = self._settings_value("No Digital Twin build job")
        self._digital_twin_processing_state_value = self._settings_value("Idle")
        self._digital_twin_processing_stage_value = self._settings_value("Waiting for build request")
        self._digital_twin_processing_backend_value = self._settings_value("External runtime not active")
        self._digital_twin_processing_resource_value = self._settings_value(
            "Available • no dynamic Digital Twin owner"
        )
        self._digital_twin_processing_gpu_value = self._settings_value("Startup resource snapshot unavailable")
        self._digital_twin_processing_output_value = self._settings_value("—")
        self._digital_twin_processing_authority_value = self._settings_value(
            "Canonical geometry remains authoritative"
        )
        self._digital_twin_processing_history_value = self._settings_value(
            "No persisted build history"
        )
        self._add_status_row(processing_layout, "Job", self._digital_twin_processing_job_value)
        self._add_status_row(processing_layout, "State", self._digital_twin_processing_state_value)
        self._add_status_row(processing_layout, "Stage", self._digital_twin_processing_stage_value)
        self._add_status_row(processing_layout, "Backend", self._digital_twin_processing_backend_value)
        self._add_status_row(processing_layout, "Resource owner", self._digital_twin_processing_resource_value)
        self._add_status_row(processing_layout, "GPU snapshot", self._digital_twin_processing_gpu_value)
        self._add_status_row(processing_layout, "Output", self._digital_twin_processing_output_value)
        self._add_status_row(processing_layout, "Authority", self._digital_twin_processing_authority_value)
        self._add_status_row(processing_layout, "Last build", self._digital_twin_processing_history_value)
        self._digital_twin_processing_authority_value.setToolTip(
            "Processing jobs cannot promote themselves to canonical geometry, metric-scale, measurement, QC, or certified-metrology authority."
        )
        processing_progress_row = QHBoxLayout()
        processing_progress_row.setSpacing(8)
        self._digital_twin_processing_progress = QProgressBar()
        self._digital_twin_processing_progress.setObjectName("reconstructionDigitalTwinProcessingProgress")
        self._digital_twin_processing_progress.setRange(0, 100)
        self._digital_twin_processing_progress.setValue(0)
        self._digital_twin_processing_progress.setFormat("Idle")
        self._digital_twin_processing_progress.setTextVisible(True)
        processing_progress_row.addWidget(self._digital_twin_processing_progress, 1)
        self._digital_twin_processing_elapsed_value = QLabel("—")
        self._digital_twin_processing_elapsed_value.setObjectName("reconstructionRuntimeElapsed")
        processing_progress_row.addWidget(
            self._digital_twin_processing_elapsed_value,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        processing_layout.addLayout(processing_progress_row)

        # Phase 9.6E2 exposes one deliberately simple full-scene DA3 action.
        # The existing Advanced/Integrate DA3 path remains the qualified Quick
        # workflow; this action is recorded-video only and runs through E1.
        self._da3_full_scene_value = self._settings_value(
            "Select a ready Recorded video for full-scene dense geometry"
        )
        self._add_status_row(processing_layout, "Dense geometry", self._da3_full_scene_value)
        self._da3_full_scene_build_button = QPushButton("Build Full Scene")
        self._da3_full_scene_build_button.setObjectName("reconstructionRuntimeAction")
        self._da3_full_scene_build_button.setToolTip(
            "Select quality-filtered keyframes across the complete recorded video and process them in bounded external DA3 windows. Scale remains arbitrary until separately qualified."
        )
        self._da3_full_scene_build_button.clicked.connect(self._start_da3_full_scene_build)
        self._da3_full_scene_build_button.setMinimumWidth(0)
        self._da3_full_scene_build_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._nerf_build_value = self._settings_value("Load/build a persistent canonical scene first")
        self._add_status_row(processing_layout, "NeRF appearance", self._nerf_build_value)
        nerf_build_profile_row = QHBoxLayout()
        nerf_build_profile_row.setSpacing(8)
        nerf_build_profile_label = QLabel("Training profile")
        nerf_build_profile_label.setObjectName("reconstructionRowLabel")
        nerf_build_profile_row.addWidget(nerf_build_profile_label)
        self._nerf_build_profile_combo = QComboBox()
        self._nerf_build_profile_combo.setObjectName("reconstructionNerfBuildProfileCombo")
        for preset in NERF_BUILD_PRESETS:
            self._nerf_build_profile_combo.addItem(preset.label, preset.key)
        default_index = self._nerf_build_profile_combo.findData(NERF_DEFAULT_BUILD_PRESET)
        self._nerf_build_profile_combo.setCurrentIndex(max(0, default_index))
        self._nerf_build_profile_combo.setToolTip(
            "Bounded Phase 9.5-qualified Nerfacto training profiles. These steps prove/train appearance but do not certify production visual quality."
        )
        nerf_build_profile_row.addWidget(self._nerf_build_profile_combo, 1)
        processing_layout.addLayout(nerf_build_profile_row)
        self._nerf_build_button = QPushButton("Build & Train NeRF")
        self._nerf_build_button.setObjectName("reconstructionRuntimeAction")
        self._nerf_build_button.setToolTip(
            "Build exact canonical RGB/camera input, train Nerfacto in the isolated eia-nerf-nerfstudio runtime, and persist appearance as a new immutable scene revision. NeRF never becomes geometry or scale authority."
        )
        self._nerf_build_button.clicked.connect(self._start_nerf_build)
        self._nerf_build_button.setMinimumWidth(0)
        self._nerf_build_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        processing_layout.addWidget(self._nerf_build_button)

        processing_actions = QHBoxLayout()
        processing_actions.setSpacing(6)
        self._digital_twin_processing_cancel_button = QPushButton("Cancel build")
        self._digital_twin_processing_cancel_button.setObjectName("reconstructionRuntimeAction")
        self._digital_twin_processing_cancel_button.setToolTip(
            "Cancel the active isolated build job with bounded SIGTERM → SIGKILL cleanup. Canonical scene state is not discarded."
        )
        self._digital_twin_processing_cancel_button.clicked.connect(
            self.cancel_digital_twin_processing_job
        )
        self._digital_twin_processing_cancel_button.setEnabled(False)
        self._digital_twin_processing_cancel_button.setMinimumWidth(0)
        self._digital_twin_processing_cancel_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        # Keep both primary build actions visible at the qualified 1366×768
        # workspace width. Horizontal Ignored policies plus an intervening
        # stretch can legally collapse both buttons to zero width in the
        # narrow settings sidebar, which makes users fall back to the unrelated
        # top-level Offline Reconstruction Start action.
        processing_actions.addWidget(self._da3_full_scene_build_button, 1)
        processing_actions.addWidget(self._digital_twin_processing_cancel_button, 1)
        processing_layout.addLayout(processing_actions)
        layout.addWidget(self._digital_twin_processing_group)
        self._refresh_digital_twin_processing_presentation()
        self._refresh_digital_twin_resource_coordination()
        self._refresh_nerf_build_controls()

        # Phase 9.3A composes the existing persistent canonical-geometry contract
        # and Phase 9.1H authority sidecar into one passive scene summary.  It
        # does not parse mesh payloads or render canonical geometry; that remains
        # the explicit Phase 9.3B boundary.
        self._canonical_scene_group = self._settings_group("CANONICAL SCENE")
        canonical_layout = self._canonical_scene_group.content_layout
        self._canonical_scene_state_value = self._settings_value("No persistent scene")
        self._canonical_scene_geometry_value = self._settings_value("—")
        self._canonical_scene_provider_value = self._settings_value("—")
        self._canonical_scene_resource_value = self._settings_value("—")
        self._canonical_scene_world_value = self._settings_value("No canonical scene")
        self._canonical_scene_cameras_value = self._settings_value("No persistent camera trajectory")
        self._canonical_scene_scale_value = self._settings_value("—")
        self._canonical_scene_authority_value = self._settings_value("No canonical scene authority")
        self._canonical_scene_status_value = self._settings_value(
            "Waiting for persistent reconstruction scene"
        )
        self._canonical_scene_authority_value.setMinimumHeight(
            self._canonical_scene_authority_value.fontMetrics().lineSpacing() * 2
        )
        self._add_status_row(canonical_layout, "Scene", self._canonical_scene_state_value)
        self._add_status_row(canonical_layout, "Geometry", self._canonical_scene_geometry_value)
        self._add_status_row(canonical_layout, "Provider", self._canonical_scene_provider_value)
        self._add_status_row(canonical_layout, "Resource", self._canonical_scene_resource_value)
        self._add_status_row(canonical_layout, "World", self._canonical_scene_world_value)
        self._add_status_row(canonical_layout, "Cameras", self._canonical_scene_cameras_value)
        self._add_status_row(canonical_layout, "Scale", self._canonical_scene_scale_value)
        self._add_status_row(canonical_layout, "Authority", self._canonical_scene_authority_value)
        self._add_status_row(canonical_layout, "State", self._canonical_scene_status_value)
        layout.addWidget(self._canonical_scene_group)
        self._refresh_canonical_scene_presentation()

        # Phase 9.6E0 keeps estimated ruler scale explicitly separate from the
        # canonical scene's metric authority. One approximate physical ruler
        # reference may provide a useful metre estimate with explicit uncertainty,
        # but it never changes unit_to_metre authority or certified-metrology state.
        self._estimated_ruler_scale_group = self._settings_group("RULER METRIC ESTIMATE")
        self._estimated_ruler_scale_group.set_expanded(False)
        estimated_scale_layout = self._estimated_ruler_scale_group.content_layout
        self._estimated_ruler_scale_state_value = self._settings_value("No estimated scale")
        self._estimated_ruler_scale_value = self._settings_value("—")
        self._estimated_ruler_reference_value = self._settings_value("Measure two canonical points first")
        self._estimated_ruler_scale_authority_value = self._settings_value(
            "Estimate only • canonical scale authority unchanged"
        )
        self._add_status_row(estimated_scale_layout, "State", self._estimated_ruler_scale_state_value)
        self._add_status_row(estimated_scale_layout, "Scale ±", self._estimated_ruler_scale_value)
        self._add_status_row(estimated_scale_layout, "Reference", self._estimated_ruler_reference_value)
        self._add_status_row(estimated_scale_layout, "Authority", self._estimated_ruler_scale_authority_value)
        self._estimated_ruler_scale_authority_value.setToolTip(
            "This session-side estimate never writes canonical unit_to_metre, never qualifies scene scale, "
            "and never certifies metrology. Phase 9.6F owns persistence/restore."
        )
        estimated_scale_actions = QHBoxLayout()
        estimated_scale_actions.setSpacing(6)
        self._estimated_ruler_scale_set_button = QPushButton("Set from last ruler")
        self._estimated_ruler_scale_set_button.setObjectName("reconstructionRuntimeAction")
        self._estimated_ruler_scale_set_button.setToolTip(
            "Use the last canonical ruler segment plus an approximate physical distance and ± uncertainty "
            "to create a scene-bound uncalibrated metre estimate."
        )
        self._estimated_ruler_scale_set_button.clicked.connect(
            self._set_estimated_metric_scale_from_last_ruler_interaction
        )
        self._estimated_ruler_scale_clear_button = QPushButton("Clear estimate")
        self._estimated_ruler_scale_clear_button.setObjectName("reconstructionRuntimeAction")
        self._estimated_ruler_scale_clear_button.setToolTip(
            "Clear only the session-side estimated scale. Canonical geometry and persisted metric authority are unchanged."
        )
        self._estimated_ruler_scale_clear_button.clicked.connect(self.clear_estimated_metric_scale_hint)
        for button in (self._estimated_ruler_scale_set_button, self._estimated_ruler_scale_clear_button):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            estimated_scale_actions.addWidget(button)
        estimated_scale_layout.addLayout(estimated_scale_actions)
        layout.addWidget(self._estimated_ruler_scale_group)
        self._refresh_estimated_metric_scale_presentation()

        # Phase 9.3F keeps the dense quick toolbar intact for backward
        # compatibility while exposing the same scene resources as a readable,
        # progressively-disclosed layer stack. Existing toolbar gates are
        # mirrored; newer canonical sublayers remain independent presentation
        # preferences. None of these switches changes evidence or authority.
        self._scene_layers_group = self._settings_group("SCENE LAYERS")
        layer_layout = self._scene_layers_group.content_layout
        layer_hint = QLabel(
            "Visibility only • geometry, scale, measurement and QC authority stay unchanged."
        )
        layer_hint.setObjectName("reconstructionSettingsHint")
        layer_hint.setWordWrap(True)
        layer_layout.addWidget(layer_hint)

        self._add_scene_layer_toggle(
            layer_layout,
            "canonical_surface",
            "Canonical surface",
            "Show or hide the already-materialized authoritative canonical surface. Hiding it never unloads or demotes the canonical artifact.",
            callback=self._reconstruction_canvas.set_canonical_mesh_visible,
        )
        gaussian_layer_row = QHBoxLayout()
        gaussian_layer_row.setSpacing(8)
        gaussian_layer_label = QLabel("Gaussian appearance")
        gaussian_layer_label.setObjectName("reconstructionRowLabel")
        gaussian_layer_label.setMinimumWidth(0)
        gaussian_layer_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        gaussian_layer_label.setToolTip(
            "Show or hide the persisted interactive Gaussian appearance layer. "
            "Canonical geometry stays loaded underneath and remains the only picking/measurement authority."
        )
        gaussian_layer_row.addWidget(gaussian_layer_label, 1)
        self._gaussian_scene_layer_toggle = ToggleSwitch(checked=False, width=32, height=18)
        self._gaussian_scene_layer_toggle.setObjectName("reconstructionGaussianSceneLayerToggle")
        self._gaussian_scene_layer_toggle.setAccessibleName("Gaussian appearance scene layer")
        self._gaussian_scene_layer_toggle.setToolTip(gaussian_layer_label.toolTip())
        self._gaussian_scene_layer_toggle.setEnabled(False)
        self._gaussian_scene_layer_toggle.toggled.connect(self._on_gaussian_scene_layer_toggled)
        gaussian_layer_row.addWidget(
            self._gaussian_scene_layer_toggle,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        layer_layout.addLayout(gaussian_layer_row)
        nerf_layer_row = QHBoxLayout()
        nerf_layer_row.setSpacing(8)
        nerf_layer_label = QLabel("NeRF appearance")
        nerf_layer_label.setObjectName("reconstructionRowLabel")
        nerf_layer_label.setMinimumWidth(0)
        nerf_layer_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        nerf_layer_label.setToolTip(
            "Show or hide the persisted interactive NeRF appearance layer. Canonical geometry remains the only picking/measurement authority."
        )
        nerf_layer_row.addWidget(nerf_layer_label, 1)
        self._nerf_scene_layer_toggle = ToggleSwitch(checked=False, width=32, height=18)
        self._nerf_scene_layer_toggle.setObjectName("reconstructionNerfSceneLayerToggle")
        self._nerf_scene_layer_toggle.setAccessibleName("NeRF appearance scene layer")
        self._nerf_scene_layer_toggle.setToolTip(nerf_layer_label.toolTip())
        self._nerf_scene_layer_toggle.setEnabled(False)
        self._nerf_scene_layer_toggle.toggled.connect(self._on_nerf_scene_layer_toggled)
        nerf_layer_row.addWidget(
            self._nerf_scene_layer_toggle,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        layer_layout.addLayout(nerf_layer_row)
        self._add_scene_layer_toggle(
            layer_layout,
            "grid",
            "World grid",
            "Mirror the viewport world-grid quick toggle.",
            toolbar_key="grid",
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "axes",
            "World axes",
            "Show or hide the viewer Z-up world-origin axes. This does not establish scale or georeferencing.",
            callback=self._reconstruction_canvas.set_axes_visible,
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "camera_frustums",
            "Camera frustums",
            "Mirror the existing camera/keyframe frustum quick toggle, including persisted canonical camera context.",
            toolbar_key="keyframes",
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "camera_trajectory",
            "Camera trajectory",
            "Mirror the existing camera-trajectory quick toggle, including persisted canonical trajectory context.",
            toolbar_key="trajectory",
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "sparse_landmarks",
            "Sparse landmarks",
            "Mirror the existing live/offline sparse landmark quick toggle.",
            toolbar_key="landmarks",
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "sparse_envelope",
            "Sparse envelope",
            "Mirror the existing offline sparse presentation-envelope quick toggle.",
            toolbar_key="offline_envelope",
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "semantic_objects",
            "Semantic support",
            "Master semantic-context visibility. Support boxes/centroids remain non-measurement presentation geometry.",
            toolbar_key="semantic_objects",
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "semantic_labels",
            "Object labels",
            "Show or hide persisted canonical semantic labels without changing semantic membership.",
            callback=self._reconstruction_canvas.set_canonical_semantic_labels_visible,
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "semantic_selection",
            "Selection highlight",
            "Show or hide the selected object's persisted canonical-surface highlight.",
            callback=self._reconstruction_canvas.set_canonical_semantic_selection_visible,
        )
        self._add_scene_layer_toggle(
            layer_layout,
            "inspection_evidence",
            "Inspection evidence",
            "Show or hide the already-qualified 9.3E measurement/QC callout. Evidence validity and rejection state are unchanged.",
            callback=self._reconstruction_canvas.set_canonical_inspection_overlay_visible,
        )
        layout.addWidget(self._scene_layers_group)
        self._refresh_scene_layer_controls()

        # Phase 9.3I exposes bounded UI-side bind/resource diagnostics only.
        # These values measure presentation work already performed in this process;
        # they never probe a backend or change geometry/metric authority.
        self._scene_health_group = self._settings_group("SCENE HEALTH")
        self._scene_health_group.set_expanded(False)
        health_layout = self._scene_health_group.content_layout
        health_hint = QLabel(
            "UI bind/resource diagnostics only • no backend benchmark or authority promotion."
        )
        health_hint.setObjectName("reconstructionSettingsHint")
        health_hint.setWordWrap(True)
        health_layout.addWidget(health_hint)
        self._scene_health_bind_value = self._settings_value("No persistent scene")
        self._scene_health_resources_value = self._settings_value("No canonical viewer resources")
        self._scene_health_lifecycle_value = self._settings_value("Idle • no scene-bound resources")
        self._scene_health_authority_value = self._settings_value(
            "Diagnostics only • no scene authority"
        )
        self._add_status_row(health_layout, "Bind", self._scene_health_bind_value)
        self._add_status_row(health_layout, "Resources", self._scene_health_resources_value)
        self._add_status_row(health_layout, "Lifecycle", self._scene_health_lifecycle_value)
        self._add_status_row(health_layout, "Authority", self._scene_health_authority_value)
        layout.addWidget(self._scene_health_group)
        self._refresh_scene_runtime_health()

        # Phase 9.3G established the read-only provider inspector. Phase 9.3H
        # adds an explicit source-driven DA3 integration action while preserving
        # the same authority boundary: provider selection remains inspection-only,
        # DA3 executes in its isolated runtime, and only the downstream canonical
        # dense/surface providers receive canonical geometry authority.
        self._advanced_reconstruction_group = self._settings_group("ADVANCED RECONSTRUCTION")
        self._advanced_reconstruction_group.set_expanded(False)
        advanced_layout = self._advanced_reconstruction_group.content_layout
        provider_hint = QLabel(
            "Provider selection is inspection-only. Integrate DA3 explicitly from the selected image/video source; calibration is optional and no metric scale is invented."
        )
        provider_hint.setObjectName("reconstructionSettingsHint")
        provider_hint.setWordWrap(True)
        advanced_layout.addWidget(provider_hint)

        provider_selector = QHBoxLayout()
        provider_selector.setSpacing(6)
        provider_label = QLabel("Provider")
        provider_label.setObjectName("reconstructionRowLabel")
        provider_selector.addWidget(provider_label)
        self._dense_provider_combo = QComboBox()
        self._dense_provider_combo.setObjectName("reconstructionDenseProviderCombo")
        self._dense_provider_combo.setMinimumContentsLength(4)
        self._dense_provider_combo.setMinimumWidth(0)
        self._dense_provider_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._dense_provider_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self._dense_provider_combo.setToolTip(
            "Inspect persisted/known reconstruction provider provenance. Changing this selection never activates a provider."
        )
        self._dense_provider_combo.currentIndexChanged.connect(self._on_dense_provider_selected)
        provider_selector.addWidget(self._dense_provider_combo, stretch=1)
        advanced_layout.addLayout(provider_selector)

        self._dense_pipeline_value = self._settings_value("No persistent reconstruction scene")
        self._dense_role_value = self._settings_value("—")
        self._dense_execution_value = self._settings_value("—")
        self._dense_scene_use_value = self._settings_value("—")
        self._dense_scale_value = self._settings_value("—")
        self._dense_integration_value = self._settings_value("DA3 integration available • explicit source-driven action")
        self._dense_state_value = self._settings_value("Waiting for persistent reconstruction scene")
        self._dense_authority_value = self._settings_value("No provider authority presented")
        self._da3_job_value = self._settings_value("Idle • select Image sequence or Recorded video")
        da3_action_row = QHBoxLayout()
        da3_action_row.setSpacing(6)
        self._da3_integrate_button = QPushButton("Integrate DA3")
        self._da3_integrate_button.setObjectName("reconstructionDa3IntegrateButton")
        self._da3_integrate_button.setToolTip(
            "Run the qualified isolated DA3 runtime, then canonicalize its relative pose/depth evidence. Calibration is not required; metric scale is never invented."
        )
        self._da3_integrate_button.clicked.connect(self._on_da3_integrate_clicked)
        self._da3_cancel_button = QPushButton("Cancel")
        self._da3_cancel_button.setObjectName("reconstructionDa3CancelButton")
        self._da3_cancel_button.clicked.connect(self._on_da3_cancel_clicked)
        da3_action_row.addWidget(self._da3_integrate_button, 1)
        da3_action_row.addWidget(self._da3_cancel_button, 0)
        advanced_layout.addLayout(da3_action_row)
        self._add_status_row(advanced_layout, "DA3 job", self._da3_job_value)
        self._add_status_row(advanced_layout, "Pipeline", self._dense_pipeline_value)
        self._add_status_row(advanced_layout, "Role", self._dense_role_value)
        self._add_status_row(advanced_layout, "Execution", self._dense_execution_value)
        self._add_status_row(advanced_layout, "Scene use", self._dense_scene_use_value)
        self._add_status_row(advanced_layout, "Scale", self._dense_scale_value)
        self._add_status_row(advanced_layout, "DA3 bridge", self._dense_integration_value)
        self._add_status_row(advanced_layout, "State", self._dense_state_value)
        self._add_status_row(advanced_layout, "Authority", self._dense_authority_value)
        self._dense_authority_value.setToolTip(
            "Dense/surface canonical authority and supporting depth-provider authority come from the persistent scene. Provider inspection and DA3 controls never promote scale, measurement, or QC authority."
        )
        layout.addWidget(self._advanced_reconstruction_group)
        self._refresh_dense_provider_controls()
        self._refresh_da3_canonical_controls()

        self._offline_inspection_group = self._settings_group("OFFLINE INSPECTION")
        inspection_layout = self._offline_inspection_group.content_layout
        selector_row = QHBoxLayout()
        selector_row.setSpacing(6)
        selector_label = QLabel("Camera")
        selector_label.setObjectName("reconstructionRowLabel")
        selector_row.addWidget(selector_label)
        self._offline_camera_combo = QComboBox()
        self._offline_camera_combo.setObjectName("reconstructionOfflineCameraCombo")
        self._offline_camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        # D4 responsive-sidebar hardening: the camera identifier may be long,
        # but it must not impose that text width as a minimum on the entire
        # scroll body.  The combo remains readable at normal sidebar widths
        # and can shrink when the splitter is constrained.
        self._offline_camera_combo.setMinimumContentsLength(4)
        self._offline_camera_combo.setMinimumWidth(0)
        self._offline_camera_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self._offline_camera_combo.addItem("Select camera…", -1)
        self._offline_camera_combo.setEnabled(False)
        selector_row.addWidget(self._offline_camera_combo, stretch=1)
        inspection_layout.addLayout(selector_row)

        self._offline_camera_position_value = self._settings_value("—")
        self._offline_camera_delta_value = self._settings_value("—")
        self._offline_structure_value = self._settings_value("No offline sparse SfM")
        self._add_status_row(inspection_layout, "Position", self._offline_camera_position_value)
        self._add_status_row(inspection_layout, "Δ previous", self._offline_camera_delta_value)
        self._add_status_row(inspection_layout, "Structure", self._offline_structure_value)

        quality_title = QLabel("SPARSE SFM HEALTH")
        quality_title.setObjectName("reconstructionRowLabel")
        quality_title.setToolTip(
            "Operational reconstruction-health telemetry from the qualified SfM backend; not a metrology score."
        )
        inspection_layout.addWidget(quality_title)
        self._offline_quality_status_value = self._status_value()
        self._offline_quality_registration_value = self._settings_value("—")
        self._offline_quality_reprojection_value = self._settings_value("—")
        self._offline_quality_support_value = self._settings_value("—")
        self._offline_quality_gate_value = self._settings_value(
            "Operational SfM health only • not metrology"
        )
        self._add_status_row(inspection_layout, "Health", self._offline_quality_status_value)
        self._add_status_row(inspection_layout, "Registration", self._offline_quality_registration_value)
        self._add_status_row(inspection_layout, "Reprojection", self._offline_quality_reprojection_value)
        self._add_status_row(inspection_layout, "Support", self._offline_quality_support_value)
        self._add_status_row(inspection_layout, "Gate", self._offline_quality_gate_value)

        spatial_title = QLabel("SPARSE SPATIAL COVERAGE")
        spatial_title.setObjectName("reconstructionRowLabel")
        spatial_title.setToolTip(
            "Presentation-space coverage derived from the bounded sparse-SfM sample; not object dimensions or metric scale."
        )
        inspection_layout.addWidget(spatial_title)
        self._offline_spatial_envelope_value = self._settings_value("—")
        self._offline_spatial_camera_span_value = self._settings_value("—")
        self._offline_spatial_standoff_value = self._settings_value("—")
        self._offline_spatial_basis_value = self._settings_value(
            "Sparse presentation envelope only • not object dimensions"
        )
        self._add_status_row(inspection_layout, "Envelope", self._offline_spatial_envelope_value)
        self._add_status_row(inspection_layout, "Camera span", self._offline_spatial_camera_span_value)
        self._add_status_row(inspection_layout, "Standoff", self._offline_spatial_standoff_value)
        self._add_status_row(inspection_layout, "Basis", self._offline_spatial_basis_value)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        self._offline_prev_camera_button = QPushButton("Prev")
        self._offline_focus_camera_button = QPushButton("Focus")
        self._offline_next_camera_button = QPushButton("Next")
        self._offline_clear_camera_button = QPushButton("Clear")
        for button in (
            self._offline_prev_camera_button,
            self._offline_focus_camera_button,
            self._offline_next_camera_button,
            self._offline_clear_camera_button,
        ):
            button.setEnabled(False)
            # Keep the D2 camera navigation row horizontally shrinkable.
            # QPushButton defaults can otherwise raise the scroll body's
            # minimum width above the narrow settings viewport once D3/D4
            # sections are visible, which clips every right-aligned value
            # even when horizontal scrolling is disabled.
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )
            nav_row.addWidget(button)
        inspection_layout.addLayout(nav_row)
        self._offline_camera_combo.currentIndexChanged.connect(self._on_offline_camera_selected)
        self._offline_prev_camera_button.clicked.connect(self._select_previous_offline_camera)
        self._offline_focus_camera_button.clicked.connect(self._focus_selected_offline_camera)
        self._offline_next_camera_button.clicked.connect(self._select_next_offline_camera)
        self._offline_clear_camera_button.clicked.connect(self._clear_offline_camera_selection)
        self._reset_offline_sfm_health()
        self._reset_offline_spatial_coverage()
        self._offline_inspection_group.setVisible(False)
        layout.addWidget(self._offline_inspection_group)

        self._semantic_objects_group = self._settings_group("SEMANTIC / OBJECTS")
        semantic_layout = self._semantic_objects_group.content_layout
        semantic_selector = QHBoxLayout()
        semantic_selector.setSpacing(6)
        semantic_label = QLabel("Object")
        semantic_label.setObjectName("reconstructionRowLabel")
        semantic_selector.addWidget(semantic_label)
        self._semantic_object_combo = QComboBox()
        self._semantic_object_combo.setObjectName("reconstructionSemanticObjectCombo")
        self._semantic_object_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._semantic_object_combo.setMinimumContentsLength(4)
        self._semantic_object_combo.setMinimumWidth(0)
        self._semantic_object_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._semantic_object_combo.addItem("Select object…", -1)
        self._semantic_object_combo.setEnabled(False)
        semantic_selector.addWidget(self._semantic_object_combo, stretch=1)
        semantic_layout.addLayout(semantic_selector)

        self._semantic_structure_value = self._settings_value("No semantic context")
        self._semantic_role_value = self._settings_value("—")
        self._semantic_support_value = self._settings_value("—")
        self._semantic_quality_value = self._settings_value("—")
        self._semantic_scale_value = self._settings_value("—")
        self._semantic_authority_value = self._settings_value(
            "Semantic context only • not inspection/QC decision"
        )
        self._add_status_row(semantic_layout, "Structure", self._semantic_structure_value)
        self._add_status_row(semantic_layout, "Role", self._semantic_role_value)
        self._add_status_row(semantic_layout, "Support", self._semantic_support_value)
        self._add_status_row(semantic_layout, "Quality", self._semantic_quality_value)
        self._add_status_row(semantic_layout, "Scale", self._semantic_scale_value)
        self._add_status_row(semantic_layout, "Authority", self._semantic_authority_value)

        semantic_nav = QHBoxLayout()
        semantic_nav.setSpacing(6)
        self._semantic_prev_button = QPushButton("Prev")
        self._semantic_focus_button = QPushButton("Focus")
        self._semantic_next_button = QPushButton("Next")
        self._semantic_clear_button = QPushButton("Clear")
        for button in (
            self._semantic_prev_button,
            self._semantic_focus_button,
            self._semantic_next_button,
            self._semantic_clear_button,
        ):
            button.setEnabled(False)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            semantic_nav.addWidget(button)
        semantic_layout.addLayout(semantic_nav)
        self._semantic_object_combo.currentIndexChanged.connect(self._on_semantic_object_selected)
        self._semantic_prev_button.clicked.connect(self._select_previous_semantic_object)
        self._semantic_focus_button.clicked.connect(self._focus_selected_semantic_object)
        self._semantic_next_button.clicked.connect(self._select_next_semantic_object)
        self._semantic_clear_button.clicked.connect(self._clear_semantic_object_selection)
        self._semantic_objects_group.setVisible(False)
        layout.addWidget(self._semantic_objects_group)

        # E1 reuses the D5 semantic selector instead of introducing a second
        # object picker.  With no semantic object selected, the active scene is
        # the target; choosing an existing D5 object changes the target scope
        # to that object.  Later E2-E5 consume this generic selection contract.
        self._inspection_target_group = self._settings_group("INSPECTION TARGET")
        inspection_target_layout = self._inspection_target_group.content_layout
        self._inspection_scope_value = self._settings_value("—")
        self._inspection_scene_value = self._settings_value("No reconstruction scene")
        self._inspection_object_value = self._settings_value("—")
        self._inspection_geometry_value = self._settings_value("No geometry target")
        self._inspection_state_value = self._settings_value("No inspection target")
        self._inspection_authority_value = self._settings_value(
            "Selection only • measurement/QC not evaluated"
        )
        self._add_status_row(inspection_target_layout, "Scope", self._inspection_scope_value)
        self._add_status_row(inspection_target_layout, "Scene", self._inspection_scene_value)
        self._add_status_row(inspection_target_layout, "Object", self._inspection_object_value)
        self._add_status_row(inspection_target_layout, "Geometry", self._inspection_geometry_value)
        self._add_status_row(inspection_target_layout, "State", self._inspection_state_value)
        self._add_status_row(inspection_target_layout, "Authority", self._inspection_authority_value)
        self._inspection_authority_value.setToolTip(
            "E1 establishes target selection only. Measurement, confidence, inspection/QC, and reference comparison remain E2-E5 responsibilities."
        )
        layout.addWidget(self._inspection_target_group)

        # E2 is deliberately presentation-only. Existing Phase 7.5 measurement
        # evidence can be attached to the active E1 object target; no depth,
        # fitting, measurement, uncertainty, QC, or reference algorithm runs here.
        self._reconstruction_measurement_group = self._settings_group("RECONSTRUCTION MEASUREMENTS")
        self._reconstruction_measurement_group.set_expanded(False)
        measurement_layout = self._reconstruction_measurement_group.content_layout
        self._measurement_target_value = self._settings_value("No target")
        self._measurement_source_value = self._settings_value("No measurement evidence")
        self._measurement_values_value = self._settings_value("—")
        self._measurement_state_value = self._settings_value("Waiting for reconstruction target")
        self._measurement_authority_value = self._settings_value(
            "Existing measurements only • E2 does not compute measurements"
        )
        self._add_status_row(measurement_layout, "Target", self._measurement_target_value)
        self._add_status_row(measurement_layout, "Source", self._measurement_source_value)
        self._add_status_row(measurement_layout, "Values", self._measurement_values_value)
        self._add_status_row(measurement_layout, "State", self._measurement_state_value)
        self._add_status_row(measurement_layout, "Authority", self._measurement_authority_value)
        self._measurement_authority_value.setToolTip(
            "E2 presents existing source-authored values only. Uncertainty, reliability, calibration/scale validity, rejection reasons, and confidence are Phase 8.8E3."
        )
        layout.addWidget(self._reconstruction_measurement_group)

        # E3 qualifies the exact E2-bound source observation. It renders only
        # fields authored by the Phase 7.5 measurement result/bundle: status,
        # uncertainty, bounds, reliability, support, scale/calibration context,
        # warnings, and rejection reasons. No thresholds or QC decisions live here.
        self._measurement_validity_group = self._settings_group("MEASUREMENT VALIDITY")
        self._measurement_validity_group.set_expanded(False)
        validity_layout = self._measurement_validity_group.content_layout
        self._measurement_validity_qualified_value = self._settings_value("—")
        self._measurement_validity_status_value = self._settings_value("Waiting for reconstruction target")
        self._measurement_validity_reliability_value = self._settings_value("—")
        self._measurement_validity_uncertainty_value = self._settings_value("—")
        self._measurement_validity_scale_value = self._settings_value("—")
        self._measurement_validity_calibration_value = self._settings_value("—")
        self._measurement_validity_support_value = self._settings_value("—")
        self._measurement_validity_reasons_value = self._settings_value("—")
        self._measurement_validity_authority_value = self._settings_value(
            "Existing validity evidence only • E3 does not compute confidence/uncertainty"
        )
        self._add_status_row(validity_layout, "Qualified", self._measurement_validity_qualified_value)
        self._add_status_row(validity_layout, "Validity", self._measurement_validity_status_value)
        self._add_status_row(validity_layout, "Reliability", self._measurement_validity_reliability_value)
        self._add_status_row(validity_layout, "Uncertainty", self._measurement_validity_uncertainty_value)
        self._add_status_row(validity_layout, "Scale", self._measurement_validity_scale_value)
        self._add_status_row(validity_layout, "Calibration", self._measurement_validity_calibration_value)
        self._add_status_row(validity_layout, "Support", self._measurement_validity_support_value)
        self._add_status_row(validity_layout, "Reasons", self._measurement_validity_reasons_value)
        self._add_status_row(validity_layout, "Authority", self._measurement_validity_authority_value)
        self._measurement_validity_authority_value.setToolTip(
            "E3 presents source-authored measurement validity and uncertainty. It does not infer reliability from semantic confidence, run inspection/QC, compare references, or certify metrology."
        )
        layout.addWidget(self._measurement_validity_group)

        # 9.3J adds an optional, fail-closed spatial registration layer over the
        # original Phase 7.5 measurement. It never makes calibration mandatory
        # and only draws canonical dimension endpoints when registration is qualified.
        self._canonical_registration_group = self._settings_group("CANONICAL MEASUREMENT REGISTRATION")
        self._canonical_registration_group.set_expanded(False)
        canonical_registration_layout = self._canonical_registration_group.content_layout
        self._canonical_registration_state_value = self._settings_value("Unavailable")
        self._canonical_registration_binding_value = self._settings_value("No exact canonical binding")
        self._canonical_registration_dimensions_value = self._settings_value("No canonical dimension geometry")
        self._canonical_registration_uncertainty_value = self._settings_value("No placement uncertainty")
        self._canonical_registration_reason_value = self._settings_value(
            "No qualified measurement→canonical registration"
        )
        self._canonical_registration_authority_value = self._settings_value(
            "Original measurement remains source truth • canonical placement is additive"
        )
        self._add_status_row(canonical_registration_layout, "State", self._canonical_registration_state_value)
        self._add_status_row(canonical_registration_layout, "Binding", self._canonical_registration_binding_value)
        self._add_status_row(canonical_registration_layout, "Dimensions", self._canonical_registration_dimensions_value)
        self._add_status_row(canonical_registration_layout, "Placement ±", self._canonical_registration_uncertainty_value)
        self._add_status_row(canonical_registration_layout, "Reason", self._canonical_registration_reason_value)
        self._add_status_row(canonical_registration_layout, "Authority", self._canonical_registration_authority_value)
        self._canonical_registration_authority_value.setToolTip(
            "Registration never rewrites the original Phase 7.5 value/uncertainty, never creates metric scale, and never certifies metrology."
        )
        layout.addWidget(self._canonical_registration_group)

        # E4 presents an already-produced Phase 8.7C inspection/QC report.
        # The decision and findings remain source-authored. Registration/QC is
        # not rerun here and the reference selection/comparison workflow is E5.
        self._inspection_qc_group = self._settings_group("INSPECTION / QC")
        self._inspection_qc_group.set_expanded(False)
        inspection_qc_layout = self._inspection_qc_group.content_layout
        self._inspection_qc_target_value = self._settings_value("No target")
        self._inspection_qc_decision_value = self._settings_value("Not evaluated")
        self._inspection_qc_findings_value = self._settings_value("—")
        self._inspection_qc_coverage_value = self._settings_value("—")
        self._inspection_qc_decisions_value = self._settings_value("—")
        self._inspection_qc_registration_value = self._settings_value("No inspection report attached")
        self._inspection_qc_comparison_value = self._settings_value("—")
        self._inspection_qc_authority_value = self._settings_value(
            "Existing QC only • E4 does not run inspection"
        )
        self._add_status_row(inspection_qc_layout, "Target", self._inspection_qc_target_value)
        self._add_status_row(inspection_qc_layout, "Decision", self._inspection_qc_decision_value)
        self._add_status_row(inspection_qc_layout, "Findings", self._inspection_qc_findings_value)
        self._add_status_row(inspection_qc_layout, "Coverage", self._inspection_qc_coverage_value)
        self._add_status_row(inspection_qc_layout, "Summary", self._inspection_qc_decisions_value)
        self._add_status_row(inspection_qc_layout, "Registration", self._inspection_qc_registration_value)
        self._add_status_row(inspection_qc_layout, "Comparison", self._inspection_qc_comparison_value)
        self._add_status_row(inspection_qc_layout, "Authority", self._inspection_qc_authority_value)
        self._inspection_qc_authority_value.setToolTip(
            "E4 displays existing Phase 8.7C inspection/QC decisions only. It does not infer decisions from semantics/measurements, rerun registration, or implement the E5 reference-comparison workflow."
        )
        layout.addWidget(self._inspection_qc_group)

        # E5 makes reference identity explicit and pairs it with the exact
        # existing Phase 8.7C report for the current scene. It does not parse
        # CAD geometry, rerun registration, apply tolerance rules, or promote
        # an arbitrary reconstruction to metric/certified authority.
        self._reference_comparison_group = self._settings_group("REFERENCE COMPARISON")
        self._reference_comparison_group.set_expanded(False)
        reference_layout = self._reference_comparison_group.content_layout
        self._reference_comparison_target_value = self._settings_value("No target")
        self._reference_comparison_reference_value = self._settings_value("No reference attached")
        self._reference_comparison_structure_value = self._settings_value("—")
        self._reference_comparison_registration_value = self._settings_value("—")
        self._reference_comparison_deviations_value = self._settings_value("—")
        self._reference_comparison_evidence_value = self._settings_value("—")
        self._reference_comparison_scale_value = self._settings_value("—")
        self._reference_comparison_state_value = self._settings_value("Waiting for reconstruction target")
        self._reference_comparison_authority_value = self._settings_value(
            "Reference workflow only • E5 does not register/apply tolerances"
        )
        self._add_status_row(reference_layout, "Target", self._reference_comparison_target_value)
        self._add_status_row(reference_layout, "Reference", self._reference_comparison_reference_value)
        self._add_status_row(reference_layout, "Structure", self._reference_comparison_structure_value)
        self._add_status_row(reference_layout, "Registration", self._reference_comparison_registration_value)
        self._add_status_row(reference_layout, "Deviation", self._reference_comparison_deviations_value)
        self._add_status_row(reference_layout, "Evidence", self._reference_comparison_evidence_value)
        self._add_status_row(reference_layout, "Scale", self._reference_comparison_scale_value)
        self._add_status_row(reference_layout, "State", self._reference_comparison_state_value)
        self._add_status_row(reference_layout, "Authority", self._reference_comparison_authority_value)
        self._reference_comparison_authority_value.setToolTip(
            "E5 loads ReferenceAssembly JSON metadata and presents exact matching Phase 8.7C comparison evidence. Registration, deviations, and tolerance decisions are source-authored; E5 does not compute them or certify metrology."
        )
        reference_buttons = QHBoxLayout()
        reference_buttons.setSpacing(6)
        self._reference_comparison_load_button = QPushButton("Load JSON")
        self._reference_comparison_clear_button = QPushButton("Clear")
        for button in (self._reference_comparison_load_button, self._reference_comparison_clear_button):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            reference_buttons.addWidget(button)
        self._reference_comparison_load_button.setEnabled(False)
        self._reference_comparison_clear_button.setEnabled(False)
        self._reference_comparison_load_button.clicked.connect(self._choose_reconstruction_reference_json)
        self._reference_comparison_clear_button.clicked.connect(self._clear_active_reconstruction_reference)
        reference_layout.addLayout(reference_buttons)
        layout.addWidget(self._reference_comparison_group)

        # F1 supplies persisted neural appearance metadata. F2/F3 supply the
        # qualified static previews. F4 reuses this selector as the explicit
        # user-driven viewport switch. F5 adds resource/fallback state below.
        self._appearance_group = self._settings_group("APPEARANCE REPRESENTATIONS")
        appearance_layout = self._appearance_group.content_layout
        appearance_selector = QHBoxLayout()
        appearance_selector.setSpacing(6)
        appearance_label = QLabel("Representation")
        appearance_label.setObjectName("reconstructionRowLabel")
        appearance_selector.addWidget(appearance_label)
        self._appearance_representation_combo = QComboBox()
        self._appearance_representation_combo.setObjectName("reconstructionAppearanceRepresentationCombo")
        self._appearance_representation_combo.setMinimumContentsLength(4)
        self._appearance_representation_combo.setMinimumWidth(0)
        self._appearance_representation_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._appearance_representation_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._appearance_representation_combo.setToolTip(
            "Inspect persisted appearance metadata. In F4, activating Gaussian or NeRF switches the viewport to its existing qualified preview without launching a backend."
        )
        self._appearance_representation_combo.currentIndexChanged.connect(
            self._on_appearance_representation_selected
        )
        self._appearance_representation_combo.activated.connect(
            self._on_appearance_representation_activated
        )
        appearance_selector.addWidget(self._appearance_representation_combo, stretch=1)
        appearance_layout.addLayout(appearance_selector)
        self._appearance_available_value = self._settings_value("No persistent appearance scene")
        self._appearance_provider_value = self._settings_value("—")
        self._appearance_artifact_value = self._settings_value("—")
        self._appearance_render_value = self._settings_value("—")
        self._appearance_scale_value = self._settings_value("—")
        self._appearance_state_value = self._settings_value("Waiting for persistent reconstruction scene")
        self._appearance_authority_value = self._settings_value(
            "Appearance only • canonical geometry remains authoritative"
        )
        self._appearance_viewport_value = self._settings_value(canonical_view_text())
        # Keep the complete authority statement visibly readable at the
        # minimum Reconstruction sidebar width. QLabel height-for-width can
        # otherwise under-allocate this final wrapped row inside the scroll
        # body, visually clipping the safety-critical last line.
        self._appearance_authority_value.setMinimumHeight(
            self._appearance_authority_value.fontMetrics().lineSpacing() * 3
        )
        self._add_status_row(appearance_layout, "Available", self._appearance_available_value)
        self._add_status_row(appearance_layout, "Provider", self._appearance_provider_value)
        self._add_status_row(appearance_layout, "Artifact", self._appearance_artifact_value)
        self._add_status_row(appearance_layout, "Render", self._appearance_render_value)
        self._add_status_row(appearance_layout, "Scale", self._appearance_scale_value)
        self._add_status_row(appearance_layout, "State", self._appearance_state_value)
        self._add_status_row(appearance_layout, "Viewport", self._appearance_viewport_value)
        self._add_status_row(appearance_layout, "Authority", self._appearance_authority_value)
        layout.addWidget(self._appearance_group)
        self._refresh_appearance_representation_controls()
        self._refresh_representation_switching_controls()

        # F2 visualizes the backend-authored Gaussian preview image only. It does
        # not parse heavy scene payloads, import Gaussian runtimes, retrain, or
        # register geometry. F4 only coordinates this already-qualified preview.
        self._gaussian_visualization_group = self._settings_group("GAUSSIAN SPLATTING VISUALIZATION")
        self._gaussian_visualization_group.set_expanded(False)
        gaussian_layout = self._gaussian_visualization_group.content_layout
        self._gaussian_provider_value = self._settings_value("—")
        self._gaussian_scene_value = self._settings_value("—")
        self._gaussian_preview_value = self._settings_value("—")
        self._gaussian_scale_value = self._settings_value("—")
        self._gaussian_state_value = self._settings_value("No persistent Gaussian representation")
        self._gaussian_display_value = self._settings_value("Interactive display not active")
        self._gaussian_resources_value = self._settings_value("Unloaded • no Gaussian viewer buffers resident")
        self._gaussian_performance_value = self._settings_value("No interactive Gaussian load measured")
        self._gaussian_lifecycle_value = self._settings_value("Unloaded • no Gaussian viewer buffers resident")
        self._gaussian_authority_value = self._settings_value(
            "Appearance only • canonical geometry remains authoritative"
        )
        self._gaussian_authority_value.setMinimumHeight(
            self._gaussian_authority_value.fontMetrics().lineSpacing() * 3
        )
        self._add_status_row(gaussian_layout, "Provider", self._gaussian_provider_value)
        self._add_status_row(gaussian_layout, "Scene", self._gaussian_scene_value)
        self._add_status_row(gaussian_layout, "Preview", self._gaussian_preview_value)
        self._add_status_row(gaussian_layout, "Scale", self._gaussian_scale_value)
        self._add_status_row(gaussian_layout, "State", self._gaussian_state_value)
        self._add_status_row(gaussian_layout, "Display", self._gaussian_display_value)
        self._add_status_row(gaussian_layout, "Resources", self._gaussian_resources_value)
        self._add_status_row(gaussian_layout, "Performance", self._gaussian_performance_value)
        self._add_status_row(gaussian_layout, "Lifecycle", self._gaussian_lifecycle_value)
        self._add_status_row(gaussian_layout, "Authority", self._gaussian_authority_value)

        gaussian_controls_hint = QLabel(
            "Interactive display only • density, size and opacity never modify the persisted Gaussian artifact."
        )
        gaussian_controls_hint.setObjectName("reconstructionSettingsHint")
        gaussian_controls_hint.setWordWrap(True)
        gaussian_layout.addWidget(gaussian_controls_hint)

        density_row = QHBoxLayout()
        density_row.setSpacing(8)
        density_label = QLabel("Density")
        density_label.setObjectName("reconstructionRowLabel")
        density_row.addWidget(density_label)
        self._gaussian_density_combo = QComboBox()
        self._gaussian_density_combo.setObjectName("reconstructionGaussianDensityCombo")
        for percent in GAUSSIAN_DENSITY_PERCENT_OPTIONS:
            label = "Full" if percent == 100 else f"{percent}%"
            self._gaussian_density_combo.addItem(label, percent)
        self._gaussian_density_combo.setCurrentIndex(0)
        self._gaussian_density_combo.setMinimumWidth(0)
        self._gaussian_density_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._gaussian_density_combo.setToolTip(
            "Display a deterministic subset of the persisted Gaussians. This changes presentation density only."
        )
        self._gaussian_density_combo.currentIndexChanged.connect(self._on_gaussian_display_controls_changed)
        density_row.addWidget(self._gaussian_density_combo, 1)
        gaussian_layout.addLayout(density_row)

        size_row = QHBoxLayout()
        size_row.setSpacing(8)
        size_label = QLabel("Splat size")
        size_label.setObjectName("reconstructionRowLabel")
        size_row.addWidget(size_label)
        self._gaussian_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._gaussian_size_slider.setObjectName("reconstructionGaussianDisplaySlider")
        self._gaussian_size_slider.setRange(GAUSSIAN_SIZE_PERCENT_MIN, GAUSSIAN_SIZE_PERCENT_MAX)
        self._gaussian_size_slider.setValue(100)
        self._gaussian_size_slider.setToolTip(
            "Scale screen-space splat size for interactive viewing only. Persisted Gaussian scales are unchanged."
        )
        self._gaussian_size_slider.valueChanged.connect(self._on_gaussian_display_controls_changed)
        size_row.addWidget(self._gaussian_size_slider, 1)
        self._gaussian_size_value = QLabel("100%")
        self._gaussian_size_value.setObjectName("reconstructionSettingsValue")
        self._gaussian_size_value.setMinimumWidth(40)
        self._gaussian_size_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        size_row.addWidget(self._gaussian_size_value)
        gaussian_layout.addLayout(size_row)

        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(8)
        opacity_label = QLabel("Opacity")
        opacity_label.setObjectName("reconstructionRowLabel")
        opacity_row.addWidget(opacity_label)
        self._gaussian_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._gaussian_opacity_slider.setObjectName("reconstructionGaussianDisplaySlider")
        self._gaussian_opacity_slider.setRange(
            GAUSSIAN_OPACITY_PERCENT_MIN, GAUSSIAN_OPACITY_PERCENT_MAX
        )
        self._gaussian_opacity_slider.setValue(100)
        self._gaussian_opacity_slider.setToolTip(
            "Scale interactive Gaussian alpha only. Persisted opacities and all scene evidence remain unchanged."
        )
        self._gaussian_opacity_slider.valueChanged.connect(self._on_gaussian_display_controls_changed)
        opacity_row.addWidget(self._gaussian_opacity_slider, 1)
        self._gaussian_opacity_value = QLabel("100%")
        self._gaussian_opacity_value.setObjectName("reconstructionSettingsValue")
        self._gaussian_opacity_value.setMinimumWidth(40)
        self._gaussian_opacity_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        opacity_row.addWidget(self._gaussian_opacity_value)
        gaussian_layout.addLayout(opacity_row)

        self._gaussian_reset_display_button = QPushButton("Reset display")
        self._gaussian_reset_display_button.setObjectName("reconstructionGaussianPreviewAction")
        self._gaussian_reset_display_button.setToolTip(
            "Restore Full density, 100% splat size and 100% opacity. No persisted data is modified."
        )
        self._gaussian_reset_display_button.clicked.connect(self._reset_gaussian_display_controls)
        gaussian_layout.addWidget(self._gaussian_reset_display_button)

        gaussian_buttons = QHBoxLayout()
        gaussian_buttons.setSpacing(6)
        self._gaussian_interactive_button = QPushButton("Interactive")
        self._gaussian_interactive_button.setObjectName("reconstructionGaussianPreviewAction")
        self._gaussian_interactive_button.setToolTip(
            "Explore the persisted Gaussian appearance interactively. No gsplat/Graphdeco runtime is imported; picking remains canonical."
        )
        self._gaussian_interactive_button.clicked.connect(self._show_interactive_gaussian)
        self._gaussian_show_button = QPushButton("Preview")
        self._gaussian_show_button.setObjectName("reconstructionGaussianPreviewAction")
        self._gaussian_show_button.setToolTip(
            "Show the existing static backend-authored Gaussian preview image. No Gaussian runtime is launched."
        )
        self._gaussian_show_button.clicked.connect(self._show_gaussian_preview)
        self._gaussian_hide_button = QPushButton("Hide")
        self._gaussian_hide_button.setObjectName("reconstructionGaussianPreviewAction")
        self._gaussian_hide_button.clicked.connect(self._hide_gaussian_visualization)
        for button in (self._gaussian_interactive_button, self._gaussian_show_button, self._gaussian_hide_button):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            gaussian_buttons.addWidget(button)
        gaussian_layout.addLayout(gaussian_buttons)
        layout.addWidget(self._gaussian_visualization_group)
        self._refresh_gaussian_visualization_controls()
        self._refresh_gaussian_runtime_health()

        # Phase 9.5E retains the qualified static preview and adds an explicit
        # Interactive mode backed by the isolated Nerfstudio process. The UI
        # process never imports Nerfstudio/tiny-cuda-nn and canonical geometry
        # remains the only picking/measurement authority.
        self._nerf_visualization_group = self._settings_group("NERF VISUALIZATION")
        self._nerf_visualization_group.set_expanded(False)
        nerf_layout = self._nerf_visualization_group.content_layout
        self._nerf_provider_value = self._settings_value("—")
        self._nerf_scene_value = self._settings_value("—")
        self._nerf_preview_value = self._settings_value("—")
        self._nerf_scale_value = self._settings_value("—")
        self._nerf_state_value = self._settings_value("No persistent NeRF representation")
        self._nerf_authority_value = self._settings_value(
            "Appearance only • canonical geometry remains authoritative"
        )
        self._nerf_authority_value.setMinimumHeight(
            self._nerf_authority_value.fontMetrics().lineSpacing() * 3
        )
        self._add_status_row(nerf_layout, "Provider", self._nerf_provider_value)
        self._add_status_row(nerf_layout, "Scene", self._nerf_scene_value)
        self._add_status_row(nerf_layout, "Preview", self._nerf_preview_value)
        self._nerf_display_value = self._settings_value("Interactive display not active")
        self._nerf_resources_value = self._settings_value("External renderer unloaded")
        self._nerf_performance_value = self._settings_value("No interactive render measured")
        self._nerf_lifecycle_value = self._settings_value("Unloaded • no NeRF worker resident")
        self._add_status_row(nerf_layout, "Scale", self._nerf_scale_value)
        self._add_status_row(nerf_layout, "State", self._nerf_state_value)
        self._add_status_row(nerf_layout, "Display", self._nerf_display_value)
        self._add_status_row(nerf_layout, "Resources", self._nerf_resources_value)
        self._add_status_row(nerf_layout, "Performance", self._nerf_performance_value)
        self._add_status_row(nerf_layout, "Lifecycle", self._nerf_lifecycle_value)
        self._add_status_row(nerf_layout, "Authority", self._nerf_authority_value)

        nerf_controls_hint = QLabel(
            "Interactive display only • detail, opacity and refresh policy never modify the persisted NeRF artifact."
        )
        nerf_controls_hint.setObjectName("reconstructionSettingsHint")
        nerf_controls_hint.setWordWrap(True)
        nerf_layout.addWidget(nerf_controls_hint)

        nerf_profile_row = QHBoxLayout()
        nerf_profile_row.setSpacing(8)
        nerf_profile_label = QLabel("Render detail")
        nerf_profile_label.setObjectName("reconstructionRowLabel")
        nerf_profile_row.addWidget(nerf_profile_label)
        self._nerf_render_profile_combo = QComboBox()
        self._nerf_render_profile_combo.setObjectName("reconstructionNerfRenderProfileCombo")
        for key, label, _width, _height in NERF_RENDER_PROFILES:
            self._nerf_render_profile_combo.addItem(label, key)
        self._nerf_render_profile_combo.setCurrentIndex(
            max(0, self._nerf_render_profile_combo.findData(NERF_DEFAULT_RENDER_PROFILE))
        )
        self._nerf_render_profile_combo.setToolTip(
            "Choose the next interactive render resolution only. Training/checkpoint quality is unchanged."
        )
        self._nerf_render_profile_combo.currentIndexChanged.connect(self._on_nerf_display_controls_changed)
        nerf_profile_row.addWidget(self._nerf_render_profile_combo, 1)
        nerf_layout.addLayout(nerf_profile_row)

        nerf_opacity_row = QHBoxLayout()
        nerf_opacity_row.setSpacing(8)
        nerf_opacity_label = QLabel("Opacity")
        nerf_opacity_label.setObjectName("reconstructionRowLabel")
        nerf_opacity_row.addWidget(nerf_opacity_label)
        self._nerf_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._nerf_opacity_slider.setObjectName("reconstructionNerfDisplaySlider")
        self._nerf_opacity_slider.setRange(NERF_OPACITY_PERCENT_MIN, NERF_OPACITY_PERCENT_MAX)
        self._nerf_opacity_slider.setValue(100)
        self._nerf_opacity_slider.setToolTip(
            "Blend the interactive NeRF image over the canonical viewport. This is presentation opacity only."
        )
        self._nerf_opacity_slider.valueChanged.connect(self._on_nerf_display_controls_changed)
        nerf_opacity_row.addWidget(self._nerf_opacity_slider, 1)
        self._nerf_opacity_value = QLabel("100%")
        self._nerf_opacity_value.setObjectName("reconstructionSettingsValue")
        self._nerf_opacity_value.setMinimumWidth(40)
        self._nerf_opacity_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        nerf_opacity_row.addWidget(self._nerf_opacity_value)
        nerf_layout.addLayout(nerf_opacity_row)

        nerf_refresh_row = QHBoxLayout()
        nerf_refresh_row.setSpacing(8)
        nerf_refresh_label = QLabel("Auto refresh")
        nerf_refresh_label.setObjectName("reconstructionRowLabel")
        nerf_refresh_label.setToolTip(
            "When enabled, request a new NeRF image after canonical navigation. Disable to render only on Refresh view."
        )
        nerf_refresh_row.addWidget(nerf_refresh_label, 1)
        self._nerf_auto_refresh_toggle = ToggleSwitch(checked=True, width=32, height=18)
        self._nerf_auto_refresh_toggle.setObjectName("reconstructionNerfAutoRefreshToggle")
        self._nerf_auto_refresh_toggle.setAccessibleName("NeRF auto refresh")
        self._nerf_auto_refresh_toggle.setToolTip(nerf_refresh_label.toolTip())
        self._nerf_auto_refresh_toggle.toggled.connect(self._on_nerf_display_controls_changed)
        nerf_refresh_row.addWidget(self._nerf_auto_refresh_toggle)
        nerf_layout.addLayout(nerf_refresh_row)

        nerf_display_actions = QHBoxLayout()
        nerf_display_actions.setSpacing(6)
        self._nerf_refresh_view_button = QPushButton("Refresh view")
        self._nerf_refresh_view_button.setObjectName("reconstructionNerfPreviewAction")
        self._nerf_refresh_view_button.setToolTip(
            "Request one render for the current canonical camera using the selected display detail."
        )
        self._nerf_refresh_view_button.clicked.connect(
            lambda _checked=False: self._request_interactive_nerf_render(force=True)
        )
        self._nerf_reset_display_button = QPushButton("Reset display")
        self._nerf_reset_display_button.setObjectName("reconstructionNerfPreviewAction")
        self._nerf_reset_display_button.setToolTip(
            "Restore Detail 640×360, 100% opacity and automatic refresh. Persisted data is unchanged."
        )
        self._nerf_reset_display_button.clicked.connect(self._reset_nerf_display_controls)
        for button in (self._nerf_refresh_view_button, self._nerf_reset_display_button):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            nerf_display_actions.addWidget(button)
        nerf_layout.addLayout(nerf_display_actions)

        nerf_buttons = QHBoxLayout()
        nerf_buttons.setSpacing(6)
        self._nerf_interactive_button = QPushButton("Interactive")
        self._nerf_interactive_button.setObjectName("reconstructionNerfPreviewAction")
        self._nerf_interactive_button.setToolTip(
            "Start the isolated persisted Nerfacto renderer. Navigation stays on the canonical viewer; NeRF remains appearance only."
        )
        self._nerf_interactive_button.clicked.connect(self._show_interactive_nerf)
        self._nerf_show_button = QPushButton("Preview")
        self._nerf_show_button.setObjectName("reconstructionNerfPreviewAction")
        self._nerf_show_button.setToolTip(
            "Show the existing backend-authored NeRF preview image without starting the NeRF runtime."
        )
        self._nerf_show_button.clicked.connect(self._show_nerf_preview)
        self._nerf_hide_button = QPushButton("Hide")
        self._nerf_hide_button.setObjectName("reconstructionNerfPreviewAction")
        self._nerf_hide_button.clicked.connect(lambda _checked=False: self._hide_nerf_visualization())
        for button in (self._nerf_interactive_button, self._nerf_show_button, self._nerf_hide_button):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            nerf_buttons.addWidget(button)
        nerf_layout.addLayout(nerf_buttons)
        layout.addWidget(self._nerf_visualization_group)
        self._refresh_nerf_visualization_controls()
        self._refresh_nerf_runtime_health()

        # F5 consumes the application's existing startup hardware snapshot plus
        # persisted appearance/fallback evidence. Existing static previews remain
        # displayable without launching an external neural backend, and failures
        # retain canonical geometry without cross-representation substitution.
        self._appearance_resource_group = self._settings_group("RESOURCE & FALLBACK")
        self._appearance_resource_group.set_expanded(False)
        resource_layout = self._appearance_resource_group.content_layout
        self._appearance_resource_device_value = self._settings_value("—")
        self._appearance_resource_resources_value = self._settings_value("Startup resource snapshot unavailable")
        self._appearance_resource_presentation_value = self._settings_value("Canonical viewport")
        self._appearance_resource_fallback_value = self._settings_value(
            "Canonical geometry • no neural cross-fallback"
        )
        self._appearance_resource_pipeline_value = self._settings_value("No persistent scene")
        self._appearance_resource_state_value = self._settings_value(
            "Waiting for persistent reconstruction scene"
        )
        self._add_status_row(resource_layout, "Device", self._appearance_resource_device_value)
        self._add_status_row(resource_layout, "Resources", self._appearance_resource_resources_value)
        self._add_status_row(resource_layout, "Presentation", self._appearance_resource_presentation_value)
        self._add_status_row(resource_layout, "Fallback", self._appearance_resource_fallback_value)
        self._add_status_row(resource_layout, "Pipeline", self._appearance_resource_pipeline_value)
        self._add_status_row(resource_layout, "State", self._appearance_resource_state_value)
        layout.addWidget(self._appearance_resource_group)
        self._refresh_appearance_resource_fallback_controls()

        authority = self._settings_group("AUTHORITY & SAFETY")
        self._authority_group = authority
        authority_layout = authority.content_layout
        self._metric_scale_value = self._settings_value("Evidence-driven")
        self._camera_geometry_value = self._settings_value("Not selected")
        self._calibration_profile_value = self._settings_value("No profile selected")
        self._calibration_resolution_value = self._settings_value("—")
        self._metrology_value = self._settings_value("Not certified")
        self._add_status_row(authority_layout, "Calibration", self._camera_geometry_value)
        self._add_status_row(authority_layout, "Profile", self._calibration_profile_value)
        self._add_status_row(authority_layout, "Calib. resolution", self._calibration_resolution_value)
        self._add_status_row(authority_layout, "Metric scale", self._metric_scale_value)
        self._add_status_row(authority_layout, "Metrology", self._metrology_value)
        calibration_actions = QHBoxLayout()
        calibration_actions.setSpacing(6)
        self._calibration_manage_button = QPushButton("Manage Calibration")
        self._calibration_manage_button.setObjectName("reconstructionRuntimeAction")
        self._calibration_manage_button.setToolTip(
            "Create, import, validate, save, or select a reusable camera calibration profile. "
            "Camera calibration remains separate from metric-scale qualification."
        )
        self._calibration_manage_button.clicked.connect(self._open_camera_calibration_workspace)
        self._calibration_clear_button = QPushButton("Clear")
        self._calibration_clear_button.setObjectName("reconstructionRuntimeAction")
        self._calibration_clear_button.setToolTip(
            "Clear only the selected Phase 9 calibration profile. Saved profiles remain in the reusable library."
        )
        self._calibration_clear_button.setEnabled(False)
        self._calibration_clear_button.clicked.connect(self.clear_camera_calibration_profile)
        for button in (self._calibration_manage_button, self._calibration_clear_button):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            calibration_actions.addWidget(button)
        authority_layout.addLayout(calibration_actions)
        self._add_settings_row(authority_layout, "Heavy backends", "Isolated workers")
        layout.addWidget(authority)

        # Phase 9.6E4 is a presentation-only consolidation. Keep every legacy
        # group, control, action and disclosure default intact, but promote the
        # unified Digital Twin context to the first visible sidebar stack. This
        # avoids duplicating old controls while making active representation,
        # authority/scale and heavyweight lifecycle understandable at a glance.
        self._apply_digital_twin_workspace_ux_order(layout)
        layout.addStretch(1)
        self._settings_scroll.setWidget(settings_body)
        panel_layout.addWidget(self._settings_scroll, stretch=1)
        return panel

    @property
    def dense_provider_presentation(self) -> DenseProviderPresentation | None:
        return self._dense_provider_presentation

    @property
    def selected_dense_provider_entry(self) -> DenseProviderEntry | None:
        presentation = self._dense_provider_presentation
        if presentation is None:
            return None
        provider_id = str(self._selected_dense_provider_id).strip()
        return next((item for item in presentation.entries if item.provider_id == provider_id), None)

    def _on_dense_provider_selected(self, combo_index: int) -> None:
        if not hasattr(self, "_dense_provider_combo"):
            return
        self._selected_dense_provider_id = str(
            self._dense_provider_combo.itemData(combo_index) or ""
        ).strip()
        self._refresh_dense_provider_values()

    def _refresh_dense_provider_controls(self) -> None:
        if not hasattr(self, "_dense_provider_combo"):
            return
        previous = self._dense_provider_presentation
        try:
            presentation = build_dense_provider_presentation(
                self._persistent_scene,
                self._canonical_scene_presentation,
            )
        except (TypeError, ValueError) as exc:
            self._dense_provider_presentation = None
            self._selected_dense_provider_id = ""
            self._dense_provider_combo.blockSignals(True)
            self._dense_provider_combo.clear()
            self._dense_provider_combo.addItem("Provider evidence unavailable", "")
            self._dense_provider_combo.setEnabled(False)
            self._dense_provider_combo.blockSignals(False)
            self._dense_pipeline_value.setText("Provider presentation rejected")
            self._dense_role_value.setText("—")
            self._dense_execution_value.setText("—")
            self._dense_scene_use_value.setText("—")
            self._dense_scale_value.setText("—")
            self._dense_integration_value.setText("DA3 integration unavailable until provider evidence is valid")
            self._dense_state_value.setText("Provider evidence unavailable")
            self._dense_state_value.setToolTip(str(exc))
            self._dense_authority_value.setText("No authority promoted")
            if previous is not None:
                self.dense_provider_presentation_changed.emit(None)
            return

        self._dense_provider_presentation = presentation
        self._dense_provider_combo.blockSignals(True)
        self._dense_provider_combo.clear()
        if not presentation.available:
            self._dense_provider_combo.addItem("No persistent scene", "")
            self._dense_provider_combo.setEnabled(False)
            self._selected_dense_provider_id = ""
        else:
            for entry in presentation.entries:
                label = f"{entry.label} • {entry.provider_id}"
                self._dense_provider_combo.addItem(label, entry.provider_id)
            self._dense_provider_combo.setEnabled(bool(presentation.entries))
            if not any(
                item.provider_id == self._selected_dense_provider_id
                for item in presentation.entries
            ):
                self._selected_dense_provider_id = (
                    presentation.entries[0].provider_id if presentation.entries else ""
                )
            index = self._dense_provider_combo.findData(self._selected_dense_provider_id)
            if index >= 0:
                self._dense_provider_combo.setCurrentIndex(index)
        self._dense_provider_combo.blockSignals(False)
        self._refresh_dense_provider_values()
        if presentation != previous:
            self.dense_provider_presentation_changed.emit(presentation)

    def _refresh_dense_provider_values(self) -> None:
        if not hasattr(self, "_dense_pipeline_value"):
            return
        presentation = self._dense_provider_presentation
        if presentation is None:
            return
        self._dense_pipeline_value.setText(presentation.pipeline_text)
        self._dense_scale_value.setText(presentation.scale_text)
        self._dense_integration_value.setText(presentation.integration_text)
        entry = self.selected_dense_provider_entry
        if entry is None:
            self._dense_role_value.setText("—")
            self._dense_execution_value.setText("—")
            self._dense_scene_use_value.setText("—")
            self._dense_state_value.setText(
                "Waiting for persistent reconstruction scene"
                if not presentation.available
                else "Provider not selected"
            )
            self._dense_authority_value.setText(presentation.authority_text)
            return
        self._dense_role_value.setText(entry.role_text)
        self._dense_execution_value.setText(entry.execution_text)
        self._dense_scene_use_value.setText(entry.scene_use_text)
        self._dense_state_value.setText(entry.state_text)
        self._dense_state_value.setToolTip(
            f"Provider ID: {entry.provider_id}\n"
            f"Scene bound: {entry.scene_bound}\n"
            f"External: {entry.external}\n"
            "Provider selection is inspection-only. Only the explicit 9.3H Integrate DA3 action can launch the isolated DA3 workflow."
        )
        self._dense_authority_value.setText(entry.authority_text)

    def _refresh_da3_canonical_controls(self) -> None:
        if not hasattr(self, "_da3_integrate_button"):
            return
        controller = self._controller
        active = self._da3_canonical_state in {
            DA3CanonicalRunState.STARTING,
            DA3CanonicalRunState.RUNNING,
            DA3CanonicalRunState.CANCELLING,
        }
        if controller is None:
            self._da3_integrate_button.setEnabled(False)
            self._da3_cancel_button.setEnabled(False)
            self._da3_job_value.setText("Controller unavailable")
            return
        selection = controller.source_selection
        source_ready = bool(
            selection.ready
            and selection.path is not None
            and selection.workflow in {
                ReconstructionWorkflow.IMAGE_SEQUENCE,
                ReconstructionWorkflow.RECORDED_VIDEO,
            }
        )
        can_start = bool(
            not controller.is_shutdown
            and source_ready
            and not controller.is_active
            and not active
            and not self._digital_twin_processing_snapshot.is_active
        )
        self._da3_integrate_button.setEnabled(can_start)
        self._da3_cancel_button.setEnabled(
            not controller.is_shutdown
            and self._da3_canonical_state in {
                DA3CanonicalRunState.STARTING,
                DA3CanonicalRunState.RUNNING,
            }
        )
        if self._da3_canonical_state is DA3CanonicalRunState.IDLE:
            self._da3_job_value.setText(
                "Ready • isolated runtime • calibration optional"
                if source_ready
                else "Idle • select Image sequence or Recorded video"
            )
        elif self._da3_canonical_state is DA3CanonicalRunState.STARTING:
            self._da3_job_value.setText("Starting isolated DA3 integration…")
        elif self._da3_canonical_state is DA3CanonicalRunState.RUNNING:
            self._da3_job_value.setText("Running • external DA3 + canonicalization")
        elif self._da3_canonical_state is DA3CanonicalRunState.CANCELLING:
            self._da3_job_value.setText("Cancelling DA3 integration…")
        elif self._da3_canonical_state is DA3CanonicalRunState.COMPLETED:
            self._da3_job_value.setText("Completed • persistent canonical scene loaded")
        elif self._da3_canonical_state is DA3CanonicalRunState.CANCELLED:
            self._da3_job_value.setText("Cancelled • previous scene preserved")
        else:
            self._da3_job_value.setText("Failed • previous scene preserved")

    def _on_da3_integrate_clicked(self) -> None:
        if self._controller is None:
            return
        if self._digital_twin_processing_snapshot.is_active:
            self.set_error("Cancel the active Digital Twin build before starting Quick DA3 integration.")
            self._refresh_da3_canonical_controls()
            return
        try:
            self._controller.start_selected_da3_canonical()
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        self._refresh_da3_canonical_controls()
        self._refresh_runtime_controls()

    def _on_da3_cancel_clicked(self) -> None:
        if self._controller is None:
            return
        try:
            self._controller.cancel_da3_canonical()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        self._refresh_da3_canonical_controls()

    def set_da3_canonical_state(self, state: DA3CanonicalRunState) -> None:
        if not isinstance(state, DA3CanonicalRunState):
            raise TypeError("state must be DA3CanonicalRunState.")
        self._da3_canonical_state = state
        self._da3_runtime_value.setText(self._humanize(state))
        self._da3_runtime_value.setProperty("statusKind", self._state_kind(state.value))
        self._refresh_style(self._da3_runtime_value)
        if state in {
            DA3CanonicalRunState.STARTING,
            DA3CanonicalRunState.RUNNING,
            DA3CanonicalRunState.CANCELLING,
        }:
            self._begin_runtime_clock("da3")
        elif state in {
            DA3CanonicalRunState.COMPLETED,
            DA3CanonicalRunState.FAILED,
            DA3CanonicalRunState.CANCELLED,
        }:
            self._freeze_runtime_clock()
        presentations = {
            DA3CanonicalRunState.STARTING: ReconstructionStatusPresentation(
                "Preparing DA3 integration",
                "Starting the isolated DA3 → canonical integration worker.",
                "running",
                indeterminate=True,
            ),
            DA3CanonicalRunState.RUNNING: ReconstructionStatusPresentation(
                "DA3 canonical integration",
                "The isolated DA3 workflow is active; metric scale remains unqualified unless separately proven.",
                "running",
                indeterminate=True,
            ),
            DA3CanonicalRunState.CANCELLING: ReconstructionStatusPresentation(
                "Cancelling DA3 integration",
                "Stopping the depth-side worker and its isolated DA3 process group.",
                "warning",
                indeterminate=True,
            ),
            DA3CanonicalRunState.COMPLETED: ReconstructionStatusPresentation(
                "DA3 canonical scene ready",
                "Persistent canonical geometry was materialized from DA3 supporting evidence without metric promotion.",
                "ready",
                progress_fraction=1.0,
            ),
            DA3CanonicalRunState.CANCELLED: ReconstructionStatusPresentation(
                "DA3 integration cancelled",
                "The DA3 integration was cancelled; the previous persistent scene remains available.",
                "warning",
            ),
            DA3CanonicalRunState.FAILED: ReconstructionStatusPresentation(
                "DA3 integration failed",
                self._last_runtime_error or "DA3 canonical integration ended with an error; the previous scene remains available.",
                "error",
            ),
        }
        presentation = presentations.get(state)
        if presentation is not None:
            self._apply_runtime_presentation(presentation)
        self._refresh_da3_canonical_controls()
        self._refresh_runtime_controls()

    def set_da3_canonical_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        event = str(payload.get("event", "")).strip()
        labels = {
            "probing_da3": "Qualifying isolated DA3 runtime…",
            "da3_qualified": "Qualified runtime • preparing source frames",
            "da3_inference_started": "DA3 inference running in eia-da3…",
            "da3_inference_completed": "DA3 evidence ready • canonicalizing…",
            "canonicalizing": "Building canonical dense/surface geometry…",
            "completed": "Completed • persistent canonical scene loaded",
            "failed": "Failed • previous scene preserved",
        }
        presentations = {
            "probing_da3": ReconstructionStatusPresentation(
                "Qualifying DA3 runtime",
                "Read-only qualification of the isolated eia-da3 runtime and compatibility lock.",
                "running",
                indeterminate=True,
            ),
            "da3_qualified": ReconstructionStatusPresentation(
                "Preparing DA3 evidence",
                "The isolated DA3 runtime is qualified; preparing a bounded source-frame burst.",
                "running",
                indeterminate=True,
            ),
            "da3_inference_started": ReconstructionStatusPresentation(
                "DA3 inference",
                "Relative depth and camera-pose evidence are being produced in the isolated eia-da3 environment.",
                "running",
                indeterminate=True,
            ),
            "da3_inference_completed": ReconstructionStatusPresentation(
                "DA3 evidence ready",
                "DA3 supporting evidence is complete; canonical dense/surface geometry is next.",
                "running",
                indeterminate=True,
            ),
            "canonicalizing": ReconstructionStatusPresentation(
                "Canonicalizing DA3 evidence",
                "Building canonical dense/surface geometry while preserving arbitrary scale and provenance.",
                "running",
                indeterminate=True,
            ),
        }
        if event in labels and hasattr(self, "_da3_job_value"):
            self._da3_job_value.setText(labels[event])
        presentation = presentations.get(event)
        if presentation is not None:
            self._apply_runtime_presentation(presentation)

    def _on_da3_canonical_result_ready(self, payload: object, _output_dir: str) -> None:
        if isinstance(payload, dict) and bool(payload.get("pass", False)):
            self._selected_dense_provider_id = "depth-anything-3-external"
            self._refresh_dense_provider_controls()
        self._refresh_da3_canonical_controls()

    def _on_da3_canonical_run_finished(self, _payload: object) -> None:
        self._refresh_da3_canonical_controls()
        self._refresh_runtime_controls()

    @staticmethod
    def _card(title_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(13, 11, 13, 11)
        layout.setSpacing(8)
        title = QLabel(title_text)
        title.setObjectName("reconstructionSectionTitle")
        layout.addWidget(title)
        return card

    def _add_scene_layer_toggle(
        self,
        layout: QVBoxLayout,
        key: str,
        label_text: str,
        tooltip: str,
        *,
        toolbar_key: str | None = None,
        callback=None,
    ) -> ToggleSwitch:
        normalized = str(key).strip()
        if not normalized or normalized in self._scene_layer_toggles:
            raise ValueError("Scene layer keys must be unique and non-empty.")
        if (toolbar_key is None) == (callback is None):
            raise ValueError("Scene layer must bind either a toolbar toggle or one canvas callback.")

        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(str(label_text).strip())
        label.setObjectName("reconstructionRowLabel")
        label.setToolTip(tooltip)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(label, 1)

        switch = ToggleSwitch(checked=True, width=32, height=18)
        switch.setObjectName("reconstructionSceneLayerToggle")
        switch.setToolTip(tooltip)
        switch.setAccessibleName(f"{label_text} scene layer")
        self._scene_layer_toggles[normalized] = switch
        if toolbar_key is not None:
            button = self._viewport_tool_buttons[toolbar_key]
            switch.setChecked(button.isChecked())
            switch.toggled.connect(button.setChecked)
            button.toggled.connect(switch.setChecked)
        else:
            switch.toggled.connect(callback)
        # Phase 9.6B keeps the unified composition projection synchronized with
        # existing visibility controls. This callback observes presentation state
        # only; it never changes layer authority or persistence.
        switch.toggled.connect(lambda _checked: self._refresh_digital_twin_scene_composition())
        row.addWidget(switch, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(row)
        return switch

    def _refresh_scene_layer_controls(self) -> None:
        if not self._scene_layer_toggles or not hasattr(self, "_reconstruction_canvas"):
            return
        canvas = self._reconstruction_canvas
        selection_active = canvas.selected_semantic_object_id is not None
        availability = build_reconstruction_scene_layer_availability(
            canonical_surface_loaded=canvas.canonical_mesh_loaded,
            semantic_objects_available=canvas.semantic_objects_available,
            canonical_semantics_loaded=canvas.canonical_semantic_loaded,
            semantic_selection_active=selection_active,
            inspection_evidence_available=(
                self._canonical_inspection_overlay_presentation is not None
                or bool(
                    self._canonical_measurement_registration_presentation
                    and self._canonical_measurement_registration_presentation.dimensions
                )
            ),
        )

        enabled = {
            "canonical_surface": availability.canonical_surface,
            "semantic_objects": availability.semantic_objects,
            "semantic_labels": availability.semantic_labels,
            "semantic_selection": availability.semantic_selection,
            "inspection_evidence": availability.inspection_evidence,
        }
        for key, state in enabled.items():
            toggle = self._scene_layer_toggles.get(key)
            if toggle is not None:
                toggle.setEnabled(state)

        # Existing live/offline context controls remain usable before a
        # persistent canonical scene is loaded.  Their toolbar buttons remain
        # the compatibility source of truth and are mirrored bidirectionally.
        for key in (
            "grid",
            "axes",
            "camera_frustums",
            "camera_trajectory",
            "sparse_landmarks",
            "sparse_envelope",
        ):
            toggle = self._scene_layer_toggles.get(key)
            if toggle is not None:
                toggle.setEnabled(True)
        self._refresh_digital_twin_workspace_presentation()

    @property
    def scene_runtime_health(self) -> ReconstructionSceneRuntimeHealth | None:
        return self._scene_runtime_health

    def _refresh_scene_runtime_health(self) -> None:
        canvas = getattr(self, "_reconstruction_canvas", None)
        scene = self._persistent_scene
        if canvas is None:
            return
        health = build_reconstruction_scene_runtime_health(
            scene_id=str(getattr(scene, "scene_id", "")).strip() if scene is not None else "",
            bind_latency_ms=self._last_scene_bind_latency_ms,
            bind_count=self._scene_bind_count,
            same_scene_reload_count=self._same_scene_reload_count,
            canonical_vertex_count=canvas.canonical_mesh_vertex_count,
            canonical_face_count=canvas.canonical_mesh_face_count,
            canonical_camera_count=canvas.canonical_camera_count,
            semantic_object_count=canvas.semantic_object_count,
            mesh_visual_reused=self._last_scene_mesh_visual_reused,
            view_preserved=self._last_scene_view_preserved,
            selection_preserved=self._last_scene_selection_preserved,
            canonical_geometry_authoritative=bool(
                getattr(scene, "canonical_geometry_authoritative", False)
            ) if scene is not None else False,
        )
        self._scene_runtime_health = health
        if not hasattr(self, "_scene_health_bind_value"):
            return
        self._scene_health_bind_value.setText(health.bind_text)
        self._scene_health_resources_value.setText(health.resources_text)
        self._scene_health_lifecycle_value.setText(health.lifecycle_text)
        self._scene_health_authority_value.setText(health.authority_text)
        self._scene_health_bind_value.setToolTip(
            "UI-side persistent-scene presentation binding time only. This is not reconstruction, DA3, GPU, or backend performance."
        )
        self._scene_health_resources_value.setToolTip(
            "Counts reflect currently materialized viewer resources after bounded presentation limits; they do not change scene authority."
        )
        self._scene_health_lifecycle_value.setToolTip(
            "9.3I preserves camera/selection on exact same-scene reloads and performs one final Fit only for a different scene."
        )
        self._scene_health_authority_value.setToolTip(
            "Scene-health diagnostics are presentation-only and never establish geometry, metric scale, measurement, QC, or certified-metrology authority."
        )

    def _record_scene_bind_runtime_health(
        self,
        *,
        started_at: float,
        same_scene_reload: bool,
        mesh_visual_reused: bool,
        view_preserved: bool,
        selection_preserved: bool,
    ) -> None:
        self._scene_bind_count += 1
        if same_scene_reload:
            self._same_scene_reload_count += 1
        self._last_scene_bind_latency_ms = max(0.0, (monotonic() - float(started_at)) * 1000.0)
        self._last_scene_mesh_visual_reused = bool(mesh_visual_reused)
        self._last_scene_view_preserved = bool(view_preserved)
        self._last_scene_selection_preserved = bool(selection_preserved)
        self._refresh_scene_runtime_health()

    def _toggle_digital_twin_workspace_details(self) -> None:
        if not hasattr(self, "_digital_twin_workspace_details"):
            return
        show = self._digital_twin_workspace_details.isHidden()
        self._digital_twin_workspace_details.setVisible(show)
        self._digital_twin_workspace_details_button.setText(
            "Hide details" if show else "Show details"
        )
        self._digital_twin_workspace_details_button.setToolTip(
            "Hide supporting workspace details."
            if show
            else "Show navigation, interaction, neural-resource and safety details for the unified workspace."
        )
        self._digital_twin_workspace_group.updateGeometry()

    def _apply_digital_twin_workspace_ux_order(self, layout: QVBoxLayout) -> None:
        """Promote unified workspace groups without removing legacy controls.

        Construction order is intentionally left untouched because older Phase
        8/9 controls and tests depend on their existing object identities and
        initialization flow. E4 only changes the final visual order inside the
        scroll body.
        """

        primary_groups = (
            self._digital_twin_workspace_group,
            self._digital_twin_processing_group,
            self._canonical_scene_group,
            self._estimated_ruler_scale_group,
        )
        for group in primary_groups:
            layout.removeWidget(group)
        for index, group in enumerate(primary_groups):
            layout.insertWidget(index, group)

        # Dense primary cards need slightly tighter content rhythm at 768p, but
        # no row/value is hidden and disclosure behavior is unchanged.
        self._digital_twin_workspace_group.content_layout.setSpacing(5)
        self._digital_twin_processing_group.content_layout.setSpacing(5)
        self._canonical_scene_group.content_layout.setSpacing(5)

    @staticmethod
    def _settings_group(title_text: str) -> ReconstructionSettingsDisclosureGroup:
        # G5 keeps the long Reconstruction sidebar fully available while making
        # detail-heavy groups collapsible. The disclosure widget is presentation
        # only: it never changes controller, persistence, runtime, or authority state.
        return ReconstructionSettingsDisclosureGroup(title_text)

    @staticmethod
    def _value_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("reconstructionValue")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _status_value() -> QLabel:
        label = QLabel()
        label.setObjectName("reconstructionStatusValue")
        return label

    @staticmethod
    def _settings_value(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("reconstructionSettingsValue")
        label.setWordWrap(True)
        # Right-side values must be allowed to wrap rather than establish a
        # horizontal minimum that forces the settings scroll body under the
        # vertical scrollbar.  Ignored affects only layout negotiation; the
        # label still receives its normal size at ordinary sidebar widths.
        label.setMinimumWidth(0)
        label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        return label

    @staticmethod
    def _add_status_row(layout: QVBoxLayout, label_text: str, value: QLabel) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("reconstructionRowLabel")
        # Give the value the row's remaining width directly.  A separate
        # expanding spacer used to consume that width first; once D4 made
        # value labels horizontally shrinkable, Qt could collapse them to
        # nearly zero width and word-wrapped labels then requested enormous
        # heights.  That produced the large blank vertical gaps seen in the
        # real Linux preview.
        row.addWidget(label, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(value, 1, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)

    def _add_settings_row(self, layout: QVBoxLayout, label_text: str, value_text: str) -> None:
        self._add_status_row(layout, label_text, self._settings_value(value_text))

    def bind_controller(self, controller) -> None:
        """Bind passive controller state. No reconstruction action is triggered here."""

        if self._controller is not None:
            raise RuntimeError("ReconstructionPage controller is already bound.")
        self._controller = controller

        controller.live_state_changed.connect(self.set_live_state)
        controller.live_observation_ready.connect(self.set_live_observation)
        controller.offline_state_changed.connect(self.set_offline_state)
        controller.offline_event_received.connect(self.set_offline_event)
        controller.offline_result_ready.connect(self._on_offline_result_ready)
        controller.offline_run_finished.connect(self._on_offline_run_finished)
        controller.da3_canonical_state_changed.connect(self.set_da3_canonical_state)
        controller.da3_canonical_event_received.connect(self.set_da3_canonical_event)
        controller.da3_canonical_result_ready.connect(self._on_da3_canonical_result_ready)
        controller.da3_canonical_run_finished.connect(self._on_da3_canonical_run_finished)
        controller.source_selection_changed.connect(self.set_source_selection)
        controller.scene_availability_changed.connect(self.set_scene_availability)
        controller.persistent_scene_changed.connect(self.set_persistent_scene)
        controller.persistent_authority_changed.connect(self.set_persistent_authority_snapshot)
        controller.checkpoint_changed.connect(self.set_checkpoint)
        controller.resume_compatibility_changed.connect(self._on_resume_compatibility_changed)
        controller.resume_context_changed.connect(self._on_resume_context_changed)
        controller.resume_preparation_changed.connect(self._on_resume_preparation_changed)
        controller.session_model_changed.connect(self.set_session_model)
        controller.error_occurred.connect(self.set_error)

        self.set_session_model(controller.session_model)
        self.set_live_state(controller.live_state)
        self.set_offline_state(controller.offline_state)
        self.set_da3_canonical_state(controller.da3_canonical_state)
        self.set_source_selection(controller.source_selection)
        self.set_scene_availability(controller.scene_availability)
        self._refresh_source_controls()
        self._refresh_runtime_controls()
        scene = controller.persistent_scene
        self.set_persistent_scene(scene, str(controller.persistent_scene_manifest or ""))
        self.set_persistent_authority_snapshot(
            controller.persistent_authority_snapshot,
            str(controller.persistent_authority_manifest or ""),
        )
        if scene is not None and controller.persistent_scene_manifest is not None:
            self._restore_digital_twin_workspace_state(controller.persistent_scene_manifest)
        self.set_checkpoint(controller.engine_checkpoint, str(controller.engine_checkpoint_manifest or ""))
        self._refresh_reconstruction_export_controls()
        self._telemetry_timer.start()
        self._refresh_runtime_telemetry()

    @property
    def reconstruction_session_model(self) -> ReconstructionSessionModel | None:
        return self._reconstruction_session_model

    def set_session_model(self, model: ReconstructionSessionModel) -> None:
        """Present the passive G1 composition; no session action is executed here."""

        if not isinstance(model, ReconstructionSessionModel):
            raise TypeError("model must be ReconstructionSessionModel.")
        previous = self._reconstruction_session_model
        self._reconstruction_session_model = model
        self._session_value.setText(model.short_session_id or "No active session")
        self._session_value.setToolTip(model.identity_details)
        self._session_context_value.setText(model.context_text)
        context_details = [
            f"Focus: {model.focus.value}",
            f"Lifecycle: {model.lifecycle.value}",
            f"Workflow: {model.workflow or 'none'}",
        ]
        if model.selected_source_label:
            context_details.append(f"Selected source: {model.selected_source_label}")
        if model.selected_source_path:
            context_details.append(f"Selected path: {model.selected_source_path}")
        self._session_context_value.setToolTip("\n".join(context_details))
        self._session_evidence_value.setText(model.evidence_text)
        self._session_evidence_value.setToolTip(model.evidence_details)
        self._scene_manifest_value.setText(
            Path(model.scene_manifest).name
            if model.scene_manifest
            else "Persistent scene linked"
            if model.scene_id
            else "No persistent scene"
        )
        self._scene_manifest_value.setToolTip(model.scene_manifest or model.scene_id)
        self._checkpoint_value.setText(
            Path(model.checkpoint_manifest).name
            if model.checkpoint_manifest
            else "Checkpoint linked"
            if model.checkpoint_id
            else "No checkpoint"
        )
        self._checkpoint_value.setToolTip(model.checkpoint_manifest or model.checkpoint_id)
        status_kind = (
            "warning"
            if model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT
            else "ready"
            if model.has_session
            else "neutral"
        )
        for widget in (self._session_value, self._session_context_value):
            widget.setProperty("statusKind", status_kind)
            self._refresh_style(widget)
        self._refresh_scene_interaction_controls()
        self._refresh_checkpoint_resume_controls()
        self._refresh_failure_recovery_controls()
        self._refresh_reconstruction_export_controls()
        if model != previous:
            self.reconstruction_session_model_changed.emit(model)

    @property
    def scene_interaction_presentation(self) -> ReconstructionSceneInteractionPresentation | None:
        return self._scene_interaction_presentation

    def _refresh_scene_interaction_controls(self) -> None:
        if not hasattr(self, "_session_load_button"):
            return
        controller = self._controller
        presentation = build_reconstruction_scene_interaction_presentation(
            self._reconstruction_session_model,
            controller_active=bool(controller and controller.is_active),
            controller_shutdown=bool(controller is None or controller.is_shutdown),
        )
        self._scene_interaction_presentation = presentation
        self._session_action_state_value.setText(presentation.text)
        self._session_action_state_value.setToolTip(presentation.details)
        self._session_action_state_value.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._session_action_state_value)
        self._session_load_button.setEnabled(presentation.can_load)
        self._session_save_button.setEnabled(presentation.can_save)
        self._session_reload_button.setEnabled(presentation.can_reload)
        self._refresh_reconstruction_export_controls()

    @property
    def last_reconstruction_export_dir(self) -> Path | None:
        return self._last_reconstruction_export_dir

    def set_reconstruction_canonical_surface_export_source(
        self,
        source: CanonicalSurfaceMeshExportSource | None,
    ) -> None:
        """Bind an already-qualified, materialized canonical surface for product export.

        A2 intentionally accepts the A1 source contract rather than reconstructing a
        ``SurfaceMeshResult`` from persistent-scene metadata. Persistent manifests do
        not contain full mesh arrays, so a loaded scene must never manufacture a mesh
        export that is not actually materialized in the current application session.
        """

        if source is not None and not isinstance(source, CanonicalSurfaceMeshExportSource):
            raise TypeError("source must be CanonicalSurfaceMeshExportSource or None.")
        self._canonical_surface_export_source = source
        self._refresh_reconstruction_export_controls()

    def _qualified_reconstruction_export_sources(
        self,
    ) -> tuple[ReconstructionExportSource, ...]:
        sources = tuple(
            source
            for source in (
                self._offline_sparse_export_source,
                self._canonical_surface_export_source,
                self._inspection_qc_export_source,
            )
            if source is not None
        )
        if not sources:
            return ()
        session_ids = {str(getattr(source, "session_id", "")).strip() for source in sources}
        if "" in session_ids or len(session_ids) != 1:
            return ()
        if self._inspection_qc_export_source is not None:
            selection = self._inspection_selection
            active_session = (
                str(selection.session_id or selection.scene_id).strip()
                if selection.has_scene
                else ""
            )
            if not active_session or active_session != self._inspection_qc_export_source.session_id:
                return ()
        model = self._reconstruction_session_model
        if model is not None:
            if model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT:
                return ()
            if model.session_id and model.session_id not in session_ids:
                return ()
        return sources

    def _refresh_reconstruction_export_controls(self) -> None:
        if not hasattr(self, "_reconstruction_export_button"):
            return
        controller = self._controller
        active = bool(controller and controller.is_active) or self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        } or self._offline_state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        }
        shutdown = bool(controller is None or controller.is_shutdown)
        raw_sources = tuple(
            source
            for source in (
                self._offline_sparse_export_source,
                self._canonical_surface_export_source,
                self._inspection_qc_export_source,
            )
            if source is not None
        )
        sources = self._qualified_reconstruction_export_sources()
        status_kind = "neutral"
        details = (
            "Product export is separate from persistent-scene Save/Reload and never changes "
            "reconstruction authority."
        )
        if shutdown:
            text = "Export unavailable"
        elif active:
            text = "Stop reconstruction before exporting"
            status_kind = "running"
        elif raw_sources and not sources:
            text = "Export evidence session mismatch"
            status_kind = "warning"
            details = (
                "Materialized export evidence does not resolve to one non-conflicting current "
                "session, so export is blocked fail-closed."
            )
        elif sources:
            has_sparse = any(isinstance(source, OfflineSparseExportSource) for source in sources)
            has_canonical = any(
                isinstance(source, CanonicalSurfaceMeshExportSource) for source in sources
            )
            has_qc = any(isinstance(source, InspectionReportExportSource) for source in sources)
            if has_sparse and has_canonical and has_qc:
                text = "Canonical mesh + sparse SfM + QC ready"
                details = (
                    "Canonical mesh exports with canonical-geometry authority; sparse SfM remains "
                    "supporting geometry with arbitrary scale; QC exports only the attached "
                    "inspection-decision report. Certified metrology remains false."
                )
            elif has_canonical and has_qc:
                text = "Canonical mesh + QC report ready"
                details = (
                    "Materialized canonical surface and the exact authoritative inspection/QC report "
                    "are ready for one bundle. QC does not become geometry or metrology authority."
                )
            elif has_sparse and has_qc:
                text = "Sparse SfM + QC report ready"
                details = (
                    "Sparse SfM remains supporting arbitrary-scale geometry; QC exports only the "
                    "attached inspection-decision report. Certified metrology remains false."
                )
            elif has_sparse and has_canonical:
                text = "Canonical mesh + sparse SfM ready"
                details = (
                    "Canonical mesh exports with canonical-geometry authority; sparse SfM remains "
                    "supporting geometry with arbitrary scale. Certified metrology remains false."
                )
            elif has_canonical:
                text = "Canonical mesh ready"
                details = (
                    "Materialized canonical surface is available for authoritative geometry export. "
                    "Certified metrology remains false."
                )
            elif has_qc:
                text = "QC report ready"
                details = (
                    "The exact attached Phase 8.7C inspection report is bound to inspection-decision "
                    "authority and can be exported without rerunning QC. Certified metrology "
                    "remains false."
                )
            else:
                text = "Sparse SfM ready • arbitrary scale"
                details = (
                    "Completed sparse SfM can be exported as supporting geometry only. It is "
                    "arbitrary-scale and is not canonical geometry or certified metrology."
                )
            status_kind = "ready"
        elif self._reconstruction_export_source_error:
            text = "Result is not export-qualified"
            status_kind = "warning"
            details = self._reconstruction_export_source_error
        else:
            text = "No materialized export evidence"
        self._reconstruction_export_state_value.setText(text)
        self._reconstruction_export_state_value.setToolTip(details)
        self._reconstruction_export_state_value.setProperty("statusKind", status_kind)
        self._refresh_style(self._reconstruction_export_state_value)
        self._reconstruction_export_button.setEnabled(bool(sources) and not active and not shutdown)
        self._reconstruction_export_open_button.setEnabled(
            bool(self._last_reconstruction_export_dir and self._last_reconstruction_export_dir.is_dir())
        )

    def _export_reconstruction_bundle_interaction(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown or controller.is_active:
            self._refresh_reconstruction_export_controls()
            return
        sources = self._qualified_reconstruction_export_sources()
        if not sources:
            self._refresh_reconstruction_export_controls()
            return
        start_dir = (
            self._last_reconstruction_export_dir.parent
            if self._last_reconstruction_export_dir is not None
            else Path.home()
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Reconstruction Export Parent Folder",
            str(start_dir),
        )
        if not selected:
            return
        parent = Path(selected).expanduser().resolve()
        session_id = str(getattr(sources[0], "session_id", "")).strip()
        target = self._next_reconstruction_export_dir(parent, session_id)
        try:
            result = export_reconstruction_bundle(
                ReconstructionExportRequest(output_dir=target, sources=sources)
            )
            verified = verify_reconstruction_export_bundle(result.export_dir)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._reconstruction_export_state_value.setText("Export failed")
            self._reconstruction_export_state_value.setToolTip(str(exc))
            self._reconstruction_export_state_value.setProperty("statusKind", "warning")
            self._refresh_style(self._reconstruction_export_state_value)
            self._reconstruction_export_button.setEnabled(True)
            return
        self._last_reconstruction_export_dir = result.export_dir
        canonical = any(item.canonical_geometry_authoritative for item in verified)
        has_qc = any(
            item.authority.value == "inspection_decision" for item in verified
        )
        has_supporting = any(
            item.authority.value == "supporting_geometry" for item in verified
        )
        if canonical and has_qc:
            authority = "canonical geometry + inspection/QC included"
        elif canonical:
            authority = "canonical geometry included"
        elif has_supporting and has_qc:
            authority = "supporting geometry + inspection/QC included"
        elif has_qc:
            authority = "inspection/QC evidence included"
        else:
            authority = "supporting geometry only"
        self._reconstruction_export_state_value.setText(
            f"Export complete • {len(verified)} files"
        )
        self._reconstruction_export_state_value.setToolTip(
            f"{result.export_dir}\n{authority} • certified metrology: false"
        )
        self._reconstruction_export_state_value.setProperty("statusKind", "ready")
        self._refresh_style(self._reconstruction_export_state_value)
        self._reconstruction_export_open_button.setEnabled(True)

    def _open_reconstruction_export_interaction(self) -> None:
        path = self._last_reconstruction_export_dir
        if path is None or not path.is_dir():
            self._reconstruction_export_open_button.setEnabled(False)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _next_reconstruction_export_dir(parent: Path, session_id: str) -> Path:
        root = Path(parent).expanduser().resolve()
        short_session = str(session_id).strip().replace("-", "")[:8] or "session"
        base = root / f"reconstruction_export_{short_session}"
        if not base.exists():
            return base
        for index in range(2, 1000):
            candidate = root / f"reconstruction_export_{short_session}_{index:02d}"
            if not candidate.exists():
                return candidate
        raise RuntimeError("Could not allocate a unique reconstruction export folder.")

    def _capture_digital_twin_workspace_state(self) -> DigitalTwinWorkspaceState | None:
        scene = self._persistent_scene
        if scene is None:
            return None
        snapshot = self._digital_twin_processing_snapshot
        terminal_snapshot = (
            snapshot
            if snapshot.lifecycle.is_terminal and snapshot.spec is not None
            else None
        )
        return build_digital_twin_workspace_state(
            scene,
            estimated_metric_scale_hint=self._estimated_metric_scale_hint,
            ruler_metric_record=self._ruler_metric_record,
            selected_appearance_artifact_id=self._selected_appearance_artifact_id,
            processing_history=self._digital_twin_processing_history,
            processing_snapshot=terminal_snapshot,
        )

    def _persist_digital_twin_workspace_state(self, scene_manifest: Path) -> Path | None:
        state = self._capture_digital_twin_workspace_state()
        if state is None:
            return None
        manifest = write_digital_twin_workspace_state(state, scene_manifest.parent)
        self._digital_twin_workspace_state = state
        self._digital_twin_workspace_manifest = manifest
        self._refresh_digital_twin_workspace_persistence_controls()
        return manifest

    def _restore_digital_twin_workspace_state(self, scene_manifest: Path) -> None:
        scene = self._persistent_scene
        if scene is None:
            return
        try:
            state, manifest = read_optional_digital_twin_workspace_state(
                scene_manifest,
                expected_scene=scene,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            # Workspace state is explicitly non-authoritative. A damaged sidecar
            # must never hide an otherwise valid canonical persistent scene.
            self._digital_twin_workspace_state = None
            self._digital_twin_workspace_manifest = None
            self._digital_twin_processing_history = None
            self._refresh_digital_twin_processing_history_presentation()
            if hasattr(self, "_digital_twin_workspace_state_value"):
                self._digital_twin_workspace_state_value.setText("Twin state rejected")
                self._digital_twin_workspace_state_value.setToolTip(str(exc))
                self._digital_twin_workspace_state_value.setProperty("statusKind", "warning")
                self._refresh_style(self._digital_twin_workspace_state_value)
            return
        if state is None:
            self._digital_twin_workspace_state = None
            self._digital_twin_workspace_manifest = None
            self._digital_twin_processing_history = None
            self._refresh_digital_twin_processing_history_presentation()
            if hasattr(self, "_digital_twin_workspace_state_value"):
                self._digital_twin_workspace_state_value.setText("Legacy scene • no twin sidecar")
                self._digital_twin_workspace_state_value.setToolTip(
                    "The canonical scene loaded normally. No Phase 9.6F workspace sidecar was present."
                )
                self._digital_twin_workspace_state_value.setProperty("statusKind", "neutral")
                self._refresh_style(self._digital_twin_workspace_state_value)
            self._refresh_digital_twin_workspace_persistence_controls()
            return
        self._apply_digital_twin_workspace_state(state)
        self._digital_twin_workspace_manifest = manifest
        self._refresh_digital_twin_workspace_persistence_controls()

    def _apply_digital_twin_workspace_state(self, state: DigitalTwinWorkspaceState) -> None:
        scene = self._persistent_scene
        if scene is None:
            raise ValueError("A persistent canonical scene is required before restoring workspace state.")
        state.validate_against_scene(scene)
        self._digital_twin_workspace_state = state
        self._digital_twin_processing_history = state.processing_history

        if state.estimated_metric_scale_hint is None:
            self.clear_estimated_metric_scale_hint()
        else:
            self.set_estimated_metric_scale_hint(state.estimated_metric_scale_hint)

        selected = state.selected_appearance_artifact_id
        if selected and hasattr(self, "_appearance_representation_combo"):
            index = self._appearance_representation_combo.findData(selected)
            if index >= 0:
                self._selected_appearance_artifact_id = selected
                self._appearance_representation_combo.blockSignals(True)
                self._appearance_representation_combo.setCurrentIndex(index)
                self._appearance_representation_combo.blockSignals(False)
                self._refresh_appearance_representation_values()
                self._refresh_gaussian_visualization_controls()
                self._refresh_nerf_visualization_controls()
                self._refresh_representation_switching_controls()

        saved_ruler = state.ruler_metric_record
        if saved_ruler is None:
            self._interactive_measure_points_reconstruction = []
            self._interactive_distance_estimate = None
            self._ruler_metric_record = None
            self._last_ruler_reference_distance_reconstruction_units = None
            self._reconstruction_canvas.clear_interactive_measurement_overlay()
            self._interactive_measure_hud.hide()
        else:
            points = [
                np.asarray(saved_ruler.point_a_reconstruction, dtype=np.float64),
                np.asarray(saved_ruler.point_b_reconstruction, dtype=np.float64),
            ]
            self._interactive_measure_points_reconstruction = points
            self._reconstruction_canvas.set_interactive_measurement_overlay(
                np.asarray(points, dtype=np.float64)
            )
            # Recompute the best *current* readout against today's persisted
            # authority while carrying the saved original estimate as provenance.
            # A later qualified scale may therefore supersede the visible value
            # without erasing the original Phase 9.6E0 estimate.
            restored = build_reconstruction_ruler_metric_record(
                scene_id=saved_ruler.scene_id,
                point_a_reconstruction=saved_ruler.point_a_reconstruction,
                point_b_reconstruction=saved_ruler.point_b_reconstruction,
                canonical_scene=self._canonical_scene_presentation,
                estimated_scale_hint=self._estimated_metric_scale_hint,
                previous=saved_ruler,
            )
            self._last_ruler_reference_distance_reconstruction_units = (
                restored.current.distance_source_units
            )
            self._apply_ruler_metric_record(restored)
            self._interactive_measure_hud.show()

        self._refresh_estimated_metric_scale_presentation()
        self._refresh_digital_twin_processing_history_presentation()
        if hasattr(self, "_digital_twin_workspace_state_value"):
            self._digital_twin_workspace_state_value.setText("Workspace state restored")
            self._digital_twin_workspace_state_value.setToolTip(
                f"{state.scene_id}\n"
                f"Workspace fingerprint: {state.workspace_fingerprint_sha256}\n"
                "Estimated-scale/ruler provenance + appearance selection + terminal build history restored.\n"
                "No active build job was resumed and no authority was changed."
            )
            self._digital_twin_workspace_state_value.setProperty("statusKind", "ready")
            self._refresh_style(self._digital_twin_workspace_state_value)

    def _refresh_digital_twin_workspace_persistence_controls(self) -> None:
        if not hasattr(self, "_digital_twin_state_export_button"):
            return
        controller = self._controller
        scene_ready = bool(
            controller is not None
            and controller.persistent_scene is not None
            and controller.persistent_scene_manifest is not None
        )
        busy = bool(
            controller is None
            or controller.is_shutdown
            or controller.is_active
            or self._digital_twin_processing_snapshot.is_active
        )
        self._digital_twin_state_export_button.setEnabled(scene_ready and not busy)
        self._digital_twin_state_export_open_button.setEnabled(
            bool(
                self._last_digital_twin_state_export_dir
                and self._last_digital_twin_state_export_dir.is_dir()
            )
        )
        if self._persistent_scene is None:
            self._digital_twin_workspace_state_value.setText("No Digital Twin workspace sidecar")
            self._digital_twin_workspace_state_value.setToolTip(
                "Load a persistent canonical scene before saving/restoring Phase 9.6F workspace state."
            )
            self._digital_twin_workspace_state_value.setProperty("statusKind", "neutral")
            self._refresh_style(self._digital_twin_workspace_state_value)
        elif self._digital_twin_workspace_manifest is not None:
            self._digital_twin_workspace_state_value.setText("Workspace state persisted")
            self._digital_twin_workspace_state_value.setToolTip(
                str(self._digital_twin_workspace_manifest)
            )
            self._digital_twin_workspace_state_value.setProperty("statusKind", "ready")
            self._refresh_style(self._digital_twin_workspace_state_value)

    def _export_digital_twin_state_interaction(self) -> None:
        controller = self._controller
        if (
            controller is None
            or controller.is_shutdown
            or controller.is_active
            or self._digital_twin_processing_snapshot.is_active
            or controller.persistent_scene is None
            or controller.persistent_scene_manifest is None
        ):
            self._refresh_digital_twin_workspace_persistence_controls()
            return
        state = self._capture_digital_twin_workspace_state()
        if state is None:
            return
        start_dir = (
            self._last_digital_twin_state_export_dir.parent
            if self._last_digital_twin_state_export_dir is not None
            else controller.persistent_scene_manifest.parent
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Digital Twin State Export Parent Folder",
            str(start_dir),
        )
        if not selected:
            return
        parent = Path(selected).expanduser().resolve()
        target = self._next_digital_twin_state_export_dir(parent, state.scene_id)
        authority_manifest = controller.persistent_authority_manifest
        try:
            export_manifest = export_digital_twin_state_bundle(
                scene_manifest_path=controller.persistent_scene_manifest,
                workspace_state=state,
                authority_manifest_path=authority_manifest,
                output_dir=target,
            )
            verified = verify_digital_twin_state_bundle(export_manifest.parent)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._digital_twin_workspace_state_value.setText("State export failed")
            self._digital_twin_workspace_state_value.setToolTip(str(exc))
            self._digital_twin_workspace_state_value.setProperty("statusKind", "warning")
            self._refresh_style(self._digital_twin_workspace_state_value)
            return
        self._last_digital_twin_state_export_dir = export_manifest.parent
        self._refresh_digital_twin_workspace_persistence_controls()
        self._digital_twin_workspace_state_value.setText("Twin state export complete")
        self._digital_twin_workspace_state_value.setToolTip(
            f"{export_manifest.parent}\n"
            f"Referenced payloads: {verified.get('referenced_payload_count', 0)} • not embedded\n"
            "State/audit export only • Product export remains the geometry/QC payload route."
        )
        self._digital_twin_workspace_state_value.setProperty("statusKind", "ready")
        self._refresh_style(self._digital_twin_workspace_state_value)

    def _open_digital_twin_state_export_interaction(self) -> None:
        path = self._last_digital_twin_state_export_dir
        if path is None or not path.is_dir():
            self._digital_twin_state_export_open_button.setEnabled(False)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _next_digital_twin_state_export_dir(parent: Path, scene_id: str) -> Path:
        root = Path(parent).expanduser().resolve()
        short_scene = str(scene_id).strip().replace("prs-", "")[:8] or "scene"
        base = root / f"digital_twin_state_{short_scene}"
        if not base.exists():
            return base
        for index in range(2, 1000):
            candidate = root / f"digital_twin_state_{short_scene}_{index:02d}"
            if not candidate.exists():
                return candidate
        raise RuntimeError("Could not allocate a unique Digital Twin state-export folder.")

    def _load_persistent_scene_interaction(self) -> None:
        controller = self._controller
        if controller is None:
            return
        current = controller.persistent_scene_manifest
        start_dir = str(current.parent if current is not None else Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load Persistent Reconstruction Scene",
            start_dir,
            (
                f"Persistent reconstruction scene ({PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST});;"
                "JSON files (*.json);;All files (*)"
            ),
        )
        if not selected:
            return
        try:
            selected_manifest = Path(selected).expanduser().resolve()
            controller.load_persistent_scene(
                selected_manifest,
                verify_file_references=True,
            )
            active_manifest = controller.persistent_scene_manifest
            if active_manifest is not None:
                self._restore_digital_twin_workspace_state(active_manifest)
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.set_error(f"Could not load persistent reconstruction scene: {exc}")
        finally:
            self._refresh_scene_interaction_controls()

    def _save_persistent_scene_interaction(self) -> None:
        controller = self._controller
        if controller is None:
            return
        current = controller.persistent_scene_manifest
        start_dir = str(current.parent if current is not None else Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Save Persistent Reconstruction Scene",
            start_dir,
        )
        if not selected:
            return
        output_dir = Path(selected).expanduser().resolve()
        target_manifest = output_dir / PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST
        current_manifest = current.resolve() if current is not None else None
        if target_manifest.exists() and target_manifest.resolve() != current_manifest:
            response = QMessageBox.question(
                self,
                "Replace Persistent Reconstruction Scene",
                (
                    "The selected folder already contains a persistent reconstruction scene. "
                    "Replace its manifest and checksum? Referenced artifact payload files are not copied."
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        try:
            persisted_calibration = (
                None
                if self._persistent_authority_snapshot is None
                else self._persistent_authority_snapshot.camera_calibration_profile
            )
            if self._selected_calibration_profile is None and persisted_calibration is None:
                # Preserve the Phase 8 G2 one-argument save boundary when there is
                # no calibration authority to add or remove. This keeps legacy
                # callers/monkeypatches compatible while the controller continues
                # to preserve any existing metric-only authority sidecar itself.
                saved_manifest = controller.save_persistent_scene(output_dir)
            else:
                # Passing None explicitly is intentional when a loaded scene had a
                # persisted calibration that the user has since cleared: the new
                # sidecar must remove that calibration instead of silently retaining it.
                saved_manifest = controller.save_persistent_scene(
                    output_dir,
                    camera_calibration_profile=self._selected_calibration_profile,
                )
            # Legacy tests/callers may monkeypatch the save seam without returning
            # a manifest. Real controller saves always return the durable path.
            if isinstance(saved_manifest, Path):
                self._persist_digital_twin_workspace_state(saved_manifest.resolve())
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.set_error(f"Could not save persistent reconstruction scene: {exc}")
        finally:
            self._refresh_scene_interaction_controls()

    def _reload_persistent_scene_interaction(self) -> None:
        controller = self._controller
        if controller is None:
            return
        try:
            controller.reload_persistent_scene()
            active_manifest = controller.persistent_scene_manifest
            if active_manifest is not None:
                self._restore_digital_twin_workspace_state(active_manifest)
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.set_error(f"Could not reload persistent reconstruction scene: {exc}")
        finally:
            self._refresh_scene_interaction_controls()

    @property
    def checkpoint_resume_presentation(self) -> ReconstructionCheckpointResumePresentation | None:
        return self._checkpoint_resume_presentation

    def _refresh_checkpoint_resume_controls(self) -> None:
        if not hasattr(self, "_session_checkpoint_load_button"):
            return
        controller = self._controller
        presentation = build_reconstruction_checkpoint_resume_presentation(
            self._reconstruction_session_model,
            checkpoint=controller.engine_checkpoint if controller is not None else None,
            compatibility=controller.resume_compatibility if controller is not None else None,
            context_available=bool(controller and controller.engine_resume_context is not None),
            prepared=bool(controller and controller.engine_resume_preparation is not None),
            controller_active=bool(controller and controller.is_active),
            controller_shutdown=bool(controller is None or controller.is_shutdown),
        )
        self._checkpoint_resume_presentation = presentation
        self._session_resume_state_value.setText(presentation.text)
        self._session_resume_state_value.setToolTip(presentation.details)
        self._session_resume_state_value.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._session_resume_state_value)
        self._session_checkpoint_load_button.setEnabled(presentation.can_load_checkpoint)
        self._session_resume_button.setEnabled(presentation.can_prepare_resume)

    def _load_engine_checkpoint_interaction(self) -> None:
        controller = self._controller
        if controller is None:
            return
        current = controller.engine_checkpoint_manifest
        start_dir = str(current.parent if current is not None else Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load Reconstruction Checkpoint",
            start_dir,
            (
                f"Reconstruction checkpoint ({ENGINE_CHECKPOINT_MANIFEST});;"
                "JSON files (*.json);;All files (*)"
            ),
        )
        if not selected:
            return
        try:
            controller.load_engine_checkpoint(Path(selected).expanduser().resolve())
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.set_error(f"Could not load reconstruction checkpoint: {exc}")
        finally:
            self._refresh_checkpoint_resume_controls()

    def _prepare_engine_resume_interaction(self) -> None:
        controller = self._controller
        if controller is None:
            return
        try:
            controller.prepare_loaded_engine_resume()
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.set_error(f"Could not prepare reconstruction resume: {exc}")
        finally:
            self._refresh_checkpoint_resume_controls()

    def _on_resume_compatibility_changed(self, _compatibility: object) -> None:
        self._refresh_checkpoint_resume_controls()
        self._refresh_failure_recovery_controls()

    def _on_resume_context_changed(self, _context: object) -> None:
        self._refresh_checkpoint_resume_controls()
        self._refresh_failure_recovery_controls()

    def _on_resume_preparation_changed(self, _preparation: object) -> None:
        self._refresh_checkpoint_resume_controls()
        self._refresh_failure_recovery_controls()

    @property
    def failure_recovery_presentation(self) -> ReconstructionFailureRecoveryPresentation | None:
        return self._failure_recovery_presentation

    def _refresh_failure_recovery_controls(self) -> None:
        if not hasattr(self, "_session_retry_button"):
            return
        controller = self._controller
        stats = controller.live_stats if controller is not None else None
        source_ready = bool(controller and controller.source_selection.ready)
        presentation = build_reconstruction_failure_recovery_presentation(
            self._reconstruction_session_model,
            live_state=self._live_state,
            offline_state=self._offline_state,
            live_stats=stats,
            source_ready=source_ready,
            last_error=self._last_runtime_error,
            checkpoint_resume=self._checkpoint_resume_presentation,
            controller_active=bool(controller and controller.is_active),
            controller_shutdown=bool(controller is None or controller.is_shutdown),
        )
        previous = self._failure_recovery_presentation
        self._failure_recovery_presentation = presentation
        self._session_recovery_state_value.setText(presentation.text)
        self._session_recovery_state_value.setToolTip(presentation.details)
        self._session_recovery_state_value.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._session_recovery_state_value)
        self._session_retry_button.setEnabled(presentation.can_retry_from_source)
        if presentation != previous:
            self.failure_recovery_changed.emit(presentation)

    def _retry_failed_runtime_interaction(self) -> None:
        presentation = self._failure_recovery_presentation
        if presentation is None or not presentation.can_retry_from_source:
            return
        # G4 deliberately reuses the existing start path. This is a fresh attempt
        # from the selected source, not checkpoint resume and not an unbounded manual
        # restart of an already-recovering live worker.
        self._start_runtime()
        self._refresh_failure_recovery_controls()

    def set_source_selection(self, selection: ReconstructionSourceSelection) -> None:
        if not isinstance(selection, ReconstructionSourceSelection):
            raise TypeError("selection must be ReconstructionSourceSelection.")
        workflow = selection.workflow
        self._source_value.setText(selection.display_name)
        self._source_value.setToolTip(str(selection.path or selection.detail))
        self._workflow_value.setText(self._workflow_label(workflow))
        self._workflow_value.setToolTip(selection.detail)
        if workflow is ReconstructionWorkflow.NONE:
            readiness_text = "Not selected"
            readiness_kind = "neutral"
        elif selection.ready:
            readiness_text = "Ready"
            readiness_kind = "ready"
        else:
            readiness_text = "Needs attention"
            readiness_kind = "warning"
        self._source_readiness_value.setText(readiness_text)
        self._source_readiness_value.setProperty("statusKind", readiness_kind)
        self._refresh_style(self._source_readiness_value)
        for candidate, button in self._source_buttons.items():
            button.setChecked(candidate is workflow)
        if workflow is ReconstructionWorkflow.LIVE_SOURCE:
            identity = (
                selection.live_source_kind or "",
                selection.display_name,
                str(selection.path or ""),
            )
            if (
                self._last_live_source_identity is not None
                and identity != self._last_live_source_identity
            ):
                self._reset_live_camera_selection()
            self._last_live_source_identity = identity
        elif workflow is not self._last_workflow:
            self._reset_live_camera_selection()
            self._last_live_source_identity = None
        self._last_workflow = workflow
        self._sync_diagnostics_visibility()
        if workflow is ReconstructionWorkflow.IMAGE_SEQUENCE and selection.item_count is not None:
            self._workflow_input_value.setText(f"{selection.item_count} images")
        elif workflow is ReconstructionWorkflow.RECORDED_VIDEO:
            self._workflow_input_value.setText(selection.display_name)
        elif workflow is ReconstructionWorkflow.LIVE_SOURCE:
            self._workflow_input_value.setText(selection.display_name)
        else:
            self._workflow_input_value.setText("No source selected")
        self._workflow_input_value.setToolTip(selection.detail)
        if workflow is ReconstructionWorkflow.LIVE_SOURCE:
            self._refresh_live_camera_presentation()
        elif workflow in {ReconstructionWorkflow.IMAGE_SEQUENCE, ReconstructionWorkflow.RECORDED_VIDEO}:
            self._runtime_prerequisite_value.setText("Isolated PyCOLMAP worker")
            self._runtime_prerequisite_value.setToolTip(
                "The production depth environment launches PyCOLMAP in its isolated interpreter."
            )
        else:
            self._runtime_prerequisite_value.setText("Choose a workflow")
            self._runtime_prerequisite_value.setToolTip("")
            self._refresh_camera_authority()
        preserve_live_terminal = (
            workflow is ReconstructionWorkflow.LIVE_SOURCE
            and self._live_state in {
                LiveCaptureRuntimeState.STOPPED,
                LiveCaptureRuntimeState.FAILED,
            }
        )
        if (
            (self._controller is None or not self._controller.is_active)
            and not preserve_live_terminal
        ):
            if selection.ready:
                self._apply_runtime_presentation(
                    ReconstructionStatusPresentation(
                        "Ready to start",
                        "Source validated. Start is available when application resource guards are clear.",
                        "ready",
                    )
                )
            elif workflow is ReconstructionWorkflow.NONE:
                self._apply_runtime_presentation(
                    ReconstructionStatusPresentation(
                        "Waiting for source",
                        "Choose a Reconstruction workflow to begin.",
                    )
                )
            else:
                self._apply_runtime_presentation(
                    ReconstructionStatusPresentation(
                        "Source needs attention",
                        selection.detail,
                        "warning",
                    )
                )
        self._refresh_da3_canonical_controls()
        self._refresh_da3_full_scene_controls()
        self._refresh_runtime_controls()

    def _select_live_source(self) -> None:
        if self._controller is None:
            return
        try:
            self._controller.select_live_source()
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))

    def _choose_image_sequence(self) -> None:
        if self._controller is None:
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Reconstruction Image Sequence",
            str(Path.home()),
        )
        if not selected:
            self.set_source_selection(self._controller.source_selection)
            return
        try:
            self._controller.select_image_sequence(selected)
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_source_selection(self._controller.source_selection)
            self.set_error(str(exc))

    def _choose_recorded_video(self) -> None:
        if self._controller is None:
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reconstruction Video",
            str(Path.home()),
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)",
        )
        if not selected:
            self.set_source_selection(self._controller.source_selection)
            return
        try:
            self._controller.select_recorded_video(selected)
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_source_selection(self._controller.source_selection)
            self.set_error(str(exc))

    def _start_runtime(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown:
            return
        if self._digital_twin_processing_snapshot.is_active:
            self.set_error("Cancel the active Digital Twin build before starting reconstruction.")
            self._refresh_runtime_controls()
            return
        selection = controller.source_selection
        if not selection.ready:
            self.set_error(selection.detail or "Select a ready reconstruction source first.")
            self._refresh_runtime_controls()
            return
        try:
            self._release_appearance_resources_for_heavy_handoff("Reconstruction")
            self._prepare_runtime_presentation_for_start(selection.workflow)
            if selection.workflow is ReconstructionWorkflow.LIVE_SOURCE:
                prepared = self._prepare_live_camera_for_start()
                if prepared is None:
                    self._refresh_runtime_controls()
                    return
                camera, policy = prepared
                if policy is CameraIntrinsicsPolicy.CALIBRATED_ONLY:
                    session_id = controller.start_selected_live(camera=camera)
                else:
                    session_id = controller.start_selected_live(
                        camera=camera,
                        camera_intrinsics_policy=policy,
                    )
            elif selection.workflow in {
                ReconstructionWorkflow.IMAGE_SEQUENCE,
                ReconstructionWorkflow.RECORDED_VIDEO,
            }:
                session_id = controller.start_selected_offline()
            else:
                raise RuntimeError("Choose a Reconstruction workflow before starting.")
            self._session_value.setText(str(session_id)[:8])
            self._session_value.setToolTip(str(session_id))
            output_dir = controller.offline_output_dir
            if output_dir is not None:
                self._runtime_output_value.setText(output_dir.name)
                self._runtime_output_value.setToolTip(str(output_dir))
            elif selection.workflow is ReconstructionWorkflow.LIVE_SOURCE:
                self._runtime_output_value.setText("Live session")
                self._runtime_output_value.setToolTip(
                    "Live reconstruction has no offline output directory yet."
                )
            else:
                self._runtime_output_value.setText("Preparing output")
                self._runtime_output_value.setToolTip("")
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        finally:
            self._refresh_source_controls()
            self._refresh_runtime_controls()
            self._refresh_digital_twin_resource_coordination()

    @property
    def selected_camera_calibration_profile(self) -> CameraCalibrationProfile | None:
        return self._selected_calibration_profile

    def set_camera_calibration_profile(self, profile: CameraCalibrationProfile) -> None:
        """Select one reusable Phase 9 calibration profile without starting a backend."""

        if not isinstance(profile, CameraCalibrationProfile):
            raise TypeError("profile must be a CameraCalibrationProfile.")
        self._selected_calibration_profile = profile
        self._live_camera = None
        self._live_calibration_path = None
        self._live_calibration_application = None
        if self._live_camera_mode == "calibrated":
            self._live_camera_mode = None
        self._refresh_camera_authority()
        self._refresh_live_camera_presentation()
        self.camera_calibration_profile_changed.emit(profile)

    def clear_camera_calibration_profile(self) -> None:
        """Clear only the active Phase 9 selection; the reusable store is untouched."""

        self._selected_calibration_profile = None
        self._live_camera = None
        self._live_calibration_path = None
        self._live_calibration_application = None
        if self._live_camera_mode == "calibrated":
            self._live_camera_mode = None
        self._refresh_camera_authority()
        self._refresh_live_camera_presentation()
        self.camera_calibration_profile_changed.emit(None)

    def _camera_calibration_dialog_context(self) -> tuple[str, str]:
        source_label = "No source selected"
        resolution_text = "Resolution unavailable"
        controller = self._controller
        if controller is None:
            return source_label, resolution_text
        source_label = controller.current_application_source_label()
        try:
            size = controller.current_application_image_size()
            resolution_text = f"{size.width} × {size.height}"
        except (RuntimeError, TypeError, ValueError):
            resolution_text = "Waiting for source frame"
        return source_label, resolution_text

    def open_camera_calibration_workspace(self) -> None:
        """Open the shared application calibration workspace.

        Measure & Inspect and Reconstruction intentionally reuse this same profile
        selection/store rather than creating parallel calibration state.
        """

        self._open_camera_calibration_workspace()

    def _open_camera_calibration_workspace(self) -> None:
        controller = self._controller
        if controller is not None and controller.is_active:
            QMessageBox.information(
                self,
                "Camera Calibration",
                "Stop the active reconstruction runtime before changing camera calibration.",
            )
            return
        source_label, resolution_text = self._camera_calibration_dialog_context()
        dialog = CameraCalibrationDialog(
            self,
            profile_store=self._calibration_profile_store,
            source_label=source_label,
            target_resolution_text=resolution_text,
            metric_status_text="Not qualified • scale evidence required",
            initial_profile=self._selected_calibration_profile,
            allow_test_mode=False,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_profile is not None:
            self.set_camera_calibration_profile(dialog.selected_profile)

    def _choose_live_camera_mode(self) -> str | None:
        # A selected Phase 9 profile is an explicit user choice from the calibration
        # workspace, so Start can use it directly. Legacy B3 tests and workflows still
        # fall back to the historical chooser when no Phase 9 profile is selected.
        if self._selected_calibration_profile is not None:
            return "calibrated"

        source_label, resolution_text = self._camera_calibration_dialog_context()
        dialog = CameraCalibrationDialog(
            self,
            profile_store=self._calibration_profile_store,
            source_label=source_label,
            target_resolution_text=resolution_text,
            metric_status_text="Not qualified • scale evidence required",
            initial_profile=None,
            allow_test_mode=True,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        if dialog.test_mode_requested:
            return "estimated"
        if dialog.selected_profile is not None:
            self.set_camera_calibration_profile(dialog.selected_profile)
            return "calibrated"
        return None

    def _prepare_live_camera_for_start(
        self,
    ) -> tuple[ReconstructionCamera, CameraIntrinsicsPolicy] | None:
        mode = self._choose_live_camera_mode()
        if mode is None:
            return None
        if mode == "calibrated":
            camera = self._choose_live_camera_calibration()
            if camera is None:
                return None
            self._live_camera_mode = "calibrated"
            self._configure_visualizer_camera(camera)
            self._refresh_live_camera_presentation()
            return camera, CameraIntrinsicsPolicy.CALIBRATED_ONLY
        if mode == "estimated":
            if self._controller is None:
                return None
            camera = self._controller.build_estimated_live_camera()
            self._live_camera = camera
            self._live_calibration_path = None
            self._live_calibration_application = None
            self._live_camera_mode = "estimated"
            self._configure_visualizer_camera(camera)
            self._refresh_live_camera_presentation()
            return camera, CameraIntrinsicsPolicy.ALLOW_ESTIMATED
        raise RuntimeError(f"Unsupported live camera mode: {mode}")

    def _configure_visualizer_camera(self, camera: ReconstructionCamera) -> None:
        self._reconstruction_canvas.configure_camera_geometry(
            intrinsics_matrix=camera.intrinsics.matrix,
            image_size=(camera.image_size.width, camera.image_size.height),
        )

    def _choose_live_camera_calibration(self) -> ReconstructionCamera | None:
        profile = self._selected_calibration_profile
        if profile is not None:
            if self._controller is None:
                return None
            target_size = self._controller.current_live_image_size()
            applied = apply_camera_calibration_profile(
                profile,
                target_size,
                allow_same_aspect_full_frame_resize=True,
            )
            self._live_calibration_application = applied
            self._live_camera = applied.camera
            self._live_calibration_path = None
            return applied.camera

        # Preserve the exact Phase 8 legacy JSON path for backward compatibility.
        if self._live_camera is not None and self._live_calibration_path is not None:
            return self._live_camera
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Live Reconstruction Camera Calibration",
            str(Path.home()),
            "Camera calibration (*.json);;JSON files (*.json);;All files (*)",
        )
        if not selected:
            return None
        path = Path(selected).expanduser().resolve()
        camera = load_live_camera_calibration(path)
        self._live_camera = camera
        self._live_calibration_path = path
        self._live_calibration_application = None
        return camera

    def _reset_live_camera_selection(self) -> None:
        self._live_camera = None
        self._live_calibration_path = None
        self._live_calibration_application = None
        self._live_camera_mode = None
        if hasattr(self, "_reconstruction_canvas"):
            self._reconstruction_canvas.configure_camera_geometry(
                intrinsics_matrix=None, image_size=None
            )
        self._refresh_camera_authority()

    def _refresh_live_camera_presentation(self) -> None:
        profile = self._selected_calibration_profile
        if self._live_camera_mode == "estimated":
            self._runtime_prerequisite_value.setText("Estimated intrinsics • Test mode")
            self._runtime_prerequisite_value.setToolTip(
                "Explicit provisional mode: 70° assumed horizontal FOV, centered principal "
                "point, square pixels, zero distortion. Non-metric; metrology is not certified."
            )
        elif self._live_camera_mode == "calibrated" and profile is not None:
            applied = self._live_calibration_application
            suffix = ""
            if applied is not None and applied.was_rescaled:
                suffix = " • safely rescaled"
            self._runtime_prerequisite_value.setText(
                f"{profile.camera.camera_id} • {profile.validation_state.value.replace('_', ' ')}{suffix}"
            )
            self._runtime_prerequisite_value.setToolTip(
                f"Profile: {profile.profile_id}\nCalibration record: {profile.calibration_record_id}"
            )
        elif self._live_calibration_path is not None:
            self._runtime_prerequisite_value.setText(self._live_calibration_path.name)
            self._runtime_prerequisite_value.setToolTip(str(self._live_calibration_path))
        elif profile is not None:
            self._runtime_prerequisite_value.setText(
                f"Selected calibration • {profile.camera.camera_id}"
            )
            self._runtime_prerequisite_value.setToolTip(profile.profile_id)
        else:
            self._runtime_prerequisite_value.setText("Choose calibration or Test mode on Start")
            self._runtime_prerequisite_value.setToolTip(
                "Live Start opens the Phase 9 camera-calibration workspace or explicit provisional Test Mode."
            )
        self._refresh_camera_authority()

    def _refresh_camera_authority(self) -> None:
        if not hasattr(self, "_camera_geometry_value"):
            return
        profile = self._selected_calibration_profile
        if hasattr(self, "_calibration_clear_button"):
            self._calibration_clear_button.setEnabled(profile is not None)
        if profile is not None and hasattr(self, "_calibration_profile_value"):
            self._calibration_profile_value.setText(profile.camera.camera_id)
            self._calibration_profile_value.setToolTip(
                f"Profile: {profile.profile_id}\nRecord: {profile.calibration_record_id}"
            )
            self._calibration_resolution_value.setText(
                f"{profile.calibration_image_size.width} × {profile.calibration_image_size.height}"
            )
            self._calibration_resolution_value.setToolTip(
                "Calibration resolution. Different live resolutions are accepted only through the Phase 9.1B safe full-frame resize path."
            )
        elif hasattr(self, "_calibration_profile_value"):
            self._calibration_profile_value.setText("No profile selected")
            self._calibration_profile_value.setToolTip("")
            self._calibration_resolution_value.setText("—")
            self._calibration_resolution_value.setToolTip("")

        if self._live_camera_mode == "estimated":
            self._camera_geometry_value.setText("Estimated / test")
            self._camera_geometry_value.setToolTip(
                "Provisional intrinsics only; camera calibration has not been established."
            )
            self._metric_scale_value.setText("Unavailable (test)")
            self._metric_scale_value.setToolTip(
                "Test Mode does not establish metric scale or calibrated ray geometry."
            )
        elif self._live_camera_mode == "calibrated" and profile is not None:
            state_text = profile.validation_state.value.replace("_", " ").title()
            method_text = profile.method.value.replace("_", " ").title()
            self._camera_geometry_value.setText(f"{state_text} • {method_text}")
            self._camera_geometry_value.setToolTip(profile.provenance)
            self._metric_scale_value.setText("Not qualified")
            self._metric_scale_value.setToolTip(
                "Camera calibration does not establish scene metric scale. Separate Phase 9 metric evidence is required."
            )
        elif self._live_camera_mode == "calibrated":
            # Exact legacy Phase 8 presentation retained for existing JSON users/tests.
            self._camera_geometry_value.setText("Calibrated")
            self._camera_geometry_value.setToolTip(
                str(self._live_calibration_path or "Explicit calibrated camera profile")
            )
            self._metric_scale_value.setText("Evidence-driven")
            self._metric_scale_value.setToolTip(
                "Calibrated intrinsics do not by themselves establish monocular metric scale."
            )
        elif profile is not None:
            state_text = profile.validation_state.value.replace("_", " ").title()
            method_text = profile.method.value.replace("_", " ").title()
            self._camera_geometry_value.setText(f"{state_text} • {method_text}")
            self._camera_geometry_value.setToolTip(profile.provenance)
            self._metric_scale_value.setText("Not qualified")
            self._metric_scale_value.setToolTip(
                "A calibration profile is selected, but metric scale requires independent scale evidence."
            )
        else:
            self._camera_geometry_value.setText("Not selected")
            self._camera_geometry_value.setToolTip("")
            self._metric_scale_value.setText("Evidence-driven")
            self._metric_scale_value.setToolTip("")

        assessment = self._persisted_metric_assessment
        if self._persistent_scene is not None and assessment is not None:
            if assessment.authorizes_calibrated_metric_scale:
                self._metric_scale_value.setText("Qualified")
            elif assessment.authorizes_metric_coordinates:
                self._metric_scale_value.setText("Provisional")
            else:
                self._metric_scale_value.setText("Not qualified")
            details = [assessment.message]
            if assessment.relative_scale_uncertainty is not None:
                details.append(
                    f"Relative scale uncertainty: ±{assessment.relative_scale_uncertainty * 100.0:.2f}%"
                )
            self._metric_scale_value.setToolTip("\n".join(details))
        elif self._persistent_scene is not None and self._persistent_authority_snapshot is None:
            self._metric_scale_value.setToolTip(
                "Legacy persistent scene: no Phase 9.1H calibration/metric authority sidecar is available."
            )
        self._metrology_value.setText("Not certified")

    def _pause_or_resume_runtime(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown:
            return
        try:
            if self._live_state is LiveCaptureRuntimeState.PAUSED:
                if not controller.resume_live():
                    raise RuntimeError("Live reconstruction could not resume from its current state.")
            elif self._live_state is LiveCaptureRuntimeState.RUNNING:
                if not controller.pause_live():
                    raise RuntimeError("Live reconstruction could not pause from its current state.")
            else:
                return
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        finally:
            self._refresh_runtime_controls()

    def _stop_runtime(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown:
            return
        try:
            if self._offline_state in {
                OfflineReconstructionRunState.STARTING,
                OfflineReconstructionRunState.RUNNING,
            }:
                if not controller.cancel_offline():
                    raise RuntimeError("Offline reconstruction could not be cancelled.")
            elif self._live_state in {
                LiveCaptureRuntimeState.STARTING,
                LiveCaptureRuntimeState.RUNNING,
                LiveCaptureRuntimeState.PAUSED,
                LiveCaptureRuntimeState.RECOVERING,
            }:
                if not controller.request_stop_live():
                    raise RuntimeError("Live reconstruction could not be stopped.")
            else:
                return
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        finally:
            self._refresh_runtime_controls()

    def _prepare_runtime_presentation_for_start(self, workflow: ReconstructionWorkflow) -> None:
        self._runtime_started_at = None
        self._runtime_elapsed_frozen = 0.0
        self._active_runtime_kind = None
        self._last_runtime_error = ""
        self._offline_sparse_export_source = None
        self._canonical_surface_export_source = None
        self._inspection_qc_export_source = None
        self._reconstruction_export_source_error = ""
        self._refresh_reconstruction_export_controls()
        self.clear_reconstruction_measurement_evidence()
        self.clear_reconstruction_inspection_qc_evidence()
        self.clear_reconstruction_reference_comparison()
        self._reset_semantic_presentation()
        if workflow is ReconstructionWorkflow.LIVE_SOURCE:
            self._live_pose_count = 0
            self._live_keyframe_count = 0
            self._live_landmark_count = None
            self._telemetry_value.setText("Waiting for live telemetry")
            self._apply_runtime_presentation(
                ReconstructionStatusPresentation(
                    "Preparing live session",
                    "Validating camera geometry and live reconstruction prerequisites.",
                    "running",
                    indeterminate=True,
                )
            )
        else:
            self._pending_visual_pose = None
            self._pending_sparse_landmarks = None
            self._pending_keyframes = None
            self._visual_pose_timer.stop()
            self._reset_offline_sfm_health()
            self._reset_offline_spatial_coverage()
            self._reset_offline_inspection()
            self._reconstruction_canvas.clear_live_state()
            self._viewport_placeholder.setVisible(True)
            self._visualizer_fitted_once = False
            self._offline_visualization_path = None
            self._offline_visualization_error = ""
            self._telemetry_value.setText("Waiting for offline telemetry")
            self._apply_runtime_presentation(
                ReconstructionStatusPresentation(
                    "Preparing offline job",
                    "Preparing the isolated offline reconstruction worker.",
                    "running",
                    indeterminate=True,
                )
            )

    def _begin_runtime_clock(self, kind: str) -> None:
        if self._active_runtime_kind == kind and self._runtime_started_at is not None:
            return
        self._active_runtime_kind = kind
        self._runtime_started_at = monotonic()
        self._runtime_elapsed_frozen = 0.0
        self._refresh_elapsed_label()

    def _freeze_runtime_clock(self) -> None:
        if self._runtime_started_at is not None:
            self._runtime_elapsed_frozen = max(0.0, monotonic() - self._runtime_started_at)
            self._runtime_started_at = None
        self._active_runtime_kind = None
        self._refresh_elapsed_label()

    def _refresh_elapsed_label(self) -> None:
        elapsed = self._runtime_elapsed_frozen
        if self._runtime_started_at is not None:
            elapsed = max(0.0, monotonic() - self._runtime_started_at)
        self._elapsed_value.setText(format_elapsed_seconds(elapsed) if elapsed > 0.0 else "—")

    def _apply_runtime_presentation(self, presentation: ReconstructionStatusPresentation) -> None:
        if not isinstance(presentation, ReconstructionStatusPresentation):
            raise TypeError("presentation must be ReconstructionStatusPresentation.")
        self._stage_value.setText(presentation.stage_label)
        self._stage_value.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._stage_value)
        self._runtime_message.setText(presentation.message)
        self._runtime_message.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._runtime_message)

        if presentation.indeterminate:
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setFormat("Active")
        else:
            self._progress_bar.setRange(0, 100)
            if presentation.progress_fraction is None:
                self._progress_bar.setValue(0)
                self._progress_bar.setFormat(
                    presentation.stage_label if presentation.status_kind in {"error", "warning"} else "—"
                )
            else:
                value = max(0, min(100, round(presentation.progress_fraction * 100.0)))
                self._progress_bar.setValue(value)
                self._progress_bar.setFormat("%p%")
        self._progress_bar.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._progress_bar)

    def set_offline_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._apply_runtime_presentation(present_offline_event(payload))
        if self._offline_state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        }:
            self._begin_runtime_clock("offline")

    def _on_offline_result_ready(self, payload: object, _output_dir: str) -> None:
        if not isinstance(payload, dict):
            return
        self._offline_sparse_export_source = None
        self._reconstruction_export_source_error = ""
        if (
            str(payload.get("status", "")).strip().lower() == "completed"
            and bool(payload.get("pass", False))
        ):
            try:
                self._offline_sparse_export_source = OfflineSparseExportSource(payload)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                # Export qualification is ancillary to the completed reconstruction result.
                # Do not downgrade the reconstruction lifecycle if its optional product-export
                # evidence is missing or invalid; simply keep export disabled and explain why.
                self._reconstruction_export_source_error = str(exc)
        self._refresh_reconstruction_export_controls()
        sfm = payload.get("sfm_summary", {})
        if isinstance(sfm, dict):
            registered = sfm.get("registered_keyframes")
            total = sfm.get("input_keyframes")
            landmarks = sfm.get("landmarks")
            error = sfm.get("mean_reprojection_error_px")
            parts: list[str] = []
            if isinstance(registered, int):
                if isinstance(total, int) and total > 0:
                    parts.append(f"{registered}/{total} keyframes")
                else:
                    parts.append(f"{registered} keyframes")
            if isinstance(landmarks, int):
                parts.append(f"{landmarks:,} landmarks")
            if isinstance(error, (int, float)):
                parts.append(f"{float(error):.2f} px")
            if parts:
                self._telemetry_value.setText(" • ".join(parts))
                self._telemetry_value.setToolTip("Offline sparse-reconstruction result summary.")
            try:
                self._apply_offline_sfm_health(build_offline_sfm_health_presentation(sfm))
            except (TypeError, ValueError) as exc:
                self._apply_offline_sfm_health(unavailable_offline_sfm_health(str(exc)))
        else:
            self._apply_offline_sfm_health(
                unavailable_offline_sfm_health("Offline result does not contain an SfM summary.")
            )

        if str(payload.get("status", "")).strip().lower() != "completed":
            return
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            self._offline_visualization_error = "Offline result does not expose visualization artifacts."
            return
        visualization_path = str(artifacts.get("offline_visualization_json", "")).strip()
        if not visualization_path:
            # Older A3 results remain valid but simply cannot be visualized by D1/D2.
            self._offline_visualization_error = "Offline result predates the D1 visualization artifact."
            self._reset_offline_spatial_coverage()
            self._reset_offline_inspection()
            return
        try:
            visual = load_offline_sparse_visualization(visualization_path)
            expected_session = str(payload.get("session_id", "")).strip()
            if expected_session and visual.session_id != expected_session:
                raise ValueError("Offline visualization session identity does not match the worker result.")
            self._reconstruction_canvas.show_offline_sparse_reconstruction(
                session_id=visual.session_id,
                scale_status=visual.scale_status,
                points_reconstruction=visual.landmark_points_reconstruction,
                camera_matrices_reconstruction=visual.camera_matrices_reconstruction,
                camera_ids=visual.camera_ids,
                landmark_colors_rgb=visual.landmark_colors_rgb,
                landmark_total=visual.landmark_total,
                registered_camera_total=visual.registered_camera_total,
            )
            self._offline_visualization_path = Path(visualization_path).expanduser().resolve()
            self._offline_visualization_error = ""
            self._activate_offline_inspection(visual)
            self._apply_offline_spatial_coverage(build_offline_sparse_spatial_coverage(visual))
            self._viewport_state.setToolTip("")
            if self._reconstruction_canvas.offline_visualization_loaded:
                self._viewport_placeholder.setVisible(False)
                self._reconstruction_canvas.fit_scene()
                self._visualizer_fitted_once = True
            self._refresh_viewport_state()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            # A visualization artifact is presentation-only. A completed authoritative
            # offline worker result remains completed even if this optional view fails.
            self._offline_visualization_error = str(exc)
            self._reset_offline_spatial_coverage()
            self._reset_offline_inspection()
            self._viewport_state.setText("Offline visualization unavailable")
            self._viewport_state.setToolTip(str(exc))

    def set_live_observation(self, observation: object) -> None:
        if self._viewport_shutdown:
            return
        diagnostics = self._live_diagnostics.update(observation)
        self._diagnostics_hud.set_presentation(diagnostics)
        visual_update_pending = False
        pose = getattr(observation, "pose", None)
        if pose is not None:
            self._live_pose_count += 1
            matrix = getattr(pose, "T_reconstruction_from_camera", None)
            if matrix is not None:
                value = np.asarray(matrix, dtype=np.float64)
                if value.shape == (4, 4) and np.isfinite(value).all():
                    # Worker-thread observations are already queued onto the Qt/UI
                    # thread by the controller. C1 additionally coalesces visual
                    # updates so a high-rate live stream cannot force one GL update
                    # per observation. C3 coalesces sparse-map samples on the same
                    # presentation timer.
                    self._pending_visual_pose = np.array(value, copy=True)
                    visual_update_pending = True
        if bool(getattr(observation, "is_keyframe", False)):
            self._live_keyframe_count += 1
        snapshot = getattr(observation, "map_snapshot", None)
        if snapshot is not None:
            self._live_keyframe_count = max(
                self._live_keyframe_count,
                int(getattr(snapshot, "keyframe_count", 0)),
            )
            self._live_landmark_count = int(getattr(snapshot, "landmark_count", 0))
            if hasattr(snapshot, "landmarks_reconstruction"):
                raw_points = getattr(snapshot, "landmarks_reconstruction")
                points = np.asarray(raw_points, dtype=np.float64)
                if points.size == 0:
                    points = np.zeros((0, 3), dtype=np.float64)
                if points.ndim == 2 and points.shape[1:] == (3,) and np.isfinite(points).all():
                    self._pending_sparse_landmarks = (
                        np.array(points, copy=True),
                        int(getattr(snapshot, "landmark_count", points.shape[0])),
                        int(getattr(snapshot, "revision", 0)),
                    )
                    visual_update_pending = True
            if hasattr(snapshot, "keyframes_reconstruction"):
                raw_keyframes = tuple(getattr(snapshot, "keyframes_reconstruction"))
                matrices: list[np.ndarray] = []
                keyframe_ids: list[str] = []
                for sample in raw_keyframes:
                    matrix = getattr(sample, "T_reconstruction_from_camera", None)
                    keyframe_id = str(getattr(sample, "keyframe_id", "")).strip()
                    if matrix is None or not keyframe_id:
                        continue
                    value = np.asarray(matrix, dtype=np.float64)
                    if value.shape == (4, 4) and np.isfinite(value).all():
                        matrices.append(np.array(value, copy=True))
                        keyframe_ids.append(keyframe_id)
                keyframe_array = (
                    np.asarray(matrices, dtype=np.float64)
                    if matrices
                    else np.zeros((0, 4, 4), dtype=np.float64)
                )
                self._pending_keyframes = (
                    keyframe_array,
                    tuple(keyframe_ids),
                    int(getattr(snapshot, "keyframe_count", keyframe_array.shape[0])),
                    int(getattr(snapshot, "revision", 0)),
                )
                visual_update_pending = True
        if visual_update_pending and not self._visual_pose_timer.isActive():
            self._visual_pose_timer.start()

    def _flush_pending_visual_pose(self) -> None:
        if self._viewport_shutdown:
            self._pending_visual_pose = None
            self._pending_sparse_landmarks = None
            self._pending_keyframes = None
            return
        matrix = self._pending_visual_pose
        landmark_update = self._pending_sparse_landmarks
        keyframe_update = self._pending_keyframes
        self._pending_visual_pose = None
        self._pending_sparse_landmarks = None
        self._pending_keyframes = None
        if matrix is None and landmark_update is None and keyframe_update is None:
            return
        try:
            if matrix is not None:
                self._reconstruction_canvas.update_live_pose(matrix)
            if landmark_update is not None:
                points, total_count, revision = landmark_update
                self._reconstruction_canvas.update_sparse_landmarks(
                    points, total_count=total_count, revision=revision
                )
            if keyframe_update is not None:
                matrices, keyframe_ids, total_count, revision = keyframe_update
                self._reconstruction_canvas.update_keyframe_poses(
                    matrices,
                    keyframe_ids=keyframe_ids,
                    total_count=total_count,
                    revision=revision,
                )
            if (
                self._reconstruction_canvas.has_pose
                or self._reconstruction_canvas.has_sparse_map
                or self._reconstruction_canvas.has_keyframes
            ):
                self._viewport_placeholder.setVisible(False)
            if not self._visualizer_fitted_once and (
                self._reconstruction_canvas.has_pose
                or self._reconstruction_canvas.has_sparse_map
                or self._reconstruction_canvas.has_keyframes
            ):
                self._reconstruction_canvas.fit_scene()
                self._visualizer_fitted_once = True
            self._refresh_viewport_state()
        except (TypeError, ValueError) as exc:
            # Presentation errors must never terminate the authoritative live
            # reconstruction runtime. Surface the issue as a UI warning only.
            self.set_error(f"Reconstruction viewport update rejected: {exc}")

    def _refresh_viewport_state(self) -> None:
        if self.interactive_gaussian_visible:
            self._viewport_state.setText("Interactive Gaussian appearance • canonical picking")
            return
        if self.nerf_preview_visible:
            self._viewport_state.setText("NeRF preview • appearance only")
            return
        if self.gaussian_preview_visible:
            self._viewport_state.setText("Gaussian Splatting preview • appearance only")
            return
        if (
            self._reconstruction_canvas.canonical_mesh_loaded
            and self._reconstruction_canvas.canonical_camera_loaded
        ):
            self._viewport_state.setText("Canonical scene • cameras + trajectory")
            return
        if self._reconstruction_canvas.canonical_mesh_loaded:
            self._viewport_state.setText("Canonical surface mesh")
            return
        if self._reconstruction_canvas.canonical_camera_loaded:
            self._viewport_state.setText("Canonical camera context")
            return
        if self._canonical_viewer_error and self._persistent_scene is not None:
            self._viewport_state.setText("Canonical surface unavailable")
            return
        if self._scene_availability is ReconstructionSceneAvailability.READY:
            self._viewport_state.setText("Scene ready")
            return
        if self._reconstruction_canvas.offline_visualization_loaded:
            self._viewport_state.setText("Offline sparse reconstruction")
            return
        if self._reconstruction_canvas.semantic_objects_available:
            self._viewport_state.setText("Semantic object context")
            return
        if self._reconstruction_canvas.has_pose:
            suffix = " retained" if self._live_state in {
                LiveCaptureRuntimeState.STOPPED,
                LiveCaptureRuntimeState.FAILED,
            } else ""
            self._viewport_state.setText(f"Live camera pose{suffix}")
            return
        if self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
        }:
            self._viewport_state.setText("Waiting for live pose")
            return
        self._viewport_state.setText("Awaiting scene")

    def _refresh_runtime_telemetry(self) -> None:
        self._refresh_elapsed_label()
        controller = self._controller
        if controller is None:
            return
        if self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        }:
            stats = controller.live_stats
            self._telemetry_value.setText(
                format_live_telemetry(
                    stats,
                    pose_count=self._live_pose_count,
                    keyframe_count=self._live_keyframe_count,
                    landmark_count=self._live_landmark_count,
                )
            )
            self._telemetry_value.setToolTip(
                "Live statistics are presentation-only telemetry from the authoritative live runtime."
            )
            self._apply_runtime_presentation(present_live_state(self._live_state, stats))
            self._refresh_failure_recovery_controls()

    def _on_offline_run_finished(self, _result: object) -> None:
        self._refresh_runtime_telemetry()
        self._refresh_source_controls()
        self._refresh_runtime_controls()

    def _refresh_source_controls(self) -> None:
        controller = self._controller
        enabled = bool(
            controller is not None
            and not controller.is_active
            and not controller.is_shutdown
            and not self._digital_twin_processing_snapshot.is_active
        )
        for button in self._source_buttons.values():
            button.setEnabled(enabled)

    def _refresh_runtime_controls(self) -> None:
        controller = self._controller
        if controller is None:
            self._start_button.setEnabled(False)
            self._pause_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            if hasattr(self, "_calibration_manage_button"):
                self._calibration_manage_button.setEnabled(True)
                self._calibration_clear_button.setEnabled(
                    self._selected_calibration_profile is not None
                )
            self._refresh_scene_interaction_controls()
            self._refresh_checkpoint_resume_controls()
            self._refresh_failure_recovery_controls()
            return

        selection = controller.source_selection
        self._refresh_failure_recovery_controls()
        recovery_owns_retry = bool(
            self._failure_recovery_presentation
            and self._failure_recovery_presentation.state
            in {
                ReconstructionFailureRecoveryState.RETRY_AVAILABLE,
                ReconstructionFailureRecoveryState.CANCELLED_RETRY_AVAILABLE,
            }
        )
        live_busy = self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        }
        offline_busy = self._offline_state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        }
        busy = (
            live_busy
            or offline_busy
            or controller.is_active
            or self._digital_twin_processing_snapshot.is_active
        )
        if hasattr(self, "_calibration_manage_button"):
            calibration_mutable = not controller.is_shutdown and not busy
            self._calibration_manage_button.setEnabled(calibration_mutable)
            self._calibration_clear_button.setEnabled(
                calibration_mutable and self._selected_calibration_profile is not None
            )
        can_start = (
            not controller.is_shutdown
            and selection.ready
            and selection.workflow is not ReconstructionWorkflow.NONE
            and not busy
            and not recovery_owns_retry
        )
        self._start_button.setEnabled(can_start)

        can_pause = (
            not controller.is_shutdown
            and not offline_busy
            and self._live_state in {
                LiveCaptureRuntimeState.RUNNING,
                LiveCaptureRuntimeState.PAUSED,
            }
        )
        self._pause_button.setEnabled(can_pause)
        self._pause_button.setText(
            "Resume" if self._live_state is LiveCaptureRuntimeState.PAUSED else "Pause"
        )

        can_stop_live = self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
        }
        can_stop_offline = self._offline_state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
        }
        self._stop_button.setEnabled(
            not controller.is_shutdown and (can_stop_live or can_stop_offline)
        )
        self._stop_button.setText("Cancel" if offline_busy else "Stop")
        self._refresh_scene_interaction_controls()
        self._refresh_checkpoint_resume_controls()
        self._refresh_failure_recovery_controls()
        self._refresh_da3_canonical_controls()

    def _clear_warning_if_possible(self) -> None:
        if self._controller is None or self._controller.is_shutdown:
            return
        if (
            self._controller.source_selection.workflow is ReconstructionWorkflow.LIVE_SOURCE
            and self._live_camera_mode == "estimated"
        ):
            self._readiness_badge.setText("TEST MODE • NON-METRIC")
            self._readiness_badge.setProperty("statusKind", "warning")
            self._readiness_badge.setToolTip(
                "Live reconstruction is using estimated intrinsics. Visual/relative testing is "
                "allowed; metric geometry and metrology are not certified."
            )
            self._refresh_style(self._readiness_badge)
            return
        self._readiness_badge.setText("CONTROLLER READY")
        self._readiness_badge.setProperty("statusKind", "")
        self._readiness_badge.setToolTip("")
        self._refresh_style(self._readiness_badge)

    @staticmethod
    def _workflow_label(workflow: ReconstructionWorkflow) -> str:
        labels = {
            ReconstructionWorkflow.NONE: "Choose a workflow",
            ReconstructionWorkflow.LIVE_SOURCE: "Live source",
            ReconstructionWorkflow.IMAGE_SEQUENCE: "Image sequence",
            ReconstructionWorkflow.RECORDED_VIDEO: "Recorded video",
        }
        return labels[workflow]

    def set_live_state(self, state: LiveCaptureRuntimeState) -> None:
        previous_state = self._live_state
        self._live_state = state
        if state is LiveCaptureRuntimeState.STARTING and previous_state is not LiveCaptureRuntimeState.STARTING:
            self._pending_visual_pose = None
            self._pending_sparse_landmarks = None
            self._pending_keyframes = None
            self._visual_pose_timer.stop()
            self._reset_offline_sfm_health()
            self._reset_offline_inspection()
            self._reset_semantic_presentation()
            self._reconstruction_canvas.clear_live_state()
            self._viewport_placeholder.setVisible(True)
            self._visualizer_fitted_once = False
            self._diagnostics_hud.set_presentation(self._live_diagnostics.reset_session())
        elif state in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED}:
            self._diagnostics_hud.set_presentation(self._live_diagnostics.mark_terminal(True))
        elif state in {
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
        }:
            self._diagnostics_hud.set_presentation(self._live_diagnostics.mark_terminal(False))
        self._sync_diagnostics_visibility()
        self._live_value.setText(self._humanize(state))
        self._live_value.setProperty("statusKind", self._state_kind(state.value))
        self._refresh_style(self._live_value)
        if state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        }:
            self._begin_runtime_clock("live")
        elif state in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED}:
            self._freeze_runtime_clock()
        stats = self._controller.live_stats if self._controller is not None else None
        presentation = present_live_state(state, stats)
        if state is LiveCaptureRuntimeState.FAILED and self._last_runtime_error:
            presentation = ReconstructionStatusPresentation(
                presentation.stage_label,
                self._last_runtime_error,
                "error",
            )
        self._apply_runtime_presentation(presentation)
        self._refresh_session_label()
        self._refresh_viewport_state()
        self._refresh_source_controls()
        self._refresh_runtime_controls()
        self._refresh_digital_twin_resource_coordination()

    def set_offline_state(self, state: OfflineReconstructionRunState) -> None:
        self._offline_state = state
        self._offline_value.setText(self._humanize(state))
        self._offline_value.setProperty("statusKind", self._state_kind(state.value))
        self._refresh_style(self._offline_value)
        if state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        }:
            self._begin_runtime_clock("offline")
        elif state in {
            OfflineReconstructionRunState.COMPLETED,
            OfflineReconstructionRunState.FAILED,
            OfflineReconstructionRunState.CANCELLED,
        }:
            self._freeze_runtime_clock()
        # Preserve a richer worker event while RUNNING; state transitions own terminal presentation.
        if state is not OfflineReconstructionRunState.RUNNING:
            presentation = present_offline_state(state)
            if state is OfflineReconstructionRunState.FAILED and self._last_runtime_error:
                presentation = ReconstructionStatusPresentation(
                    presentation.stage_label,
                    self._last_runtime_error,
                    "error",
                )
            self._apply_runtime_presentation(presentation)
        self._refresh_session_label()
        self._refresh_source_controls()
        self._refresh_runtime_controls()
        self._refresh_reconstruction_export_controls()
        self._refresh_digital_twin_resource_coordination()

    def set_scene_availability(self, availability: ReconstructionSceneAvailability) -> None:
        self._scene_availability = availability
        self._scene_value.setText(self._humanize(availability))
        kind = {
            ReconstructionSceneAvailability.READY: "ready",
            ReconstructionSceneAvailability.DEGRADED: "warning",
            ReconstructionSceneAvailability.PARTIAL: "running",
            ReconstructionSceneAvailability.NONE: "neutral",
        }[availability]
        self._scene_value.setProperty("statusKind", kind)
        self._refresh_style(self._scene_value)
        self._refresh_inspection_selection()
        self._refresh_viewport_state()

    def set_persistent_scene(self, scene, manifest_path: str) -> None:
        bind_started = monotonic()
        previous_scene_id = (
            str(getattr(self._persistent_scene, "scene_id", "")).strip()
            if self._persistent_scene is not None
            else ""
        )
        next_scene_id = str(getattr(scene, "scene_id", "")).strip() if scene is not None else ""
        selected_before = str(self._reconstruction_canvas.selected_semantic_object_id or "").strip()
        bind_plan = build_reconstruction_scene_bind_plan(
            previous_scene_id,
            next_scene_id,
            selected_object_id=selected_before,
        )
        if next_scene_id != previous_scene_id:
            # E0 estimates and the Phase 9.6F workspace companion are scene-bound.
            # A different scene must never inherit approximate scale/ruler state,
            # appearance selection, or terminal build history from another scene.
            self._estimated_metric_scale_hint = None
            self._ruler_metric_record = None
            self._last_ruler_reference_distance_reconstruction_units = None
            self._digital_twin_workspace_state = None
            self._digital_twin_workspace_manifest = None
            self._digital_twin_processing_history = None

        self._hide_interactive_gaussian()
        self._hide_gaussian_preview()
        self._hide_nerf_visualization(refresh=False)
        self._appearance_fallback_reason = ""
        # Semantic/camera resources remain scene-bound. Clear them before changing
        # identity, but keep the current non-empty canonical MeshVisual alive until
        # the replacement resource has been validated. 9.3I then overwrites that
        # node in place on success instead of detach/recreate churn. Exact same-scene
        # semantic rebinding also keeps the non-empty selected highlight visual hidden
        # and reuses it once the persisted membership is validated again.
        self._reconstruction_canvas.clear_canonical_semantic_objects(
            preserve_selected_mesh_visual=bind_plan.same_scene_reload
        )
        self._canonical_semantic_presentation = None
        self._canonical_semantic_error = ""
        self._canonical_inspection_overlay_presentation = None
        self._canonical_inspection_overlay_error = ""
        self._canonical_measurement_registration_presentation = None
        self._reconstruction_canvas.clear_canonical_registered_dimensions(update_canvas=False)
        self._reconstruction_canvas.clear_canonical_camera_trajectory()
        self._canonical_camera_presentation = None
        self._canonical_camera_error = ""
        self._persistent_scene = scene
        # Scene identity changed; any previously restored metric assessment is
        # no longer bound until the matching Phase 9.1H authority sidecar arrives.
        # Keep the operational calibration selection intact so a newly completed
        # reconstruction can persist the calibration that was actually used.
        self._persistent_authority_snapshot = None
        self._persisted_metric_assessment = None
        self._refresh_camera_authority()
        self._refresh_canonical_scene_presentation()
        self._refresh_dense_provider_controls()
        mesh_visual_reused = self._refresh_canonical_geometry_viewer(fit_scene=False)
        self._refresh_canonical_camera_viewer(fit_scene=False)
        self._refresh_scene_layer_controls()
        self._refresh_reconstruction_export_controls()
        self._refresh_appearance_representation_controls()
        self._refresh_gaussian_visualization_controls()
        self._refresh_nerf_visualization_controls()
        self._refresh_representation_switching_controls()
        # 9.6E3 build eligibility is scene-bound. Persistent-scene load/reload
        # happens after the page was initially bound, so refresh the build controls
        # on every scene transition instead of leaving the startup "no scene"
        # state latched until some unrelated runtime/source event occurs.
        self._refresh_nerf_build_controls()

        if scene is None:
            self._scene_manifest_value.setText("No persistent scene")
            self.clear_reconstruction_measurement_evidence()
            self.clear_reconstruction_inspection_qc_evidence()
            self.clear_reconstruction_reference_comparison()
            self._reset_semantic_presentation()
            self._refresh_inspection_selection()
        else:
            path = Path(manifest_path).name if manifest_path else "Persistent scene ready"
            self._scene_manifest_value.setText(path)
            try:
                presentation = semantic_presentation_from_persistent_scene(scene)
            except (TypeError, ValueError) as exc:
                self._reset_semantic_presentation(hide=False)
                self._semantic_structure_value.setText("Semantic context unavailable")
                self._semantic_structure_value.setToolTip(str(exc))
            else:
                if presentation is None:
                    self._reset_semantic_presentation()
                else:
                    self._apply_semantic_presentation(presentation)
                    self._refresh_canonical_semantic_viewer(
                        preserve_selected_mesh_visual=bind_plan.same_scene_reload
                    )
                    preserve_id = bind_plan.preserve_selected_object_id
                    if preserve_id:
                        for index, item in enumerate(presentation.objects):
                            if item.object_id == preserve_id:
                                self._semantic_object_combo.setCurrentIndex(index + 1)
                                break
                    self._refresh_scene_layer_controls()

        # 9.3B/9.3C previously fitted once for mesh and again for camera context.
        # 9.3I performs one final Fit only after a different scene is fully bound;
        # an exact same-scene reload preserves the user's current orbit/pan/zoom.
        if bind_plan.fit_after_bind and (
            self._reconstruction_canvas.canonical_mesh_loaded
            or self._reconstruction_canvas.canonical_camera_loaded
        ):
            self._reconstruction_canvas.fit_scene()
            self._visualizer_fitted_once = True

        selection_preserved = bool(
            bind_plan.preserve_selected_object_id
            and self._reconstruction_canvas.selected_semantic_object_id
            == bind_plan.preserve_selected_object_id
        )
        # Count only real persistent-scene binds. Controller attachment and an
        # explicit scene clear both call set_persistent_scene(None, ...); those
        # are lifecycle transitions, not scene binds, and must not inflate the
        # user-facing bind/reload diagnostics.
        if next_scene_id:
            self._record_scene_bind_runtime_health(
                started_at=bind_started,
                same_scene_reload=bind_plan.same_scene_reload,
                mesh_visual_reused=mesh_visual_reused,
                view_preserved=bind_plan.preserve_view,
                selection_preserved=selection_preserved,
            )
        else:
            self._last_scene_mesh_visual_reused = False
            self._last_scene_view_preserved = False
            self._last_scene_selection_preserved = False
            self._refresh_scene_runtime_health()
        # 9.6C records the final scene-bind navigation policy only after the 9.3I
        # bind plan has either preserved the same-scene view or fitted the new scene.
        self._refresh_digital_twin_navigation_sync()
        self._refresh_digital_twin_processing_history_presentation()
        self._refresh_digital_twin_workspace_persistence_controls()

    def set_persistent_authority_snapshot(
        self,
        snapshot: PersistentSceneAuthoritySnapshot | None,
        manifest_path: str,
    ) -> None:
        if snapshot is not None and not isinstance(snapshot, PersistentSceneAuthoritySnapshot):
            raise TypeError("snapshot must be PersistentSceneAuthoritySnapshot or None.")
        self._persistent_authority_snapshot = snapshot
        self._persisted_metric_assessment = (
            None if snapshot is None else snapshot.metric_qualification_assessment
        )
        if snapshot is not None and snapshot.camera_calibration_profile is not None:
            profile = snapshot.camera_calibration_profile
            if self._selected_calibration_profile is None or (
                self._selected_calibration_profile.as_dict() != profile.as_dict()
            ):
                self.set_camera_calibration_profile(profile)
        elif self._persistent_scene is not None and self._selected_calibration_profile is not None:
            # Loading a legacy/no-sidecar scene must not silently retain unrelated
            # calibration evidence from a previously focused scene.
            self.clear_camera_calibration_profile()
        self._refresh_camera_authority()
        self._refresh_canonical_scene_presentation()
        self._refresh_dense_provider_controls()
        self._apply_canonical_viewer_status()

    def set_checkpoint(self, checkpoint, manifest_path: str) -> None:
        if checkpoint is None:
            self._checkpoint_value.setText("No checkpoint")
            self._checkpoint_value.setToolTip("")
            self._refresh_checkpoint_resume_controls()
            self._refresh_failure_recovery_controls()
            return
        path = Path(manifest_path).name if manifest_path else "Checkpoint ready"
        self._checkpoint_value.setText(path)
        self._checkpoint_value.setToolTip(str(manifest_path).strip() or str(getattr(checkpoint, "checkpoint_id", "")))
        self._refresh_checkpoint_resume_controls()
        self._refresh_failure_recovery_controls()

    def set_error(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        self._last_runtime_error = text
        self._readiness_badge.setText("CONTROLLER WARNING")
        self._readiness_badge.setProperty("statusKind", "warning")
        self._readiness_badge.setToolTip(text)
        self._refresh_style(self._readiness_badge)
        self._runtime_message.setText(text)
        self._runtime_message.setProperty("statusKind", "error")
        self._refresh_style(self._runtime_message)
        self._refresh_failure_recovery_controls()

    def _refresh_session_label(self) -> None:
        controller = self._controller
        if controller is None:
            return
        self.set_session_model(controller.session_model)
        output_dir = controller.offline_output_dir
        if output_dir is not None:
            self._runtime_output_value.setText(output_dir.name)
            self._runtime_output_value.setToolTip(str(output_dir))

    def shutdown_viewport(self) -> None:
        """Release the dedicated reconstruction GL canvas during app shutdown."""

        if self._viewport_shutdown:
            return
        self._viewport_shutdown = True
        if hasattr(self, "_digital_twin_processing_controller"):
            self._digital_twin_processing_controller.shutdown()
        self._visual_pose_timer.stop()
        if hasattr(self, "_gaussian_display_update_timer"):
            self._gaussian_display_update_timer.stop()
        self._gaussian_display_update_pending = False
        if self._interactive_gaussian_scene is not None or self._gaussian_display_buffers is not None:
            self._hide_interactive_gaussian()
        if self._interactive_nerf_visible or self._nerf_worker_client.is_running:
            self._hide_interactive_nerf(blocking=True)
        self._pending_visual_pose = None
        self._pending_sparse_landmarks = None
        self._pending_keyframes = None
        if hasattr(self, "_orientation_gizmo"):
            self._orientation_gizmo.shutdown()
        self._canonical_viewer_mesh = None
        self._canonical_viewer_error = ""
        self._canonical_camera_presentation = None
        self._canonical_camera_error = ""
        self._canonical_semantic_presentation = None
        self._canonical_semantic_error = ""
        self._canonical_inspection_overlay_presentation = None
        self._canonical_inspection_overlay_error = ""
        self._canonical_measurement_registration_presentation = None
        self._reconstruction_canvas.close()

    @staticmethod
    def _humanize(value) -> str:
        raw = getattr(value, "value", value)
        return str(raw).replace("_", " ").strip().title() or "—"

    @staticmethod
    def _state_kind(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized in {"running", "starting", "recovering", "cancelling"}:
            return "running"
        if normalized in {"failed"}:
            return "error"
        if normalized in {"paused", "degraded"}:
            return "warning"
        if normalized in {"completed", "ready", "stopped"}:
            return "ready"
        return "neutral"

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
