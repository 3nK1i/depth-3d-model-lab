"""Dormant Reconstruction page foundation for Phase 8.8B1.

The page intentionally exposes no start/stop actions yet.  It establishes the
visual hierarchy and passively mirrors the already-qualified Phase 8.8A
controller state while primary navigation remains disabled.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.theme import (
    PAGE_MARGIN_BOTTOM,
    PAGE_MARGIN_H,
    PAGE_MARGIN_TOP,
    PAGE_SPACING,
    SPLITTER_HANDLE,
)
from core.reconstruction import LiveCaptureRuntimeState


class ReconstructionPage(QWidget):
    """Professional Reconstruction workspace shell with passive state binding."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("reconstructionPage")
        self._controller = None
        self._build_ui()
        self.set_live_state(LiveCaptureRuntimeState.IDLE)
        self.set_offline_state(OfflineReconstructionRunState.IDLE)
        self.set_scene_availability(ReconstructionSceneAvailability.NONE)
        self.set_checkpoint(None, "")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            PAGE_MARGIN_H,
            PAGE_MARGIN_TOP,
            PAGE_MARGIN_H,
            PAGE_MARGIN_BOTTOM,
        )
        outer.setSpacing(PAGE_SPACING)

        outer.addWidget(self._build_header_card())

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setObjectName("reconstructionBodySplitter")
        body.setHandleWidth(SPLITTER_HANDLE)
        body.setChildrenCollapsible(False)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(PAGE_SPACING)

        top = QHBoxLayout()
        top.setSpacing(PAGE_SPACING)
        top.addWidget(self._build_source_card(), stretch=1)
        top.addWidget(self._build_runtime_card(), stretch=1)
        main_layout.addLayout(top)
        main_layout.addWidget(self._build_viewport_card(), stretch=1)

        body.addWidget(main)
        body.addWidget(self._build_settings_panel())
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
        self._workflow_value = self._value_label("Workflow controls unlock in Phase 8.8B2")
        self._add_status_row(layout, "Source", self._source_value)
        self._add_status_row(layout, "Workflow", self._workflow_value)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        for text in ("Live source", "Image sequence", "Recorded video"):
            button = QPushButton(text)
            button.setObjectName("reconstructionDisabledAction")
            button.setEnabled(False)
            button.setToolTip("Source selection is introduced in Phase 8.8B2.")
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return card

    def _build_runtime_card(self) -> QFrame:
        card = self._card("RUNTIME STATUS")
        layout = card.layout()

        self._live_value = self._status_value()
        self._offline_value = self._status_value()
        self._scene_value = self._status_value()
        self._add_status_row(layout, "Live SLAM", self._live_value)
        self._add_status_row(layout, "Offline", self._offline_value)
        self._add_status_row(layout, "Scene", self._scene_value)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        for text in ("Start", "Pause", "Stop"):
            button = QPushButton(text)
            button.setObjectName("reconstructionDisabledAction")
            button.setEnabled(False)
            button.setToolTip("Runtime actions are introduced in Phase 8.8B3.")
            controls.addWidget(button)
        layout.addLayout(controls)
        return card

    def _build_viewport_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionViewportCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
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

        center = QVBoxLayout()
        center.setSpacing(6)
        center.addStretch(1)
        glyph = QLabel("◇")
        glyph.setObjectName("reconstructionViewportGlyph")
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(glyph)
        headline = QLabel("Reconstruction viewport")
        headline.setObjectName("reconstructionViewportHeadline")
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(headline)
        detail = QLabel(
            "Camera trajectory, sparse/canonical geometry, scene overlays, and neural appearance views will be introduced progressively from Phase 8.8C."
        )
        detail.setObjectName("reconstructionViewportHint")
        detail.setWordWrap(True)
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(detail)
        center.addStretch(1)
        layout.addLayout(center, stretch=1)
        return card

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("reconstructionSettingsPanel")
        panel.setMinimumWidth(290)
        panel.setMaximumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("RECONSTRUCTION SETTINGS")
        title.setObjectName("reconstructionSectionTitle")
        layout.addWidget(title)

        workflow = self._settings_group("WORKFLOW")
        workflow_layout = workflow.layout()
        self._add_settings_row(workflow_layout, "Profile", "Efficient")
        self._add_settings_row(workflow_layout, "Geometry", "Canonical authority")
        self._add_settings_row(workflow_layout, "Appearance", "Deferred")
        layout.addWidget(workflow)

        session = self._settings_group("SESSION & PERSISTENCE")
        session_layout = session.layout()
        self._session_value = self._settings_value("No active session")
        self._checkpoint_value = self._settings_value("No checkpoint")
        self._scene_manifest_value = self._settings_value("No persistent scene")
        self._add_status_row(session_layout, "Session", self._session_value)
        self._add_status_row(session_layout, "Scene", self._scene_manifest_value)
        self._add_status_row(session_layout, "Checkpoint", self._checkpoint_value)
        layout.addWidget(session)

        authority = self._settings_group("AUTHORITY & SAFETY")
        authority_layout = authority.layout()
        self._add_settings_row(authority_layout, "Metric scale", "Evidence-driven")
        self._add_settings_row(authority_layout, "Metrology", "Not certified")
        self._add_settings_row(authority_layout, "Heavy backends", "Isolated workers")
        layout.addWidget(authority)
        layout.addStretch(1)
        return panel

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

    @staticmethod
    def _settings_group(title_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionSettingsGroup")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(7)
        title = QLabel(title_text)
        title.setObjectName("reconstructionSettingsGroupTitle")
        layout.addWidget(title)
        return card

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
        return label

    @staticmethod
    def _add_status_row(layout: QVBoxLayout, label_text: str, value: QLabel) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("reconstructionRowLabel")
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(value)
        layout.addLayout(row)

    def _add_settings_row(self, layout: QVBoxLayout, label_text: str, value_text: str) -> None:
        self._add_status_row(layout, label_text, self._settings_value(value_text))

    def bind_controller(self, controller) -> None:
        """Bind passive controller state. No reconstruction action is triggered here."""

        if self._controller is not None:
            raise RuntimeError("ReconstructionPage controller is already bound.")
        self._controller = controller

        controller.live_state_changed.connect(self.set_live_state)
        controller.offline_state_changed.connect(self.set_offline_state)
        controller.scene_availability_changed.connect(self.set_scene_availability)
        controller.persistent_scene_changed.connect(self.set_persistent_scene)
        controller.checkpoint_changed.connect(self.set_checkpoint)
        controller.error_occurred.connect(self.set_error)

        self.set_live_state(controller.live_state)
        self.set_offline_state(controller.offline_state)
        self.set_scene_availability(controller.scene_availability)
        scene = controller.persistent_scene
        self.set_persistent_scene(scene, str(controller.persistent_scene_manifest or ""))
        self.set_checkpoint(controller.engine_checkpoint, str(controller.engine_checkpoint_manifest or ""))

    def set_live_state(self, state: LiveCaptureRuntimeState) -> None:
        self._live_value.setText(self._humanize(state))
        self._live_value.setProperty("statusKind", self._state_kind(state.value))
        self._refresh_style(self._live_value)
        self._refresh_session_label()

    def set_offline_state(self, state: OfflineReconstructionRunState) -> None:
        self._offline_value.setText(self._humanize(state))
        self._offline_value.setProperty("statusKind", self._state_kind(state.value))
        self._refresh_style(self._offline_value)
        self._refresh_session_label()

    def set_scene_availability(self, availability: ReconstructionSceneAvailability) -> None:
        self._scene_value.setText(self._humanize(availability))
        kind = {
            ReconstructionSceneAvailability.READY: "ready",
            ReconstructionSceneAvailability.DEGRADED: "warning",
            ReconstructionSceneAvailability.PARTIAL: "running",
            ReconstructionSceneAvailability.NONE: "neutral",
        }[availability]
        self._scene_value.setProperty("statusKind", kind)
        self._refresh_style(self._scene_value)
        self._viewport_state.setText(
            "Scene ready" if availability is ReconstructionSceneAvailability.READY else "Awaiting scene"
        )

    def set_persistent_scene(self, scene, manifest_path: str) -> None:
        if scene is None:
            self._scene_manifest_value.setText("No persistent scene")
            return
        path = Path(manifest_path).name if manifest_path else "Persistent scene ready"
        self._scene_manifest_value.setText(path)

    def set_checkpoint(self, checkpoint, manifest_path: str) -> None:
        if checkpoint is None:
            self._checkpoint_value.setText("No checkpoint")
            return
        path = Path(manifest_path).name if manifest_path else "Checkpoint ready"
        self._checkpoint_value.setText(path)

    def set_error(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        self._readiness_badge.setText("CONTROLLER WARNING")
        self._readiness_badge.setProperty("statusKind", "warning")
        self._readiness_badge.setToolTip(text)
        self._refresh_style(self._readiness_badge)

    def _refresh_session_label(self) -> None:
        controller = self._controller
        if controller is None:
            return
        session_id = controller.live_session_id or controller.offline_session_id
        if session_id:
            self._session_value.setText(session_id)
        else:
            self._session_value.setText("No active session")

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
