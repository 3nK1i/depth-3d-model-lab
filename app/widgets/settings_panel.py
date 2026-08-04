"""Right settings sidebar and hardware-aware model profile controls.

The three independent accordion cards preserve the Phase 4 layout. Phase 5
replaces the MODEL card's static GPU presets with a real startup
hardware snapshot and a typed Efficient/Balanced/Ultra resolver. The
resolved config is exposed through :attr:`model_runtime_config`; model loading
still happens only in the worker and no CUDA tensors are allocated in the UI
process.
"""

from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.settings import AppSettings
from app.widgets.settings_card import CollapsibleSettingsCard
from app.widgets.settings_controls import (
    SettingsComboBox,
    SettingsSlider,
    SliderValueBinding,
    StatusBadge,
    ValueBox,
    make_toggle,
)
from core.hardware.gpu_info import HardwareInfo, detect_hardware
from core.hardware.profiles import (
    ComputeProfile,
    ResolvedProfile,
    resolve_profile,
)
from core.models.metadata import ModelDescriptor, ModelProfileMetadata
from core.models.registry import (
    check_model_availability,
    get_default_model_descriptor,
    get_model_descriptor,
    iter_model_descriptors,
    resolve_model_profile_metadata,
    resolve_model_runtime_config,
)
from core.models.runtime_config import ModelRuntimeContext

SIDEBAR_MIN_WIDTH = 320
SIDEBAR_PREFERRED_WIDTH = 348
SIDEBAR_MAX_WIDTH = 370
SIDEBAR_PADDING_H = 0
SIDEBAR_PADDING_V = 0
CARD_GAP = 8

def _fmt_metres(v: int) -> str:
    return f"{v / 10:.1f} m"


def _fmt_ratio(v: int) -> str:
    return f"{v / 100:.1f}"


class SettingsPanel(QFrame):
    """Scrollable MODEL / DEPTH SETTINGS / RENDER SETTINGS cards."""

    compute_profile_changed = pyqtSignal(object)  # ResolvedProfile
    model_selection_changed = pyqtSignal(str)

    # Live viewport/depth settings. These signals let the controller refresh
    # an already displayed static result immediately, without running the AI
    # model again.
    depth_visual_changed = pyqtSignal()
    geometry_settings_changed = pyqtSignal()
    temporal_smoothing_changed = pyqtSignal(float)
    background_changed = pyqtSignal(str)
    grid_visibility_changed = pyqtSignal(bool)
    axes_visibility_changed = pyqtSignal(bool)
    point_budget_changed = pyqtSignal(int)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        hardware_info: HardwareInfo | None = None,
        app_settings: AppSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsSidebar")
        self.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.setMaximumWidth(SIDEBAR_MAX_WIDTH)

        self._hardware_info = hardware_info or detect_hardware()
        self._app_settings = app_settings or AppSettings()
        self._resolved_profile: ResolvedProfile | None = None
        self._profile_metadata: ModelProfileMetadata | None = None
        self._bindings: list[SliderValueBinding] = []
        self._model_controls_locked = False

        self._build_ui()
        self._initialize_compute_profile()
        self._initialize_model_selection()

    def sizeHint(self):  # noqa: N802 - Qt override
        hint = super().sizeHint()
        hint.setWidth(SIDEBAR_PREFERRED_WIDTH)
        return hint

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("settingsContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(
            SIDEBAR_PADDING_H,
            SIDEBAR_PADDING_V,
            SIDEBAR_PADDING_H,
            SIDEBAR_PADDING_V,
        )
        layout.setSpacing(CARD_GAP)

        model_card = self._build_model_card()
        depth_card = self._build_depth_card()
        render_card = self._build_render_card()

        self._card_stretches: dict[CollapsibleSettingsCard, int] = {
            model_card: 33,
            depth_card: 36,
            render_card: 31,
        }
        for card, stretch in self._card_stretches.items():
            layout.addWidget(card, stretch)
            card.expansion_changed.connect(
                lambda expanded, c=card, s=stretch: layout.setStretchFactor(
                    c, s if expanded else 0
                )
            )

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # -- Cards -------------------------------------------------------------

    def _build_model_card(self) -> CollapsibleSettingsCard:
        card = CollapsibleSettingsCard("MODEL", uppercase_labels=True)

        self._model_combo = SettingsComboBox()
        for descriptor in iter_model_descriptors(
            selectable_only=True,
            enabled_only=True,
        ):
            self._model_combo.addItem(descriptor.display_name, descriptor.model_id)
        self._model_combo.currentIndexChanged.connect(self._on_model_selected)
        self._model_combo.setToolTip(
            "Choose a registry-declared adapter to load in the isolated inference worker."
        )
        self._model_badge = StatusBadge("READY")
        card.add_wide_row(self._model_combo, self._model_badge)

        self._gpu_profile = SettingsComboBox()
        self._populate_profile_combo()
        self._gpu_profile.currentIndexChanged.connect(self._on_profile_selected)
        card.add_control_row("GPU Profile", self._gpu_profile)

        # Registry-resolved outputs. They remain read-only until a dedicated
        # custom profile workflow is added in a later phase.
        self._model_variant = SettingsComboBox()
        self._model_variant.setEnabled(False)
        self._model_variant.setToolTip(
            "Registry-resolved model variant for the selected adapter and profile."
        )
        card.add_control_row("Model Variant", self._model_variant)

        self._resolution = SettingsComboBox()
        self._resolution.setEnabled(False)
        self._resolution.setToolTip(
            "Registry-resolved inference detail for the selected adapter."
        )
        card.add_control_row("Inference Detail", self._resolution)

        self._precision = SettingsComboBox()
        self._precision.setEnabled(False)
        self._precision.setToolTip(
            "Registry-resolved numeric precision for the selected adapter."
        )
        card.add_control_row("Precision", self._precision)
        return card

    def _build_depth_card(self) -> CollapsibleSettingsCard:
        card = CollapsibleSettingsCard("DEPTH SETTINGS", accent_marker=True)

        self._colormap = SettingsComboBox(["Turbo", "Inferno", "Magma", "Viridis", "Greyscale"])
        self._colormap.currentTextChanged.connect(
            lambda _text: self.depth_visual_changed.emit()
        )
        card.add_control_row("Colormap", self._colormap)

        self._min_depth_slider = SettingsSlider(0, 100, 1)
        min_box = ValueBox()
        self._bindings.append(SliderValueBinding(self._min_depth_slider, min_box, _fmt_metres))
        self._min_depth_slider.valueChanged.connect(
            lambda value: self._on_depth_slider_changed("min", value)
        )
        card.add_slider_row("Min Depth", self._min_depth_slider, min_box)

        self._max_depth_slider = SettingsSlider(0, 200, 80)
        max_box = ValueBox()
        self._bindings.append(SliderValueBinding(self._max_depth_slider, max_box, _fmt_metres))
        self._max_depth_slider.valueChanged.connect(
            lambda value: self._on_depth_slider_changed("max", value)
        )
        card.add_slider_row("Max Depth", self._max_depth_slider, max_box)

        self._edge_filter_slider = SettingsSlider(0, 100, 60)
        edge_box = ValueBox()
        self._bindings.append(SliderValueBinding(self._edge_filter_slider, edge_box, _fmt_ratio))
        self._edge_filter_slider.valueChanged.connect(
            lambda _value: self.geometry_settings_changed.emit()
        )
        card.add_slider_row("Edge Filtering", self._edge_filter_slider, edge_box)

        self._temporal_smoothing_slider = SettingsSlider(0, 100, 40)
        smoothing_box = ValueBox()
        self._bindings.append(
            SliderValueBinding(self._temporal_smoothing_slider, smoothing_box, _fmt_ratio)
        )
        self._temporal_smoothing_slider.valueChanged.connect(
            lambda value: self.temporal_smoothing_changed.emit(value / 100.0)
        )
        self._temporal_smoothing_slider.setToolTip(
            "Exponential smoothing for consecutive video/camera/RTSP depth and "
            "geometry maps. Static images are never temporally blended."
        )
        card.add_slider_row("Temporal Smoothing", self._temporal_smoothing_slider, smoothing_box)

        self._bg_removal = make_toggle(True)
        self._bg_removal.setToolTip(
            "When enabled, use the model validity mask and the selected min/max "
            "depth range to remove invalid and out-of-range geometry."
        )
        self._bg_removal.toggled_changed.connect(
            lambda _checked: self._emit_depth_and_geometry_changed()
        )
        card.add_toggle_row("Background Removal", self._bg_removal)
        return card

    def _build_render_card(self) -> CollapsibleSettingsCard:
        card = CollapsibleSettingsCard("RENDER SETTINGS", accent_marker=True)

        self._background = SettingsComboBox(["Dark", "Black", "White", "Transparent"])
        self._background.currentTextChanged.connect(
            lambda text: self.background_changed.emit(text.lower())
        )
        card.add_control_row("Background", self._background)

        self._grid_toggle = make_toggle(True)
        self._grid_toggle.toggled_changed.connect(self.grid_visibility_changed.emit)
        card.add_toggle_row("Grid", self._grid_toggle)

        self._axes_toggle = make_toggle(True)
        self._axes_toggle.toggled_changed.connect(self.axes_visibility_changed.emit)
        card.add_toggle_row("Axes", self._axes_toggle)

        self._point_budget = SettingsComboBox(["500K", "1M", "250K", "100K"])
        self._point_budget.currentIndexChanged.connect(self._emit_point_budget_changed)
        card.add_control_row("Point Budget", self._point_budget)

        # The Qt/OpenGL swap interval is selected when the SceneCanvas is
        # constructed. Runtime switching is deliberately not exposed because
        # recreating the canvas would destroy the live camera/visual state.
        self._vsync_toggle = make_toggle(True)
        self._vsync_toggle.setEnabled(False)
        self._vsync_toggle.setToolTip(
            "VSync is enabled when the VisPy canvas is created. Runtime switching "
            "is disabled to preserve the OpenGL context and current 3D view."
        )
        card.add_toggle_row("VSync", self._vsync_toggle)
        return card

    def _on_depth_slider_changed(self, changed: str, _value: int) -> None:
        """Keep the near/far interval valid while either slider is dragged."""
        near = self._min_depth_slider.value()
        far = self._max_depth_slider.value()
        if near >= far:
            if changed == "min":
                if near < self._max_depth_slider.maximum():
                    blocker = QSignalBlocker(self._max_depth_slider)
                    self._max_depth_slider.setValue(near + 1)
                    del blocker
                else:
                    blocker = QSignalBlocker(self._min_depth_slider)
                    self._min_depth_slider.setValue(max(self._min_depth_slider.minimum(), far - 1))
                    del blocker
            else:
                if far > self._min_depth_slider.minimum():
                    blocker = QSignalBlocker(self._min_depth_slider)
                    self._min_depth_slider.setValue(far - 1)
                    del blocker
                else:
                    blocker = QSignalBlocker(self._max_depth_slider)
                    self._max_depth_slider.setValue(min(self._max_depth_slider.maximum(), near + 1))
                    del blocker
        self._emit_depth_and_geometry_changed()

    def _emit_depth_and_geometry_changed(self, _value: object | None = None) -> None:
        self.depth_visual_changed.emit()
        self.geometry_settings_changed.emit()

    def _emit_point_budget_changed(self, _index: int) -> None:
        self.point_budget_changed.emit(self.point_budget)
        self.geometry_settings_changed.emit()

    # -- Active-model lifecycle -------------------------------------------

    def _initialize_model_selection(self) -> None:
        loader = getattr(self._app_settings, "load_last_model_id", None)
        saved = loader() if callable(loader) else None
        selectable_ids = {
            descriptor.model_id
            for descriptor in iter_model_descriptors(
                selectable_only=True,
                enabled_only=True,
            )
        }
        default_id = get_default_model_descriptor().model_id
        model_id = saved if saved in selectable_ids else default_id
        index = self._model_combo.findData(model_id)
        _set_combo_index_blocked(self._model_combo, max(index, 0))
        self._apply_profile(self.resolved_profile.requested, emit_signal=False)
        self._refresh_model_badge()
        self._sync_profile_control_state()

    def _on_model_selected(self, _index: int) -> None:
        saver = getattr(self._app_settings, "save_last_model_id", None)
        if callable(saver):
            saver(self.selected_model_id)
        self._apply_profile(self.resolved_profile.requested, emit_signal=False)
        self._refresh_model_badge()
        self._sync_profile_control_state()
        self.model_selection_changed.emit(self.selected_model_id)

    def _refresh_model_badge(self) -> None:
        descriptor = self.selected_model_descriptor
        availability = check_model_availability(
            descriptor.model_id,
            self._hardware_info,
        )
        if descriptor.development_only:
            self._model_badge.setText(descriptor.presentation.idle_badge)
            self._model_badge.setToolTip(descriptor.presentation.description)
            self._model_combo.setToolTip(descriptor.presentation.description)
            return
        if descriptor.uses_compute_profile and self.resolved_profile.downgraded:
            self._model_badge.setText("SAFE FALLBACK")
        elif self._hardware_info.cuda_ready:
            self._model_badge.setText("GPU READY")
        else:
            self._model_badge.setText("CPU MODE")

        tooltip_parts = [descriptor.presentation.description, self._hardware_info.details]
        if availability.reason:
            tooltip_parts.append(f"Availability: {availability.reason}")
        if descriptor.uses_compute_profile and self.resolved_profile.warning:
            tooltip_parts.append(f"Profile: {self.resolved_profile.warning}")
        tooltip = "\n".join(v for v in tooltip_parts if v)
        self._model_badge.setToolTip(tooltip)
        self._model_combo.setToolTip(tooltip)

    def set_model_controls_locked(self, locked: bool) -> None:
        """Prevent config/model changes while a worker model is loaded."""
        self._model_controls_locked = bool(locked)
        self._model_combo.setEnabled(not locked)
        self._sync_profile_control_state()

    def _sync_profile_control_state(self) -> None:
        enabled = (
            not self._model_controls_locked
            and self.selected_model_descriptor.uses_compute_profile
        )
        self._gpu_profile.setEnabled(enabled)

    def set_model_loading(self) -> None:
        self.set_model_stage("LOADING", "Loading model weights.")

    def set_model_stage(self, label: str, message: str = "") -> None:
        """Display a worker lifecycle stage and lock unsafe controls."""
        self._model_badge.setText(str(label).upper())
        self._model_badge.setToolTip(message)
        self.set_model_controls_locked(True)

    def set_model_loaded(self, metadata: dict | None) -> None:
        metadata = metadata or {}
        descriptor = self.selected_model_descriptor
        self._model_badge.setText(descriptor.presentation.active_badge)
        details = [
            str(metadata.get("display_name", "")),
            str(metadata.get("device", "")),
            str(metadata.get("precision", "")),
        ]
        self._model_badge.setToolTip(" · ".join(v for v in details if v))
        # The controller unlocks these controls only in safe idle/result states.

    def set_model_unloaded(self) -> None:
        self.set_model_controls_locked(False)
        self._refresh_model_badge()

    def set_model_error(
        self,
        message: str,
        *,
        label: str = "ERROR",
        unlock_controls: bool = True,
    ) -> None:
        self._model_badge.setText(label)
        self._model_badge.setToolTip(message)
        self.set_model_controls_locked(not unlock_controls)

    # -- Hardware/profile lifecycle ---------------------------------------

    def _populate_profile_combo(self) -> None:
        auto_label = f"Auto ({self._hardware_info.short_label})"
        self._gpu_profile.addItem(auto_label, ComputeProfile.AUTO.value)
        for profile in _CONCRETE_PROFILES:
            self._gpu_profile.addItem(profile.display_name, profile.value)
        self._gpu_profile.setToolTip(self._hardware_info.details)

    def _initialize_compute_profile(self) -> None:
        saved = self._app_settings.load_compute_profile()
        try:
            requested = ComputeProfile(saved) if saved else ComputeProfile.AUTO
        except ValueError:
            requested = ComputeProfile.AUTO

        index = self._gpu_profile.findData(requested.value)
        _set_combo_index_blocked(self._gpu_profile, max(index, 0))
        self._apply_profile(requested, emit_signal=False)

    def _on_profile_selected(self, _index: int) -> None:
        raw = self._gpu_profile.currentData()
        try:
            requested = ComputeProfile(str(raw))
        except ValueError:
            requested = ComputeProfile.AUTO
        self._apply_profile(requested, emit_signal=True)

    def _apply_profile(self, requested: ComputeProfile, *, emit_signal: bool) -> None:
        resolved = resolve_profile(requested, self._hardware_info)
        self._resolved_profile = resolved
        self._app_settings.save_compute_profile(requested.value)

        profile_metadata = resolve_model_profile_metadata(
            self.selected_model_id,
            resolved,
        )
        self._profile_metadata = profile_metadata
        _set_combo_single_value(self._model_variant, profile_metadata.variant_label)
        _set_combo_single_value(self._resolution, profile_metadata.detail_label)
        _set_combo_single_value(self._precision, profile_metadata.precision_label)

        budget_label = _point_budget_label(profile_metadata.point_budget)
        budget_index = self._point_budget.findText(budget_label)
        if budget_index < 0:
            self._point_budget.addItem(budget_label)
            budget_index = self._point_budget.findText(budget_label)
        _set_combo_index_blocked(self._point_budget, max(budget_index, 0))

        live_cap = profile_metadata.live_point_budget
        self._point_budget.setToolTip(
            "Maximum point count for a static image. Lower budgets resample the "
            "cached result immediately; requesting more points than the worker "
            "returned takes effect on the next inference. "
            f"Continuous video/camera/RTSP rendering is safety-capped at "
            f"{_point_budget_label(live_cap)} for the resolved registry profile."
        )

        tooltip_parts = [self._hardware_info.details, profile_metadata.tooltip]
        profile_tooltip = "\n".join(part for part in tooltip_parts if part)
        self._gpu_profile.setToolTip(profile_tooltip)

        self._refresh_model_badge()
        if self.selected_model_descriptor.uses_compute_profile:
            self._model_badge.setToolTip(profile_tooltip)

        if emit_signal:
            self.compute_profile_changed.emit(resolved)

    # -- Public API: hardware/model profile -------------------------------

    @property
    def selected_model_id(self) -> str:
        value = self._model_combo.currentData()
        return str(value) if value else get_default_model_descriptor().model_id

    @property
    def selected_model_descriptor(self) -> ModelDescriptor:
        return get_model_descriptor(self.selected_model_id)

    @property
    def selected_model_label(self) -> str:
        return self._model_combo.currentText()

    @property
    def hardware_info(self) -> HardwareInfo:
        return self._hardware_info

    @property
    def resolved_profile(self) -> ResolvedProfile:
        assert self._resolved_profile is not None
        return self._resolved_profile

    @property
    def model_runtime_config(self) -> dict[str, object]:
        return resolve_model_runtime_config(
            self.selected_model_id,
            ModelRuntimeContext(
                hardware_info=self._hardware_info,
                resolved_profile=self.resolved_profile,
                point_budget=self.point_budget,
            ),
        )

    # -- Public API: live-read geometry settings --------------------------

    def set_colormap_name(self, name: str) -> None:
        """Select a colormap from the compact Depth Map toolbar."""
        target = str(name).strip().lower()
        for index in range(self._colormap.count()):
            if self._colormap.itemText(index).strip().lower() == target:
                self._colormap.setCurrentIndex(index)
                return

    def set_depth_range(self, near_m: float, far_m: float) -> None:
        """Apply a valid metric interval from the compact depth controls."""
        near_value = int(round(float(near_m) * 10.0))
        far_value = int(round(float(far_m) * 10.0))
        near_value = max(self._min_depth_slider.minimum(), min(self._min_depth_slider.maximum(), near_value))
        far_value = max(self._max_depth_slider.minimum(), min(self._max_depth_slider.maximum(), far_value))
        if near_value >= far_value:
            if near_value < self._max_depth_slider.maximum():
                far_value = near_value + 1
            else:
                near_value = max(self._min_depth_slider.minimum(), far_value - 1)

        min_blocker = QSignalBlocker(self._min_depth_slider)
        max_blocker = QSignalBlocker(self._max_depth_slider)
        self._min_depth_slider.setValue(near_value)
        self._max_depth_slider.setValue(far_value)
        del min_blocker, max_blocker
        self._emit_depth_and_geometry_changed()

    @property
    def colormap_name(self) -> str:
        return self._colormap.currentText()

    @property
    def min_depth_m(self) -> float:
        return self._min_depth_slider.value() / 10.0

    @property
    def max_depth_m(self) -> float:
        return self._max_depth_slider.value() / 10.0

    @property
    def edge_filter_strength(self) -> float:
        return self._edge_filter_slider.value() / 100.0

    @property
    def temporal_smoothing(self) -> float:
        return self._temporal_smoothing_slider.value() / 100.0

    @property
    def background_removal_enabled(self) -> bool:
        return self._bg_removal.isChecked()

    @property
    def background_name(self) -> str:
        return self._background.currentText().strip().lower()

    @property
    def grid_visible(self) -> bool:
        return self._grid_toggle.isChecked()

    @property
    def axes_visible(self) -> bool:
        return self._axes_toggle.isChecked()

    @property
    def point_budget(self) -> int:
        text = self._point_budget.currentText().upper()
        if text.endswith("M"):
            return int(float(text[:-1]) * 1_000_000)
        if text.endswith("K"):
            return int(float(text[:-1]) * 1_000)
        return int(text)

    @property
    def live_point_budget(self) -> int:
        """Safe point-cloud cap for continuous video/camera/RTSP sources."""
        metadata = self._profile_metadata
        cap = metadata.live_point_budget if metadata is not None else self.point_budget
        return min(self.point_budget, cap)


_CONCRETE_PROFILES = (
    ComputeProfile.EFFICIENT,
    ComputeProfile.BALANCED,
    ComputeProfile.ULTRA,
)


def _point_budget_label(point_budget: int) -> str:
    if point_budget >= 1_000_000:
        return f"{point_budget / 1_000_000:g}M"
    return f"{point_budget // 1_000}K"


def _set_combo_index_blocked(combo: SettingsComboBox, index: int) -> None:
    blocker = QSignalBlocker(combo)
    combo.setCurrentIndex(index)
    del blocker


def _set_combo_single_value(combo: SettingsComboBox, value: str) -> None:
    blocker = QSignalBlocker(combo)
    combo.clear()
    combo.addItem(str(value))
    combo.setCurrentIndex(0)
    del blocker