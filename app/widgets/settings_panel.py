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
    PROFILE_SPECS,
    ComputeProfile,
    ResolvedProfile,
    resolve_profile,
)
from core.models.catalog import (
    MOGE2_DISPLAY_LABEL,
    MOGE2_MODEL_ID,
    MOCK_MODEL_ID,
)

SIDEBAR_MIN_WIDTH = 320
SIDEBAR_PREFERRED_WIDTH = 348
SIDEBAR_MAX_WIDTH = 370
SIDEBAR_PADDING_H = 0
SIDEBAR_PADDING_V = 0
CARD_GAP = 8

_LIVE_POINT_BUDGET_CAPS: dict[ComputeProfile, int] = {
    ComputeProfile.EFFICIENT: 100_000,
    ComputeProfile.BALANCED: 250_000,
    ComputeProfile.ULTRA: 500_000,
}


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
        self._bindings: list[SliderValueBinding] = []

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
        self._model_combo.addItem(MOGE2_DISPLAY_LABEL, MOGE2_MODEL_ID)
        self._model_combo.addItem("Development Mock", MOCK_MODEL_ID)
        self._model_combo.currentIndexChanged.connect(self._on_model_selected)
        self._model_combo.setToolTip(
            "Choose the adapter to load in the isolated inference worker. "
            "MoGe-2 is the default; the development mock remains available "
            "only for lifecycle diagnostics."
        )
        self._model_badge = StatusBadge("READY")
        card.add_wide_row(self._model_combo, self._model_badge)

        self._gpu_profile = SettingsComboBox()
        self._populate_profile_combo()
        self._gpu_profile.currentIndexChanged.connect(self._on_profile_selected)
        card.add_control_row("GPU Profile", self._gpu_profile)

        # These rows are profile outputs in Step 1. They become editable only
        # when a dedicated Custom profile is introduced later.
        self._model_variant = SettingsComboBox(
            [PROFILE_SPECS[p].model_label for p in _CONCRETE_PROFILES]
        )
        self._model_variant.setEnabled(False)
        self._model_variant.setToolTip(
            "Resolved MoGe-2 variant for the selected hardware profile."
        )
        card.add_control_row("Model Variant", self._model_variant)

        self._resolution = SettingsComboBox(
            [
                _resolution_label(PROFILE_SPECS[p].resolution_level, PROFILE_SPECS[p].num_tokens)
                for p in _CONCRETE_PROFILES
            ]
        )
        self._resolution.setEnabled(False)
        self._resolution.setToolTip(
            "Resolved inference detail for the selected hardware profile."
        )
        card.add_control_row("Inference Detail", self._resolution)

        self._precision = SettingsComboBox(["FP16 (Half)", "FP32", "BF16"])
        self._precision.setEnabled(False)
        self._precision.setToolTip(
            "Resolved numeric precision for the selected hardware profile."
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
        self._min_depth_slider.valueChanged.connect(self._emit_depth_and_geometry_changed)
        card.add_slider_row("Min Depth", self._min_depth_slider, min_box)

        self._max_depth_slider = SettingsSlider(0, 200, 80)
        max_box = ValueBox()
        self._bindings.append(SliderValueBinding(self._max_depth_slider, max_box, _fmt_metres))
        self._max_depth_slider.valueChanged.connect(self._emit_depth_and_geometry_changed)
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
        model_id = saved if saved in {MOGE2_MODEL_ID, MOCK_MODEL_ID} else MOGE2_MODEL_ID
        index = self._model_combo.findData(model_id)
        _set_combo_index_blocked(self._model_combo, max(index, 0))
        self._refresh_model_badge()

    def _on_model_selected(self, _index: int) -> None:
        saver = getattr(self._app_settings, "save_last_model_id", None)
        if callable(saver):
            saver(self.selected_model_id)
        self._refresh_model_badge()
        self.model_selection_changed.emit(self.selected_model_id)

    def _refresh_model_badge(self) -> None:
        if self.selected_model_id == MOCK_MODEL_ID:
            self._model_badge.setText("DEV ONLY")
            self._model_badge.setToolTip(
                "Synthetic depth used only to test the runtime lifecycle."
            )
            return
        if self.resolved_profile.downgraded:
            self._model_badge.setText("SAFE FALLBACK")
        elif self._hardware_info.cuda_ready:
            self._model_badge.setText("GPU READY")
        else:
            self._model_badge.setText("CPU MODE")

        tooltip_parts = [self._hardware_info.details]
        if self.resolved_profile.warning:
            tooltip_parts.append(f"Profile: {self.resolved_profile.warning}")
        self._model_badge.setToolTip("\n".join(v for v in tooltip_parts if v))

    def set_model_controls_locked(self, locked: bool) -> None:
        """Prevent config/model changes while a worker model is loaded."""
        self._model_combo.setEnabled(not locked)
        self._gpu_profile.setEnabled(not locked)

    def set_model_loading(self) -> None:
        self._model_badge.setText("LOADING")
        self.set_model_controls_locked(True)

    def set_model_loaded(self, metadata: dict | None) -> None:
        metadata = metadata or {}
        if self.selected_model_id == MOCK_MODEL_ID:
            self._model_badge.setText("MOCK ACTIVE")
        elif bool(metadata.get("is_metric")):
            self._model_badge.setText("METRIC READY")
        else:
            self._model_badge.setText("READY")
        details = [
            str(metadata.get("display_name", "")),
            str(metadata.get("device", "")),
            str(metadata.get("precision", "")),
        ]
        self._model_badge.setToolTip(" · ".join(v for v in details if v))
        self.set_model_controls_locked(True)

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

        variant_index = _CONCRETE_PROFILES.index(resolved.effective)
        _set_combo_index_blocked(self._model_variant, variant_index)
        _set_combo_index_blocked(self._resolution, variant_index)

        precision_index = self._precision.findText(resolved.precision.display_name)
        _set_combo_index_blocked(self._precision, max(precision_index, 0))

        budget_label = _point_budget_label(resolved.spec.point_budget)
        budget_index = self._point_budget.findText(budget_label)
        if budget_index >= 0:
            _set_combo_index_blocked(self._point_budget, budget_index)

        live_cap = _LIVE_POINT_BUDGET_CAPS[resolved.effective]
        self._point_budget.setToolTip(
            "Maximum point count for a static image. Lower budgets resample the "
            "cached result immediately; requesting more points than the worker "
            "returned takes effect on the next inference. "
            f"Continuous video/camera/RTSP rendering is safety-capped at "
            f"{_point_budget_label(live_cap)} for the "
            f"{resolved.effective.display_name} profile to keep the UI responsive."
        )

        tooltip_parts = [self._hardware_info.details]
        if resolved.warning:
            tooltip_parts.append(f"Profile: {resolved.warning}")
        profile_tooltip = "\n".join(part for part in tooltip_parts if part)
        self._gpu_profile.setToolTip(profile_tooltip)

        self._refresh_model_badge()
        if self.selected_model_id == MOGE2_MODEL_ID:
            self._model_badge.setToolTip(profile_tooltip)

        if emit_signal:
            self.compute_profile_changed.emit(resolved)

    # -- Public API: hardware/model profile -------------------------------

    @property
    def selected_model_id(self) -> str:
        value = self._model_combo.currentData()
        return str(value) if value else MOGE2_MODEL_ID

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
        config = self.resolved_profile.as_model_config()
        config["point_budget"] = self.point_budget
        return config

    # -- Public API: live-read geometry settings --------------------------

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
        cap = _LIVE_POINT_BUDGET_CAPS[self.resolved_profile.effective]
        return min(self.point_budget, cap)


_CONCRETE_PROFILES = (
    ComputeProfile.EFFICIENT,
    ComputeProfile.BALANCED,
    ComputeProfile.ULTRA,
)


def _resolution_label(level: int, num_tokens: int) -> str:
    return f"Level {level} · {num_tokens} tok"


def _point_budget_label(point_budget: int) -> str:
    if point_budget >= 1_000_000:
        return f"{point_budget / 1_000_000:g}M"
    return f"{point_budget // 1_000}K"


def _set_combo_index_blocked(combo: SettingsComboBox, index: int) -> None:
    blocker = QSignalBlocker(combo)
    combo.setCurrentIndex(index)
    del blocker
