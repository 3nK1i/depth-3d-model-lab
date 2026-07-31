"""Right settings sidebar: a vertical stack of independent accordion cards.

Structure:

    SettingsPanel
    └── QScrollArea
        └── content widget
            └── QVBoxLayout
                ├── CollapsibleSettingsCard("MODEL")            stretch=33
                ├── gap
                ├── CollapsibleSettingsCard("DEPTH SETTINGS")   stretch=36
                ├── gap
                └── CollapsibleSettingsCard("RENDER SETTINGS")  stretch=31

No stretch after the cards: when the sidebar has more height available
than the cards need, the three stretch factors redistribute it across
all three cards (in roughly 33/36/31 proportion) so Render Settings'
visible bottom border lands on the 3D viewport's bottom border, instead
of leaving a black void beneath it. Each card converts its extra share
into balanced spacing between its own rows — see
:meth:`CollapsibleSettingsCard._mark_row_stretch` — not by stretching
individual controls.

The sidebar itself is a plain surface with no card styling; each card
brings its own background, border and padding.

Phase 1 note: values are presets — the model registry starts driving the
MODEL card in Phase 5.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.widgets.settings_card import CollapsibleSettingsCard
from app.widgets.settings_controls import (
    SettingsComboBox,
    SettingsSlider,
    SliderValueBinding,
    StatusBadge,
    ValueBox,
    make_toggle,
)

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
    """Scrollable stack of MODEL / DEPTH SETTINGS / RENDER SETTINGS cards."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # No "panel" objectName here: the sidebar is a bare surface, the
        # cards supply the visible chrome.
        self.setObjectName("settingsSidebar")
        self.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.setMaximumWidth(SIDEBAR_MAX_WIDTH)
        self._bindings: list[SliderValueBinding] = []
        self._build_ui()

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
        # Zero outer margins: the page grid provides the gutters, so the
        # first card's top edge aligns exactly with the Original/Depth row
        # and the last card can end level with the 3D viewer.
        layout.setContentsMargins(
            SIDEBAR_PADDING_H, SIDEBAR_PADDING_V, SIDEBAR_PADDING_H, SIDEBAR_PADDING_V
        )
        layout.setSpacing(CARD_GAP)

        # Stretch factors distribute leftover sidebar height across all
        # three cards (33/36/31) instead of collecting below the last one.
        # No addStretch() after them: that is what created the black void
        # beneath Render Settings.
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
            # On collapse, free this card's stretch share (drop to 0) so
            # the still-expanded cards absorb the released height instead
            # of a blank gap appearing where this card used to be.
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

        self._model_combo = SettingsComboBox(["MoGe-2 ViT-L Normal", "No model registered yet"])
        self._model_badge = StatusBadge("RECOMMENDED")
        # Selector and badge share one row; the badge never wraps.
        card.add_wide_row(self._model_combo, self._model_badge)

        self._gpu_profile = SettingsComboBox(
            ["Auto (RTX 5090)", "Efficient", "Balanced", "Ultra", "Custom"]
        )
        card.add_control_row("GPU Profile", self._gpu_profile)

        self._quality = SettingsComboBox(["Ultra", "Balanced", "Efficient"])
        card.add_control_row("Quality Preset", self._quality)

        self._resolution = SettingsComboBox(["768 px", "448 px", "384 px"])
        card.add_control_row("Inference Resolution", self._resolution)

        self._precision = SettingsComboBox(["FP16 (Half)", "FP32", "BF16"])
        card.add_control_row("Precision", self._precision)
        return card

    def _build_depth_card(self) -> CollapsibleSettingsCard:
        card = CollapsibleSettingsCard("DEPTH SETTINGS", accent_marker=True)

        self._colormap = SettingsComboBox(["Turbo", "Inferno", "Magma", "Viridis", "Greyscale"])
        card.add_control_row("Colormap", self._colormap)

        self._min_depth_slider = SettingsSlider(0, 100, 1)
        min_box = ValueBox()
        self._bindings.append(SliderValueBinding(self._min_depth_slider, min_box, _fmt_metres))
        card.add_slider_row("Min Depth", self._min_depth_slider, min_box)

        self._max_depth_slider = SettingsSlider(0, 200, 80)
        max_box = ValueBox()
        self._bindings.append(SliderValueBinding(self._max_depth_slider, max_box, _fmt_metres))
        card.add_slider_row("Max Depth", self._max_depth_slider, max_box)

        self._edge_filter_slider = SettingsSlider(0, 100, 60)
        edge_box = ValueBox()
        self._bindings.append(SliderValueBinding(self._edge_filter_slider, edge_box, _fmt_ratio))
        card.add_slider_row("Edge Filtering", self._edge_filter_slider, edge_box)

        self._temporal_smoothing_slider = SettingsSlider(0, 100, 40)
        smoothing_box = ValueBox()
        self._bindings.append(
            SliderValueBinding(self._temporal_smoothing_slider, smoothing_box, _fmt_ratio)
        )
        card.add_slider_row("Temporal Smoothing", self._temporal_smoothing_slider, smoothing_box)

        self._bg_removal = make_toggle(True)
        card.add_toggle_row("Background Removal", self._bg_removal)
        return card

    def _build_render_card(self) -> CollapsibleSettingsCard:
        card = CollapsibleSettingsCard("RENDER SETTINGS", accent_marker=True)

        self._background = SettingsComboBox(["Dark", "Black", "White", "Transparent"])
        card.add_control_row("Background", self._background)

        self._grid_toggle = make_toggle(True)
        card.add_toggle_row("Grid", self._grid_toggle)

        self._axes_toggle = make_toggle(True)
        card.add_toggle_row("Axes", self._axes_toggle)

        self._point_budget = SettingsComboBox(["500K", "1M", "250K", "100K"])
        card.add_control_row("Point Budget", self._point_budget)

        self._vsync_toggle = make_toggle(True)
        card.add_toggle_row("VSync", self._vsync_toggle)
        return card

    # -- Public API: live-read values for the geometry pipeline -------------

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