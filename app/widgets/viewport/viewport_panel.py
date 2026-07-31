"""Interactive 3D viewport — a single immersive visualization card.

Structure (spec: layered composition, not sibling columns):

    ViewportPanel (#viewportCard)
    └── QGridLayout, one cell (0, 0)
        ├── _CanvasSurface (#viewportCanvas)  <- fills the whole cell
        └── _OverlayLayer (transparent, same cell)
            ├── _ViewportHeader        AlignLeft  | AlignTop
            ├── _ToolOverlay           AlignLeft  | AlignVCenter
            ├── _RenderModeCard        AlignRight | AlignTop
            └── _AxisIndicator         AlignLeft  | AlignBottom

Everything except the canvas floats *inside* the canvas, so there is no
separate control column and no unused region below the canvas.

_CanvasSurface itself is ALSO a single-cell QGridLayout (idle label and
the native VisPy widget share cell (0,0)) rather than a QVBoxLayout with
both stacked as separate stretch=1 items — that stacking was a real bug:
hiding the idle label doesn't remove it from the layout, so it kept
competing for vertical space with the native canvas, silently giving the
real VisPy widget only about half its rightful height. Single-cell
layering is what makes both fill the complete surface.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.icons import render_icon
from app.theme import PALETTE
from app.widgets.icon_button import IconButton

OVERLAY_MARGIN = 14
CARD_MARGIN = 10
TOOLBAR_WIDTH = 42
RENDER_CARD_WIDTH = 162


class _AxisIndicator(QWidget):
    """Small transparent XYZ gizmo (X red, Y green, Z blue), no opaque box.

    Phase 4 replaces this with a real VisPy orientation visual so it
    tracks the camera; until then it is a static hint drawn with QPainter.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(64, 64)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        origin = QPointF(14, 50)
        font = QFont()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)

        axes = (
            (QPointF(40, 50), "#e0293e", "X", QPointF(46, 53)),   # X right, red
            (QPointF(32, 34), "#2ecc71", "Y", QPointF(36, 31)),   # Y diagonal, green
            (QPointF(14, 14), "#3a7de8", "Z", QPointF(9, 11)),    # Z up, blue
        )
        for end, color, label, label_pos in axes:
            self._draw_arrow(painter, origin, end, QColor(color))
            painter.setPen(QPen(QColor(color)))
            painter.drawText(label_pos, label)
        painter.end()

    @staticmethod
    def _draw_arrow(painter: QPainter, start: QPointF, end: QPointF, color: QColor) -> None:
        """Draw a line from start to end with a filled triangular head."""

        pen = QPen(color)
        pen.setWidthF(1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        head_len = 6.0
        head_width = math.radians(24)
        # Shorten the shaft so it doesn't poke through the head tip.
        shaft_end = QPointF(
            end.x() - (head_len * 0.6) * math.cos(angle),
            end.y() - (head_len * 0.6) * math.sin(angle),
        )
        painter.drawLine(start, shaft_end)

        left = QPointF(
            end.x() - head_len * math.cos(angle - head_width),
            end.y() - head_len * math.sin(angle - head_width),
        )
        right = QPointF(
            end.x() - head_len * math.cos(angle + head_width),
            end.y() - head_len * math.sin(angle + head_width),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([end, left, right]))


class _ViewportHeader(QWidget):
    """Overlaid '● 3D VIEWPORT   ● LIVE' status line (top-left)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {PALETTE.accent}; font-size: 9px; background: transparent;")
        title = QLabel("3D VIEWPORT")
        title.setObjectName("viewportTitle")

        self._status_dot = QLabel("○")
        self._status_dot.setStyleSheet(
            f"color: {PALETTE.viewport_text_dim}; font-size: 9px; background: transparent;"
        )
        self._status_text = QLabel("IDLE")
        self._status_text.setObjectName("viewportStatus")

        row.addWidget(dot)
        row.addWidget(title)
        row.addSpacing(10)
        row.addWidget(self._status_dot)
        row.addWidget(self._status_text)
        row.addStretch(1)

    def set_live(self, is_live: bool) -> None:
        if is_live:
            self._status_dot.setText("●")
            self._status_dot.setStyleSheet(
                f"color: {PALETTE.accent}; font-size: 9px; background: transparent;"
            )
            self._status_text.setText("LIVE")
            self._status_text.setStyleSheet(
                f"color: {PALETTE.viewport_text}; font-size: 11px; "
                f"font-weight: 600; background: transparent;"
            )
        else:
            self._status_dot.setText("○")
            self._status_dot.setStyleSheet(
                f"color: {PALETTE.viewport_text_dim}; font-size: 9px; background: transparent;"
            )
            self._status_text.setText("IDLE")
            self._status_text.setStyleSheet(
                f"color: {PALETTE.viewport_text_dim}; font-size: 11px; "
                f"font-weight: 600; background: transparent;"
            )


class _ToolOverlay(QFrame):
    """Floating vertical tool strip pinned inside the canvas' left edge."""

    tool_changed = pyqtSignal(str)
    fit_requested = pyqtSignal()
    fullscreen_requested = pyqtSignal()

    TOOLS = (
        ("cursor", "select", "Select"),
        ("orbit", "orbit", "Rotate / Orbit"),
        ("pan", "pan", "Pan"),
        ("zoom", "zoom", "Zoom"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolOverlay")
        self.setFixedWidth(TOOLBAR_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        col = QVBoxLayout(self)
        col.setContentsMargins(4, 6, 4, 6)
        col.setSpacing(4)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        first: IconButton | None = None
        for icon_name, tool_id, tooltip in self.TOOLS:
            btn = IconButton(icon_name, size=32, icon_size=16, tooltip=tooltip, checkable=True)
            btn.clicked.connect(lambda _c, t=tool_id: self.tool_changed.emit(t))
            self._group.addButton(btn)
            col.addWidget(btn)
            if first is None:
                first = btn
        if first is not None:
            first.setChecked(True)

        fit_btn = IconButton("fit", size=32, icon_size=16, tooltip="Fit scene to view")
        fit_btn.clicked.connect(self.fit_requested.emit)
        col.addWidget(fit_btn)

        full_btn = IconButton("fullscreen", size=32, icon_size=16, tooltip="Full screen")
        full_btn.clicked.connect(self.fullscreen_requested.emit)
        col.addWidget(full_btn)


class _CardSlider(QWidget):
    """Compact 'LABEL / slider + value' block used inside the render card."""

    value_changed = pyqtSignal(int)

    def __init__(
        self,
        label_text: str,
        *,
        minimum: int,
        maximum: int,
        value: int,
        formatter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._formatter = formatter

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)

        caption = QLabel(label_text)
        caption.setObjectName("cardSectionTitle")
        col.addWidget(caption)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(minimum)
        self._slider.setMaximum(maximum)
        self._slider.setValue(value)

        self._value = QLabel(formatter(value))
        self._value.setObjectName("cardSliderValue")
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value.setFixedWidth(42)

        self._slider.valueChanged.connect(self._on_changed)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._value)
        col.addLayout(row)

    def _on_changed(self, value: int) -> None:
        self._value.setText(self._formatter(value))
        self.value_changed.emit(value)

    @property
    def slider(self) -> QSlider:
        return self._slider


class _RenderModeCard(QFrame):
    """Compact floating render-mode card (top-right, sized to its content)."""

    render_mode_changed = pyqtSignal(str)
    reset_view_clicked = pyqtSignal()

    MODES = ("RGB Points", "Depth Colors", "Surface Normals", "Mesh (Poisson)")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("floatingCard")
        self.setFixedWidth(RENDER_CARD_WIDTH)
        # Fixed vertical policy keeps the card sized to its contents so it
        # never stretches into a full-height sidebar.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        col = QVBoxLayout(self)
        col.setContentsMargins(12, 10, 12, 10)
        col.setSpacing(7)

        title = QLabel("RENDER MODE")
        title.setObjectName("cardSectionTitle")
        col.addWidget(title)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._current_render_mode = self.MODES[0]
        for i, mode in enumerate(self.MODES):
            radio = QRadioButton(mode)
            if i == 0:
                radio.setChecked(True)
            radio.toggled.connect(
                lambda checked, m=mode: self._on_mode_toggled(m, checked)
            )
            self._mode_group.addButton(radio)
            col.addWidget(radio)

        col.addSpacing(2)
        self._point_size = _CardSlider(
            "POINT SIZE", minimum=1, maximum=100, value=20, formatter=lambda v: f"{v / 10:.1f}"
        )
        self._density = _CardSlider(
            "DENSITY", minimum=1, maximum=100, value=100, formatter=lambda v: f"{v}%"
        )
        self._smoothing = _CardSlider(
            "SMOOTHING", minimum=0, maximum=100, value=50, formatter=lambda v: f"{v / 100:.1f}"
        )
        for widget in (self._point_size, self._density, self._smoothing):
            col.addWidget(widget)

        col.addSpacing(2)
        reset_btn = QPushButton("  Reset View")
        reset_btn.setIcon(render_icon("reset", PALETTE.viewport_text, 13))
        reset_btn.clicked.connect(self.reset_view_clicked.emit)
        col.addWidget(reset_btn)

    def _on_mode_toggled(self, mode: str, checked: bool) -> None:
        if checked:
            self._current_render_mode = mode
            self.render_mode_changed.emit(mode)

    @property
    def render_mode(self) -> str:
        return self._current_render_mode

    @property
    def point_size(self) -> float:
        return self._point_size.slider.value() / 10.0

    @property
    def density_ratio(self) -> float:
        return self._density.slider.value() / 100.0

    @property
    def smoothing(self) -> float:
        return self._smoothing.slider.value() / 100.0


class PerspectiveFloorGrid:
    """Reusable static perspective floor-grid painter for the Phase 1
    placeholder canvas.

    Draws a restrained, one-point-perspective ground plane: minor + major
    depth lines that compress toward the horizon, and perspective rays
    converging on a discreet vanishing point, with alpha fading by
    distance and toward the side edges so nothing reads as a hard line or
    a flat 2D fan.

    In Phase 4 this whole helper is replaced by a real VisPy ``Lines``
    visual on a 3D ground plane sharing the point-cloud's camera (see the
    module docstring on :class:`_CanvasSurface`); every value here is
    derived from the widget's current rect, never a fixed resolution, so
    porting the same proportions to the VisPy scene later is direct.
    """

    # Vertical extent: the grid lives in the lower ~60% of the viewport.
    VANISHING_Y_FRACTION = 0.42
    GRID_BOTTOM_FRACTION = 0.985

    # Line counts (spec: 17-21 rays, 11-14 depth lines).
    N_RAYS = 19
    N_DEPTH_LINES = 13
    MAJOR_EVERY = 4

    # How aggressively depth lines compress toward the horizon.
    # >1 = denser near the horizon, sparser near the camera.
    PERSPECTIVE_POWER = 1.7

    # Rays extend this far past each side at the near edge so the floor
    # still spans the full width close to the camera.
    RAY_OVERHANG = 1.7

    # Base colors (never pure grey/white; tuned per spec §5).
    MINOR_COLOR = "#252b32"
    MAJOR_COLOR = "#303740"
    MINOR_ALPHA_MAX = 30
    MAJOR_ALPHA_MAX = 44

    def paint(self, painter: QPainter, viewport_rect: QRectF) -> None:
        width = viewport_rect.width()
        height = viewport_rect.height()
        if width <= 0 or height <= 0:
            return

        origin_x = viewport_rect.left()
        origin_y = viewport_rect.top()
        vanishing_y = origin_y + height * self.VANISHING_Y_FRACTION
        grid_bottom = origin_y + height * self.GRID_BOTTOM_FRACTION
        vanishing_point = QPointF(origin_x + width / 2.0, vanishing_y)
        floor_span = grid_bottom - vanishing_y

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipRect(viewport_rect)

        self._draw_depth_lines(painter, origin_x, width, vanishing_y, floor_span)
        self._draw_rays(painter, origin_x, width, grid_bottom, vanishing_point)

        painter.restore()

    def _draw_depth_lines(
        self,
        painter: QPainter,
        origin_x: float,
        width: float,
        vanishing_y: float,
        floor_span: float,
    ) -> None:
        """Horizontal lines, compressed toward the horizon, fading in from
        near-zero alpha there up to full strength in the foreground."""
        for i in range(1, self.N_DEPTH_LINES + 1):
            t = i / self.N_DEPTH_LINES
            y = vanishing_y + floor_span * (t**self.PERSPECTIVE_POWER)
            is_major = (i % self.MAJOR_EVERY) == 0
            # t alone (not i) drives alpha, so the very first lines —
            # right where a "horizon band" would otherwise appear — stay
            # almost invisible regardless of how many lines there are.
            alpha_fraction = t
            pen = self._make_pen(is_major, alpha_fraction)
            painter.setPen(pen)
            painter.drawLine(QPointF(origin_x, y), QPointF(origin_x + width, y))

    def _draw_rays(
        self,
        painter: QPainter,
        origin_x: float,
        width: float,
        grid_bottom: float,
        vanishing_point: QPointF,
    ) -> None:
        """Perspective rays converging on the vanishing point, each faded
        by how far it sits from the horizontal center (edge attenuation)
        and kept faint near the vanishing point itself."""
        near_left = origin_x - width * (self.RAY_OVERHANG - 1.0) / 2.0
        near_span = width * self.RAY_OVERHANG
        center_index = self.N_RAYS / 2.0

        for i in range(self.N_RAYS + 1):
            x = near_left + near_span * i / self.N_RAYS
            is_major = (i % self.MAJOR_EVERY) == 0
            # Fade rays that land far from center (left/right edge fade);
            # never let the center ray spike to full alpha on its own.
            distance_from_center = abs(i - center_index) / center_index
            edge_fraction = 1.0 - 0.55 * distance_from_center
            pen = self._make_pen(is_major, edge_fraction)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, grid_bottom), vanishing_point)

    def _make_pen(self, is_major: bool, fraction: float) -> QPen:
        fraction = max(0.0, min(1.0, fraction))
        color_hex = self.MAJOR_COLOR if is_major else self.MINOR_COLOR
        alpha_max = self.MAJOR_ALPHA_MAX if is_major else self.MINOR_ALPHA_MAX
        color = QColor(color_hex)
        color.setAlpha(int(alpha_max * fraction))
        pen = QPen(color)
        pen.setWidthF(1.0)
        pen.setCosmetic(True)  # stays 1 device-independent pixel at any DPI
        return pen


class _CanvasSurface(QFrame):
    """Full-size host surface for the embedded VisPy canvas.

    The idle label and the native VisPy widget share the SAME grid cell
    (0, 0) — layered, not stacked. Stacking them as two items in a
    QVBoxLayout (the original Phase 1 implementation) was a real bug:
    hiding the idle label doesn't remove it from the layout, so it kept
    competing for vertical space via its own stretch factor, silently
    giving the real VisPy widget only about half its rightful height.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setObjectName("viewportCanvas")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumHeight(260)

        self._grid = PerspectiveFloorGrid()
        self._native: QWidget | None = None

        # Layer the idle label and native VisPy canvas in the same cell.
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.setRowStretch(0, 1)
        self._layout.setColumnStretch(0, 1)

        self._idle = QLabel(
            "Interactive 3D viewport\n"
            "Start inference to generate geometry"
        )
        self._idle.setObjectName("idleMessage")
        self._idle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._idle.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )

        self._layout.addWidget(
            self._idle,
            0,
            0,
            Qt.AlignmentFlag.AlignCenter,
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)

        # The static Phase 1 grid is needed only before VisPy is attached.
        if self._native is not None:
            return

        painter = QPainter(self)
        self._grid.paint(painter, QRectF(self.rect()))
        painter.end()

    def attach_native(self, native: QWidget) -> None:
        """Attach the native VisPy widget so it fills the complete surface."""

        if self._native is native:
            return

        if self._native is not None:
            self._layout.removeWidget(self._native)
            self._native.hide()
            self._native.setParent(None)

        self._native = native
        native.setParent(self)
        native.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # Prevent a model/backend widget minimum size from constraining the layout.
        native.setMinimumSize(0, 0)
        native.setMaximumSize(16_777_215, 16_777_215)

        # Same cell as the idle layer: the native canvas gets the entire area.
        self._layout.addWidget(native, 0, 0)
        native.show()
        native.raise_()

        self._idle.hide()

        self._layout.invalidate()
        self.updateGeometry()
        self.update()

    def set_idle_visible(self, visible: bool) -> None:
        self._idle.setVisible(visible)

        # Show the message above the canvas when loading or awaiting geometry.
        if visible:
            self._idle.raise_()
        elif self._native is not None:
            self._native.raise_()

    @property
    def native(self) -> QWidget | None:
        return self._native


class _OverlayHost(QWidget):
    """Small content-sized host used to inset one overlay widget.

    A full-viewport mouse-transparent overlay cannot be used here because
    ``WA_TransparentForMouseEvents`` also makes its descendants transparent
    to mouse input. Each host therefore occupies only its child's area. The
    VisPy canvas receives mouse events everywhere else, while the tool strip
    and render card remain fully interactive.
    """

    def __init__(
        self,
        child: QWidget,
        *,
        left: int = 0,
        top: int = 0,
        right: int = 0,
        bottom: int = 0,
        mouse_transparent: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(left, top, right, bottom)
        layout.setSpacing(0)
        layout.addWidget(child)

        if mouse_transparent:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)


class ViewportPanel(QFrame):
    """3D viewport card: full-bleed canvas with floating overlay controls."""

    render_mode_changed = pyqtSignal(str)
    reset_view_clicked = pyqtSignal()
    tool_changed = pyqtSignal(str)
    fit_requested = pyqtSignal()
    fullscreen_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("viewportCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._camera_initialized = False
        self._build_ui()

    def _build_ui(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(CARD_MARGIN, CARD_MARGIN, CARD_MARGIN, CARD_MARGIN)
        grid.setSpacing(0)

        # Base layer: the native VisPy widget fills the complete card.
        self._canvas = _CanvasSurface()
        grid.addWidget(self._canvas, 0, 0)

        # Overlay widgets are added directly to the same grid cell. Their
        # hosts are content-sized, so empty viewport areas remain available
        # to the VisPy camera while the interactive controls receive clicks.
        self._header = _ViewportHeader()
        self._header_host = _OverlayHost(
            self._header,
            left=OVERLAY_MARGIN,
            top=OVERLAY_MARGIN,
            mouse_transparent=True,
        )
        grid.addWidget(
            self._header_host,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self._tools = _ToolOverlay()
        self._tools.tool_changed.connect(self.tool_changed.emit)
        self._tools.fit_requested.connect(self.fit_requested.emit)
        self._tools.fullscreen_requested.connect(self.fullscreen_requested.emit)
        self._tools_host = _OverlayHost(self._tools, left=OVERLAY_MARGIN)
        grid.addWidget(
            self._tools_host,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self._render_card = _RenderModeCard()
        self._render_card.render_mode_changed.connect(self.render_mode_changed.emit)
        self._render_card.reset_view_clicked.connect(self._on_reset_view)
        self._render_host = _OverlayHost(
            self._render_card,
            top=OVERLAY_MARGIN,
            right=OVERLAY_MARGIN,
        )
        grid.addWidget(
            self._render_host,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )

        self._axis = _AxisIndicator()
        self._axis_host = _OverlayHost(
            self._axis,
            left=OVERLAY_MARGIN,
            bottom=OVERLAY_MARGIN,
            mouse_transparent=True,
        )
        grid.addWidget(
            self._axis_host,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )

        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)

    # -- Camera lifecycle (spec: fit once, never per frame) ----------------

    def _on_reset_view(self) -> None:
        self._camera_initialized = False
        self.reset_view_clicked.emit()

    @property
    def camera_initialized(self) -> bool:
        return self._camera_initialized

    def mark_camera_initialized(self) -> None:
        self._camera_initialized = True

    def invalidate_camera(self) -> None:
        """Allow one re-fit — call on a new source or new inference session."""
        self._camera_initialized = False

    # -- Public API --------------------------------------------------------

    def attach_canvas(self, native: QWidget) -> None:
        self._canvas.attach_native(native)

    def set_idle_message_visible(self, visible: bool) -> None:
        self._canvas.set_idle_visible(visible)

    def set_live(self, is_live: bool) -> None:
        self._header.set_live(is_live)

    def set_axis_visible(self, visible: bool) -> None:
        """Show or hide the single bottom-left orientation indicator."""
        self._axis_host.setVisible(visible)

    @property
    def canvas_surface(self) -> _CanvasSurface:
        return self._canvas

    @property
    def render_mode(self) -> str:
        return self._render_card.render_mode

    @property
    def point_size(self) -> float:
        return self._render_card.point_size

    @property
    def density_ratio(self) -> float:
        return self._render_card.density_ratio

    @property
    def smoothing(self) -> float:
        return self._render_card.smoothing