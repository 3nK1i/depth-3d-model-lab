"""Vector icon registry and renderer.

All interface icons are defined here as SVG path data on a 24x24 viewBox
(Lucide-style: stroked, round caps/joins, no fill). They are rendered to
QIcon / QPixmap on demand at the requested pixel size and device pixel
ratio, so they stay crisp on HiDPI displays. No emoji or Unicode glyphs
are used anywhere in the UI — see :class:`app.widgets.icon_button.IconButton`.
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QRectF
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

# Each entry is the inner markup of a 24x24 SVG (paths only). Stroke color
# is injected at render time via {stroke}. Stroke width defaults to 2.
_ICON_PATHS: dict[str, str] = {
    # Branding / window
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "minimize": '<path d="M5 12h14"/>',
    "maximize": '<rect x="4" y="4" width="16" height="16" rx="1"/>',
    "close": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    # Mode tabs
    "cube": '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "video": '<path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/>',
    "snapshot": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="3"/>',
    "reconstruction": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><rect x="8" y="8" width="8" height="8" rx="1"/>',
    "benchmark": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    # Source types
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "camera": '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
    "rtsp": '<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    # Runtime
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "pause": '<rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    # Panel toolbars
    "screenshot": '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
    "capture": '<path d="M15 3h4a2 2 0 0 1 2 2v4"/><path d="M9 21H5a2 2 0 0 1-2-2v-4"/><path d="M21 15v4a2 2 0 0 1-2 2h-4"/><path d="M3 9V5a2 2 0 0 1 2-2h4"/>',
    "fullscreen": '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "fit": '<path d="M3 3l6 6"/><path d="M3 7V3h4"/><path d="M21 3l-6 6"/><path d="M17 3h4v4"/><path d="M3 21l6-6"/><path d="M3 17v4h4"/><path d="M21 21l-6-6"/><path d="M17 21h4v-4"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
    "grid": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/>',
    "one_to_one": "",  # rendered as text "1:1", handled specially
    # Viewport tool column
    "chevron_down": '<path d="m6 9 6 6 6-6"/>',
    "chevron_up": '<path d="m18 15-6-6-6 6"/>',
    "cursor": '<path d="m3 3 7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/><path d="m13 13 6 6"/>',
    "orbit": '<circle cx="12" cy="12" r="3"/><path d="M12 3a9 9 0 0 1 0 18"/><path d="M12 3a9 9 0 0 0 0 18"/>',
    "pan": '<path d="M5 9l-3 3 3 3"/><path d="M9 5l3-3 3 3"/><path d="M15 19l-3 3-3-3"/><path d="M19 9l3 3-3 3"/><path d="M2 12h20"/><path d="M12 2v20"/>',
    "zoom": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/><path d="M11 8v6"/><path d="M8 11h6"/>',
    "reset": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    "trajectory": '<path d="M4 18c2-6 4-9 8-9s3 6 6 6 2-3 2-5"/><circle cx="4" cy="18" r="1.5"/><circle cx="20" cy="10" r="1.5"/>',
    "landmarks": '<circle cx="5" cy="7" r="1.3"/><circle cx="11" cy="5" r="1.3"/><circle cx="18" cy="8" r="1.3"/><circle cx="8" cy="13" r="1.3"/><circle cx="15" cy="14" r="1.3"/><circle cx="20" cy="18" r="1.3"/><circle cx="4" cy="19" r="1.3"/><circle cx="12" cy="20" r="1.3"/>',
}


def render_icon(name: str, color: str, size: int = 18, stroke_width: float = 2.0) -> QIcon:
    """Return a :class:`QIcon` for ``name`` stroked in ``color``.

    Renders at 2x internally for HiDPI crispness. ``one_to_one`` is a
    text glyph and returns an empty icon (draw the label separately).
    """
    return QIcon(render_pixmap(name, color, size, stroke_width))


def render_pixmap(name: str, color: str, size: int = 18, stroke_width: float = 2.0) -> QPixmap:
    """Return a HiDPI-aware :class:`QPixmap` of the named icon."""
    inner = _ICON_PATHS.get(name, "")
    scale = 2
    px = QPixmap(size * scale, size * scale)
    px.setDevicePixelRatio(scale)
    px.fill(QColor(0, 0, 0, 0))
    if not inner:
        return px

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round">{inner}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return px


def has_icon(name: str) -> bool:
    return name in _ICON_PATHS