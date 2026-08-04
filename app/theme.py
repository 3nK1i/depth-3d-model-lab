"""Dark industrial theme for EIA TECH — Depth & 3D Model Lab.

Centralizes color constants and the global stylesheet (QSS) so no widget
hardcodes a color value. Widgets reference :data:`PALETTE` directly when
they need a color outside the stylesheet (e.g. drawing on a canvas).
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import QApplication


@dataclass(frozen=True)
class Palette:
    """Color constants for the dark industrial theme."""

    bg_root: str = "#0a0a0c"
    bg_panel: str = "#131316"
    bg_panel_alt: str = "#17181b"
    bg_input: str = "#1c1d21"
    border: str = "#2a2b30"
    border_light: str = "#38393f"

    text_primary: str = "#e8e8ea"
    text_secondary: str = "#9a9ba3"
    text_muted: str = "#5f6169"

    accent: str = "#e0293e"
    accent_hover: str = "#f0384d"
    accent_pressed: str = "#c11f32"

    success: str = "#2ecc71"
    warning: str = "#e8a83c"
    error: str = "#e0293e"
    info_cyan: str = "#3ac6d8"

    badge_recommended_bg: str = "#173a25"
    badge_recommended_fg: str = "#3ecf74"
    badge_compatible_bg: str = "#122a33"
    badge_compatible_fg: str = "#3ac6d8"
    badge_experimental_bg: str = "#3a2f14"
    badge_experimental_fg: str = "#e8a83c"
    badge_not_recommended_bg: str = "#3a1418"
    badge_not_recommended_fg: str = "#e0293e"
    badge_not_installed_bg: str = "transparent"
    badge_not_installed_fg: str = "#8a8b92"
    badge_snapshot_only_bg: str = "#231a3a"
    badge_snapshot_only_fg: str = "#a48ce8"
    badge_reconstruction_only_bg: str = "#122a33"
    badge_reconstruction_only_fg: str = "#3ac6d8"

    # 3D viewport surfaces (spec: one continuous dark visual surface).
    viewport_card_bg: str = "#0d0f12"
    viewport_canvas_bg: str = "#090b0e"
    viewport_border: str = "#292d33"
    viewport_floating_bg: str = "rgba(22, 25, 29, 235)"
    viewport_text: str = "#e7e9ed"
    viewport_text_dim: str = "#9298a3"
    viewport_grid_line: str = "#1b1f25"

    # Right settings sidebar
    settings_sidebar_bg: str = "#090c0f"
    settings_card_bg: str = "#111519"
    settings_card_border: str = "#293039"
    settings_input_bg: str = "#1b1f24"
    settings_input_border: str = "#343a42"
    settings_text: str = "#e4e7eb"
    settings_text_secondary: str = "#aeb4bd"
    settings_slider_track: str = "#2a2f36"
    settings_slider_knob: str = "#9aa1ab"
    settings_slider_knob_border: str = "#454c56"
    settings_badge_bg: str = "#14301f"
    settings_badge_fg: str = "#4ad07d"
    settings_badge_border: str = "#245c3a"

    # Performance footer
    footer_bg: str = "#111519"
    footer_border: str = "#272d34"
    footer_separator: str = "#252b31"
    footer_label: str = "#a6abb3"
    footer_value: str = "#e4e7eb"
    footer_value_positive: str = "#52d66b"
    footer_value_muted: str = "#c5c9cf"
    footer_track_bg: str = "#2a2f36"
    footer_status_green_bg: str = "#173a25"
    footer_status_green_fg: str = "#4ad07d"
    footer_status_green_border: str = "#245c3a"
    footer_status_amber_bg: str = "#3a2f14"
    footer_status_amber_fg: str = "#e0a83c"
    footer_status_amber_border: str = "#5c4a24"
    footer_status_red_bg: str = "#3a1418"
    footer_status_red_fg: str = "#e0293e"
    footer_status_red_border: str = "#5c2027"
    footer_status_neutral_bg: str = "#1c1f24"
    footer_status_neutral_fg: str = "#9298a3"
    footer_status_neutral_border: str = "#343a42"

    # Source/runtime toolbar group cards
    toolbar_card_bg: str = "#14181c"
    toolbar_card_border: str = "#20262d"
    toolbar_hover_bg: str = "#1c2126"
    toolbar_active_bg: str = "#2a1417"
    toolbar_text: str = "#c5c9cf"
    toolbar_text_muted: str = "#979da6"
    toolbar_green_bg: str = "#153323"
    toolbar_green_border: str = "#2c5c3f"
    toolbar_green_fg: str = "#6fdb92"
    toolbar_amber_bg: str = "#332913"
    toolbar_amber_border: str = "#5c4a24"
    toolbar_amber_fg: str = "#e0b768"
    toolbar_red_bg: str = "#331519"
    toolbar_red_border: str = "#5c2027"
    toolbar_red_fg: str = "#e88b93"


PALETTE = Palette()

RADIUS_SM = 4
RADIUS_MD = 6
RADIUS_LG = 8

FONT_FAMILY = '"Inter", "Segoe UI", "Ubuntu", sans-serif'
FONT_MONO = '"JetBrains Mono", "Consolas", monospace'

# ---- Live 3D page spacing grid -------------------------------------------
# One shared system so every major card sits on the same rhythm. Applied in
# MainWindow's page layout; individual cards must NOT add their own outer
# margins on top of these.
PAGE_MARGIN_H = 14
PAGE_MARGIN_TOP = 10
PAGE_MARGIN_BOTTOM = 12
PAGE_SPACING = 10
BODY_SPACING = 12
# Splitter handles are what create the visible gaps between the panes they
# separate, so the handle width IS the grid gap.
SPLITTER_HANDLE = 11
FOOTER_HEIGHT = 50


def build_stylesheet(p: Palette = PALETTE) -> str:
    """Build the global QSS stylesheet from the palette.

    Kept as one function rather than per-widget QSS strings so the whole
    visual language stays consistent and auditable in one place.
    """
    return f"""
    * {{
        font-family: {FONT_FAMILY};
        color: {p.text_primary};
        outline: none;
    }}

    QMainWindow, #root {{
        background-color: {p.bg_root};
    }}

    QWidget {{
        background-color: transparent;
    }}

    QFrame#panel {{
        background-color: {p.viewport_card_bg};
        border: 1px solid {p.viewport_border};
        border-radius: {RADIUS_MD}px;
    }}

    QFrame#headerBar {{
        background-color: {p.settings_card_bg};
        border-bottom: 1px solid {p.border};
    }}

    /* ---- Source/runtime toolbar: three independent grouped cards ---- */

    QFrame#toolbarGroupCard {{
        background-color: {p.toolbar_card_bg};
        border: 1px solid {p.toolbar_card_border};
        border-radius: 7px;
    }}

    QLabel#toolbarSectionTitle {{
        color: {p.toolbar_text};
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.4px;
        background: transparent;
    }}

    /* Segmented source-type buttons: near-invisible when inactive, a
       translucent red fill + bottom accent line when active — not a
       full bright rectangular outline. */
    QPushButton#sourceTypeButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 5px 10px;
        color: {p.toolbar_text_muted};
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }}
    QPushButton#sourceTypeButton:hover {{
        background-color: {p.toolbar_hover_bg};
        color: {p.toolbar_text};
    }}
    QPushButton#sourceTypeButton:checked {{
        background-color: {p.toolbar_active_bg};
        border: 1px solid transparent;
        border-bottom: 2px solid {p.accent};
        color: {p.accent};
        font-weight: 700;
    }}

    /* Compact combo boxes reused from the settings sidebar
       (#settingsCombo) automatically pick up their own styling — no
       toolbar-specific override needed, keeping one shared look. */

    /* Runtime action buttons: translucent tinted background + thin
       border, not a solid opaque fill. */
    QPushButton#toolbarDisconnectButton {{
        background-color: {p.toolbar_hover_bg};
        border: 1px solid {p.toolbar_card_border};
        border-radius: 5px;
        padding: 6px 14px;
        color: {p.toolbar_text};
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton#toolbarDisconnectButton:hover {{
        background-color: {p.bg_input};
    }}
    QPushButton#toolbarDisconnectButton:pressed {{
        background-color: {p.bg_root};
    }}
    QPushButton#toolbarDisconnectButton:disabled {{
        color: {p.toolbar_text_muted};
        background-color: transparent;
    }}

    QPushButton#toolbarStartButton {{
        background-color: {p.toolbar_green_bg};
        border: 1px solid {p.toolbar_green_border};
        border-radius: 5px;
        padding: 6px 14px;
        color: {p.toolbar_green_fg};
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton#toolbarStartButton:hover {{ background-color: #1c3d27; }}
    QPushButton#toolbarStartButton:disabled {{
        background-color: transparent;
        border-color: {p.toolbar_card_border};
        color: {p.toolbar_text_muted};
    }}

    QPushButton#toolbarPauseButton {{
        background-color: {p.toolbar_amber_bg};
        border: 1px solid {p.toolbar_amber_border};
        border-radius: 5px;
        padding: 6px 14px;
        color: {p.toolbar_amber_fg};
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton#toolbarPauseButton:hover {{ background-color: #3d3218; }}
    QPushButton#toolbarPauseButton:disabled {{
        background-color: transparent;
        border-color: {p.toolbar_card_border};
        color: {p.toolbar_text_muted};
    }}

    QPushButton#toolbarStopButton {{
        background-color: {p.toolbar_red_bg};
        border: 1px solid {p.toolbar_red_border};
        border-radius: 5px;
        padding: 6px 14px;
        color: {p.toolbar_red_fg};
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton#toolbarStopButton:hover {{ background-color: #3d1a1f; }}
    QPushButton#toolbarStopButton:disabled {{
        background-color: transparent;
        border-color: {p.toolbar_card_border};
        color: {p.toolbar_text_muted};
    }}

    QLabel#sourceConnectionBadge {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.3px;
        border-radius: 5px;
        padding: 3px 10px;
    }}

    QLabel {{
        color: {p.text_primary};
        background: transparent;
    }}

    QLabel#wordmark {{
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 1px;
    }}

    QLabel#subtitle {{
        font-size: 12px;
        font-weight: 600;
        color: {p.text_secondary};
        letter-spacing: 0.5px;
    }}

    QLabel#productName {{
        font-size: 13px;
        font-weight: 700;
        color: {p.text_primary};
        letter-spacing: 0.5px;
    }}

    QLabel#proBadge {{
        background-color: {p.accent};
        border: 1px solid {p.accent_hover};
        color: white;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.5px;
        border-radius: 6px;
    }}

    QLabel#sectionTitle {{
        color: {p.text_secondary};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    QLabel#panelTitle {{
        color: {p.text_primary};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    QLabel#muted {{
        color: {p.text_muted};
        font-size: 11px;
    }}

    QLabel#metricLabel {{
        color: {p.text_muted};
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}

    QLabel#metricValue {{
        font-family: {FONT_MONO};
        color: {p.text_primary};
        font-size: 12px;
        font-weight: 600;
    }}

    QLabel#settingLabel {{
        color: {p.text_secondary};
        font-size: 12px;
    }}

    QLabel#settingValue {{
        font-family: {FONT_MONO};
        color: {p.text_primary};
        font-size: 11px;
    }}

    QPushButton {{
        background-color: {p.bg_input};
        border: 1px solid {p.border_light};
        border-radius: {RADIUS_SM}px;
        padding: 6px 12px;
        color: {p.text_primary};
    }}

    QPushButton:hover {{
        background-color: #232428;
    }}

    QPushButton:pressed {{
        background-color: #0f1012;
    }}

    QPushButton:disabled {{
        color: {p.text_muted};
        background-color: {p.bg_panel_alt};
        border-color: {p.border};
    }}

    QPushButton#accentButton {{
        background-color: {p.accent};
        border: 1px solid {p.accent};
        color: white;
        font-weight: 600;
    }}
    QPushButton#accentButton:hover {{ background-color: {p.accent_hover}; }}
    QPushButton#accentButton:pressed {{ background-color: {p.accent_pressed}; }}
    QPushButton#accentButton:disabled {{
        background-color: #3a1418;
        border-color: #3a1418;
        color: #7a5458;
    }}

    QPushButton#successButton {{
        background-color: #1c5c33;
        border: 1px solid #1c5c33;
        color: white;
        font-weight: 600;
    }}
    QPushButton#successButton:disabled {{
        background-color: #16301f;
        border-color: #16301f;
        color: #5c8f70;
    }}

    QPushButton#warningButton {{
        background-color: #7a5a17;
        border: 1px solid #7a5a17;
        color: white;
        font-weight: 600;
    }}
    QPushButton#warningButton:disabled {{
        background-color: #3d2e14;
        border-color: #3d2e14;
        color: #9c8358;
    }}

    QPushButton#stopButton {{
        background-color: {p.accent};
        border: 1px solid {p.accent};
        color: white;
        font-weight: 700;
        font-size: 13px;
        padding: 8px 22px;
    }}
    QPushButton#stopButton:hover {{ background-color: {p.accent_hover}; }}
    QPushButton#stopButton:disabled {{
        background-color: #2a1417;
        border-color: #2a1417;
        color: #6b4548;
    }}

    QPushButton#sourceTypeButton {{
        background-color: transparent;
        border: 1px solid {p.viewport_border};
        border-radius: {RADIUS_SM}px;
        padding: 6px 14px;
    }}
    QPushButton#sourceTypeButton:hover {{
        background-color: {p.bg_input};
    }}
    QPushButton#sourceTypeButton:checked {{
        background-color: #2a1417;
        border: 1px solid {p.accent};
        color: {p.accent};
        font-weight: 600;
    }}

    QPushButton#iconButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: {RADIUS_SM}px;
        padding: 4px;
    }}
    QPushButton#iconButton:hover {{
        background-color: {p.bg_input};
    }}

    QTabBar::tab {{
        background: transparent;
        color: {p.text_secondary};
        padding: 10px 18px;
        border-bottom: 2px solid transparent;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.3px;
    }}
    QTabBar::tab:selected {{
        color: {p.accent};
        border-bottom: 2px solid {p.accent};
    }}
    QTabBar::tab:disabled {{
        color: {p.text_muted};
    }}
    QTabWidget::pane {{
        border: none;
        border-top: 1px solid {p.border};
    }}

    QComboBox {{
        background-color: {p.bg_input};
        border: 1px solid {p.border_light};
        border-radius: {RADIUS_SM}px;
        padding: 5px 10px;
        min-height: 20px;
    }}
    QComboBox:disabled {{
        color: {p.text_muted};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.bg_panel_alt};
        border: 1px solid {p.border_light};
        selection-background-color: #2a1417;
        selection-color: {p.accent};
        outline: none;
    }}

    QLineEdit {{
        background-color: {p.bg_input};
        border: 1px solid {p.border_light};
        border-radius: {RADIUS_SM}px;
        padding: 5px 8px;
    }}
    QLineEdit:focus {{
        border-color: {p.accent};
    }}
    QLineEdit:disabled {{
        color: {p.text_muted};
    }}

    QSlider::groove:horizontal {{
        height: 4px;
        background: {p.bg_input};
        border: 1px solid {p.border};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {p.accent};
        width: 13px;
        height: 13px;
        margin: -6px 0;
        border-radius: 6px;
    }}
    QSlider::sub-page:horizontal {{
        background: {p.accent};
        border: 1px solid {p.accent};
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal:disabled {{
        background: {p.border};
        border-color: {p.border};
    }}
    QSlider::handle:horizontal:disabled {{
        background: {p.text_muted};
    }}

    QScrollArea {{
        border: none;
        background: transparent;
    }}

    QSplitter::handle {{
        background-color: {p.bg_root};
    }}
    QSplitter::handle:hover {{
        background-color: {p.accent};
    }}

    QStatusBar {{
        background-color: {p.bg_panel};
        border-top: 1px solid {p.border};
    }}

    QToolTip {{
        background-color: {p.bg_panel_alt};
        color: {p.text_primary};
        border: 1px solid {p.border_light};
        padding: 4px;
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border_light};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {p.text_muted};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border-radius: 3px;
        border: 1px solid {p.border_light};
        background-color: {p.bg_input};
    }}
    QCheckBox::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
    }}

    QRadioButton::indicator {{
        width: 13px;
        height: 13px;
        border-radius: 7px;
        border: 1px solid {p.border_light};
        background-color: {p.bg_input};
    }}
    QRadioButton::indicator:checked {{
        border: 1px solid {p.accent};
        background-color: qradialgradient(
            cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5,
            stop: 0 {p.accent}, stop: 0.45 {p.accent},
            stop: 0.55 {p.bg_input}, stop: 1 {p.bg_input}
        );
    }}

    QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: {RADIUS_SM}px;
        padding: 6px;
    }}
    QToolButton:hover {{
        background-color: {p.bg_input};
        border-color: {p.border_light};
    }}
    QToolButton:checked {{
        background-color: #2a1417;
        border-color: {p.accent};
        color: {p.accent};
    }}

    /* ---- 3D viewport: layered overlay composition ---- */

    QFrame#viewportCard {{
        background-color: {p.viewport_card_bg};
        border: 1px solid {p.viewport_border};
        border-radius: {RADIUS_MD}px;
    }}

    QFrame#viewportCanvas {{
        background-color: {p.viewport_canvas_bg};
        border: none;
        border-radius: {RADIUS_SM}px;
    }}

    QLabel#idleMessage {{
        color: {p.viewport_text_dim};
        font-size: 12px;
        background: transparent;
    }}

    /* Shared display surface for Original Source / Depth Map / 3D canvas,
       so all three read as the same material. Uses the same background as
       the 3D viewport canvas with no inner border. */
    QFrame#mediaSurfaceHost {{
        background-color: {p.viewport_canvas_bg};
        border: none;
        border-radius: {RADIUS_SM}px;
    }}

    QLabel#mediaSurface {{
        background-color: {p.viewport_canvas_bg};
        border: none;
        border-radius: {RADIUS_SM}px;
        color: {p.viewport_text_dim};
        font-size: 12px;
    }}

    QFrame#mediaToolbarCard {{
        background-color: rgba(18, 20, 23, 36);
        border: 1px solid rgba(255, 255, 255, 14);
        border-radius: 8px;
    }}
    QFrame#mediaToolbarCard QPushButton#iconButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 4px;
    }}
    QFrame#mediaToolbarCard QPushButton#iconButton:hover {{
        background-color: rgba(255, 255, 255, 22);
        border-color: rgba(255, 255, 255, 18);
    }}
    QFrame#mediaToolbarCard QPushButton#iconButton:checked {{
        background-color: #2a1417;
        border-color: {p.accent};
    }}
    QFrame#mediaToolbarCard QPushButton#iconButton:disabled {{
        background-color: transparent;
        border-color: transparent;
    }}

    QFrame#depthControlsBar {{
        background-color: transparent;
        border: none;
    }}
    QLabel#depthControlLabel {{
        color: {p.text_muted};
        font-size: 11px;
    }}
    QComboBox#depthControlCombo {{
        background-color: {p.bg_input};
        border: 1px solid {p.border_light};
        border-radius: 5px;
        padding: 4px 9px;
        color: {p.text_primary};
        font-size: 11px;
    }}
    QComboBox#depthControlCombo:hover {{
        border-color: #4a4c53;
        background-color: #22242a;
    }}
    QComboBox#depthControlCombo::drop-down {{
        border: none;
        width: 20px;
    }}

    QLabel#viewportTitle {{
        color: {p.viewport_text};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        background: transparent;
    }}

    QLabel#viewportStatus {{
        color: {p.viewport_text_dim};
        font-size: 11px;
        font-weight: 600;
        background: transparent;
    }}

    QFrame#toolOverlay, QFrame#floatingCard {{
        background-color: {p.viewport_floating_bg};
        border: 1px solid {p.viewport_border};
        border-radius: {RADIUS_LG}px;
    }}

    QFrame#toolOverlay QPushButton#iconButton:checked {{
        background-color: #2a1417;
        border: 1px solid {p.accent};
    }}

    QLabel#cardSectionTitle {{
        color: {p.viewport_text_dim};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.8px;
        background: transparent;
    }}

    QLabel#cardSliderValue {{
        font-family: {FONT_MONO};
        color: {p.viewport_text};
        font-size: 10px;
        background-color: {p.bg_input};
        border: 1px solid {p.viewport_border};
        border-radius: 3px;
        padding: 1px 5px;
    }}

    QFrame#floatingCard QRadioButton {{
        color: {p.viewport_text};
        font-size: 11px;
        background: transparent;
        spacing: 7px;
    }}

    /* ---- Right settings sidebar: independent accordion cards ---- */

    QFrame#settingsSidebar, QWidget#settingsContent {{
        background-color: {p.settings_sidebar_bg};
        border: none;
    }}

    QFrame#settingsCard {{
        background-color: {p.settings_card_bg};
        border: 1px solid {p.settings_card_border};
        border-radius: 7px;
    }}

    QLabel#settingsCardTitle {{
        color: {p.settings_text};
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.6px;
        background: transparent;
    }}

    QPushButton#cardChevron {{
        background: transparent;
        border: none;
        padding: 0px;
    }}
    QPushButton#cardChevron:hover {{
        background-color: {p.settings_input_bg};
        border-radius: 4px;
    }}

    QLabel#settingsRowLabel {{
        color: {p.settings_text_secondary};
        font-size: 12px;
        background: transparent;
    }}

    QLabel#settingsRowLabelUpper {{
        color: {p.settings_text_secondary};
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: transparent;
    }}

    QComboBox#settingsCombo {{
        background-color: {p.settings_input_bg};
        border: 1px solid {p.settings_input_border};
        border-radius: 5px;
        padding: 4px 26px 4px 10px;
        color: {p.settings_text};
        font-size: 12px;
    }}
    QComboBox#settingsCombo:hover {{
        border-color: {p.border_light};
    }}
    QComboBox#settingsCombo::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox#settingsCombo::down-arrow {{
        image: none;
        width: 0px;
        height: 0px;
    }}
    QComboBox#settingsCombo QAbstractItemView {{
        background-color: {p.settings_card_bg};
        border: 1px solid {p.settings_input_border};
        selection-background-color: #2a1417;
        selection-color: {p.accent};
        outline: none;
    }}

    QLabel#valueBox {{
        background-color: {p.settings_input_bg};
        border: 1px solid {p.settings_input_border};
        border-radius: 4px;
        color: {p.settings_text};
        font-family: {FONT_MONO};
        font-size: 10px;
    }}

    QLabel#statusBadge {{
        background-color: {p.settings_badge_bg};
        border: 1px solid {p.settings_badge_border};
        border-radius: 4px;
        color: {p.settings_badge_fg};
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 0.3px;
        padding: 2px 7px;
    }}

    /* Sidebar sliders: red fill, dark neutral track, small grey knob. */
    QSlider#settingsSlider::groove:horizontal {{
        height: 4px;
        background: {p.settings_slider_track};
        border: none;
        border-radius: 2px;
    }}
    QSlider#settingsSlider::sub-page:horizontal {{
        background: {p.accent};
        border: none;
        border-radius: 2px;
    }}
    QSlider#settingsSlider::handle:horizontal {{
        background: {p.settings_slider_knob};
        border: 1px solid {p.settings_slider_knob_border};
        width: 12px;
        height: 12px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider#settingsSlider::handle:horizontal:hover {{
        background: {p.settings_text};
    }}

    /* ---- Interactive benchmark page ---- */

    QWidget#benchmarkPage {{
        background-color: {p.bg_root};
    }}

    QFrame#benchmarkCard {{
        background-color: {p.settings_card_bg};
        border: 1px solid {p.settings_card_border};
        border-radius: 7px;
    }}

    QLabel#benchmarkTitle {{
        color: {p.text_primary};
        font-size: 15px;
        font-weight: 700;
    }}

    QLabel#benchmarkSubtitle, QLabel#benchmarkProgressLabel {{
        color: {p.text_secondary};
        font-size: 11px;
    }}

    QLabel#benchmarkSectionTitle {{
        color: {p.text_primary};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    QLabel#benchmarkOptionLabel {{
        color: {p.settings_text_secondary};
        font-size: 11px;
    }}

    QLabel#benchmarkSafetyMessage {{
        background-color: {p.footer_status_amber_bg};
        border: 1px solid {p.footer_status_amber_border};
        border-radius: 5px;
        color: {p.footer_status_amber_fg};
        padding: 6px 8px;
        font-size: 10px;
    }}

    QLabel#benchmarkStatusBadge {{
        border-radius: 5px;
        padding: 4px 11px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.3px;
        background-color: {p.footer_status_neutral_bg};
        border: 1px solid {p.footer_status_neutral_border};
        color: {p.footer_status_neutral_fg};
    }}
    QLabel#benchmarkStatusBadge[statusKind="ready"] {{
        background-color: {p.footer_status_green_bg};
        border-color: {p.footer_status_green_border};
        color: {p.footer_status_green_fg};
    }}
    QLabel#benchmarkStatusBadge[statusKind="running"] {{
        background-color: {p.badge_compatible_bg};
        border-color: {p.badge_compatible_fg};
        color: {p.badge_compatible_fg};
    }}
    QLabel#benchmarkStatusBadge[statusKind="warning"] {{
        background-color: {p.footer_status_amber_bg};
        border-color: {p.footer_status_amber_border};
        color: {p.footer_status_amber_fg};
    }}
    QLabel#benchmarkStatusBadge[statusKind="error"] {{
        background-color: {p.footer_status_red_bg};
        border-color: {p.footer_status_red_border};
        color: {p.footer_status_red_fg};
    }}

    QListWidget#benchmarkModelList, QTableWidget#benchmarkResultsTable, QTextEdit#benchmarkLog {{
        background-color: {p.bg_input};
        border: 1px solid {p.settings_input_border};
        border-radius: 5px;
        color: {p.text_primary};
        selection-background-color: {p.toolbar_active_bg};
        selection-color: {p.text_primary};
    }}

    QListWidget#benchmarkModelList::item {{
        padding: 4px 5px;
    }}

    QTableWidget#benchmarkResultsTable {{
        gridline-color: {p.footer_separator};
        alternate-background-color: {p.bg_panel_alt};
        font-size: 10px;
    }}

    QHeaderView::section {{
        background-color: {p.bg_panel_alt};
        border: none;
        border-right: 1px solid {p.footer_separator};
        border-bottom: 1px solid {p.footer_separator};
        color: {p.text_secondary};
        padding: 5px 7px;
        font-size: 10px;
        font-weight: 700;
    }}

    QComboBox#benchmarkInput, QSpinBox#benchmarkInput, QLineEdit#benchmarkOutputPath {{
        background-color: {p.settings_input_bg};
        border: 1px solid {p.settings_input_border};
        border-radius: 4px;
        color: {p.settings_text};
        min-height: 27px;
        padding: 0 8px;
    }}

    QProgressBar#benchmarkProgress {{
        background-color: {p.footer_track_bg};
        border: none;
        border-radius: 3px;
        min-height: 6px;
        max-height: 6px;
    }}
    QProgressBar#benchmarkProgress::chunk {{
        background-color: {p.accent};
        border-radius: 3px;
    }}

    QPushButton#benchmarkPrimaryButton, QPushButton#benchmarkSecondaryButton,
    QPushButton#benchmarkDangerButton {{
        border-radius: 5px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 650;
    }}
    QPushButton#benchmarkPrimaryButton {{
        background-color: {p.toolbar_green_bg};
        border: 1px solid {p.toolbar_green_border};
        color: {p.toolbar_green_fg};
    }}
    QPushButton#benchmarkDangerButton {{
        background-color: {p.toolbar_red_bg};
        border: 1px solid {p.toolbar_red_border};
        color: {p.toolbar_red_fg};
    }}
    QPushButton#benchmarkSecondaryButton {{
        background-color: {p.toolbar_hover_bg};
        border: 1px solid {p.toolbar_card_border};
        color: {p.toolbar_text};
    }}
    QPushButton#benchmarkPrimaryButton:disabled, QPushButton#benchmarkSecondaryButton:disabled,
    QPushButton#benchmarkDangerButton:disabled {{
        background-color: transparent;
        border-color: {p.toolbar_card_border};
        color: {p.text_muted};
    }}

    QPushButton#benchmarkLinkButton {{
        background: transparent;
        border: none;
        color: {p.info_cyan};
        padding: 2px 4px;
        font-size: 10px;
    }}
    QPushButton#benchmarkLinkButton:hover {{
        color: {p.text_primary};
    }}


    /* ---- Performance footer ---- */

    QFrame#performanceFooter {{
        background-color: {p.footer_bg};
        border: 1px solid {p.footer_border};
        border-radius: 7px;
    }}

    QFrame#footerSeparator {{
        background-color: {p.footer_separator};
        border: none;
    }}

    QLabel#footerCellLabel {{
        color: {p.footer_label};
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.4px;
        background: transparent;
    }}

    QLabel#footerCellValue {{
        color: {p.footer_value};
        font-size: 13px;
        font-weight: 500;
        background: transparent;
    }}

    QLabel#footerCellValuePositive {{
        color: {p.footer_value_positive};
        font-size: 13px;
        font-weight: 600;
        background: transparent;
    }}

    QLabel#footerCellValueMuted {{
        color: {p.footer_value_muted};
        font-size: 13px;
        font-weight: 500;
        background: transparent;
    }}

    QLabel#footerStatusBadge {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.3px;
        border-radius: 5px;
        padding: 3px 10px;
    }}

    QPushButton#modelInfoButton {{
        background: transparent;
        border: 1px solid {p.footer_separator};
        border-radius: 4px;
        padding: 0px;
    }}
    QPushButton#modelInfoButton:hover {{
        background-color: {p.bg_input};
        border-color: {p.border_light};
    }}
    """


def apply_theme(app: QApplication) -> None:
    """Apply the dark industrial stylesheet to the whole application."""
    app.setStyleSheet(build_stylesheet())


def badge_style(bg: str, fg: str) -> str:
    """Return inline QSS for a small status/compatibility badge label."""
    return (
        f"background-color: {bg}; color: {fg}; border-radius: 5px; "
        f"padding: 2px 10px; font-size: 10px; font-weight: 700;"
    )