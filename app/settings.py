"""Persistent, non-sensitive UI settings via QSettings.

Only window geometry, splitter positions and last-used UI choices are
stored here. Never store credentials (e.g. RTSP passwords) through this
module — see the security note in the README.
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QSettings

ORG_NAME = "EIA Tech"
APP_NAME = "Depth3DModelLab"


class AppSettings:
    """Thin, typed wrapper around :class:`QSettings`.

    Keeping all keys in one place avoids typo'd QSettings key strings
    scattered across widgets.
    """

    KEY_WINDOW_GEOMETRY = "window/geometry"
    KEY_WINDOW_STATE = "window/state"
    KEY_SPLITTER_MAIN = "splitters/main_horizontal"
    KEY_SPLITTER_LEFT = "splitters/left_vertical"
    KEY_SPLITTER_TOP = "splitters/top_row"
    KEY_LAST_SOURCE_TYPE = "source/last_type"
    KEY_LAST_MODEL_ID = "model/last_id"
    KEY_SETTINGS_PANEL_COLLAPSED = "ui/settings_panel_collapsed"

    def __init__(self) -> None:
        self._qs = QSettings(ORG_NAME, APP_NAME)

    def save_window_geometry(self, geometry: QByteArray) -> None:
        self._qs.setValue(self.KEY_WINDOW_GEOMETRY, geometry)

    def load_window_geometry(self) -> QByteArray | None:
        value = self._qs.value(self.KEY_WINDOW_GEOMETRY)
        return value if isinstance(value, QByteArray) else None

    def save_window_state(self, state: QByteArray) -> None:
        self._qs.setValue(self.KEY_WINDOW_STATE, state)

    def load_window_state(self) -> QByteArray | None:
        value = self._qs.value(self.KEY_WINDOW_STATE)
        return value if isinstance(value, QByteArray) else None

    def save_splitter_state(self, key: str, state: QByteArray) -> None:
        self._qs.setValue(key, state)

    def load_splitter_state(self, key: str) -> QByteArray | None:
        value = self._qs.value(key)
        return value if isinstance(value, QByteArray) else None

    def save_last_source_type(self, source_type: str) -> None:
        self._qs.setValue(self.KEY_LAST_SOURCE_TYPE, source_type)

    def load_last_source_type(self) -> str | None:
        value = self._qs.value(self.KEY_LAST_SOURCE_TYPE)
        return str(value) if value is not None else None

    def save_last_model_id(self, model_id: str) -> None:
        self._qs.setValue(self.KEY_LAST_MODEL_ID, model_id)

    def load_last_model_id(self) -> str | None:
        value = self._qs.value(self.KEY_LAST_MODEL_ID)
        return str(value) if value is not None else None

    def save_settings_panel_collapsed(self, collapsed: bool) -> None:
        self._qs.setValue(self.KEY_SETTINGS_PANEL_COLLAPSED, collapsed)

    def load_settings_panel_collapsed(self) -> bool:
        value = self._qs.value(self.KEY_SETTINGS_PANEL_COLLAPSED, False)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1")

    def sync(self) -> None:
        self._qs.sync()
