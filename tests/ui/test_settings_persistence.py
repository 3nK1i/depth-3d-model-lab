"""Typed QSettings wrapper tests using the isolated test configuration."""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QSettings

from app.settings import APP_NAME, ORG_NAME, AppSettings


def _clean() -> None:
    settings = QSettings(ORG_NAME, APP_NAME)
    settings.clear()
    settings.sync()


def test_settings_round_trip_typed_values() -> None:
    _clean()
    settings = AppSettings()
    geometry = QByteArray(b"geometry")
    state = QByteArray(b"state")
    splitter = QByteArray(b"splitter")

    settings.save_window_geometry(geometry)
    settings.save_window_state(state)
    settings.save_splitter_state(AppSettings.KEY_SPLITTER_MAIN, splitter)
    settings.save_last_source_type("rtsp")
    settings.save_last_model_id("moge-2")
    settings.save_compute_profile("efficient")
    settings.save_settings_panel_collapsed(True)
    settings.sync()

    restored = AppSettings()
    assert restored.load_window_geometry() == geometry
    assert restored.load_window_state() == state
    assert restored.load_splitter_state(AppSettings.KEY_SPLITTER_MAIN) == splitter
    assert restored.load_last_source_type() == "rtsp"
    assert restored.load_last_model_id() == "moge-2"
    assert restored.load_compute_profile() == "efficient"
    assert restored.load_settings_panel_collapsed() is True


def test_wrong_binary_value_type_returns_none() -> None:
    _clean()
    raw = QSettings(ORG_NAME, APP_NAME)
    raw.setValue(AppSettings.KEY_WINDOW_GEOMETRY, "not-bytes")
    raw.sync()
    assert AppSettings().load_window_geometry() is None


def test_string_boolean_values_are_parsed() -> None:
    _clean()
    raw = QSettings(ORG_NAME, APP_NAME)
    raw.setValue(AppSettings.KEY_SETTINGS_PANEL_COLLAPSED, "1")
    raw.sync()
    assert AppSettings().load_settings_panel_collapsed() is True
