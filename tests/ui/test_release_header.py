"""Visible release identity stays synchronized with package metadata."""

from __future__ import annotations

from app.widgets.header_widget import APP_VERSION, HeaderWidget
from core.version import __version__


def test_header_displays_central_release_version(qtbot) -> None:
    header = HeaderWidget()
    qtbot.addWidget(header)

    assert APP_VERSION == f"v{__version__}"
    labels = [label.text() for label in header.findChildren(type(header._gpu_value))]
    assert APP_VERSION in labels
