from __future__ import annotations

from pathlib import Path

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_production_auto_gate import (
    ProductionAutoBoundaryStatus,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderSelectionMode,
)


@pytest.fixture
def controller(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    qtbot.addWidget(source_bar)
    qtbot.addWidget(original_view)
    application = ApplicationController(source_bar, original_view)
    value = ReconstructionController(application, parent=application)
    yield value
    value.shutdown()
    application.shutdown()


def _images(root: Path) -> Path:
    path = root / "images"
    path.mkdir()
    for index in range(4):
        (path / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return path


def test_i6_4c_controller_gate_is_deferred_and_fail_closed(controller) -> None:
    boundary = controller.reconstruction_production_auto_gate
    assert boundary.status is ProductionAutoBoundaryStatus.DEFERRED_NOT_QUALIFIED
    assert boundary.offline_shadow_qualified is True
    assert boundary.production_auto_qualified is False
    assert boundary.production_auto_authorized is False
    assert boundary.selectable is False


def test_i6_4c_source_selection_does_not_authorize_auto(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path))
    boundary = controller.reconstruction_production_auto_gate
    assert boundary.production_auto_authorized is False
    assert boundary.selectable is False
    assert boundary.current_selection_mode is ProductionProviderSelectionMode.NONE


def test_i6_4c_manual_provider_api_still_rejects_auto(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path))
    with pytest.raises(ReconstructionValidationError, match="Auto is not a concrete"):
        controller.select_reconstruction_provider("auto")
    assert controller.reconstruction_production_auto_gate.selectable is False


def test_i6_4c_gate_read_has_no_runtime_effect(controller) -> None:
    assert controller.offline_is_running is False
    _ = controller.reconstruction_production_auto_gate
    assert controller.offline_is_running is False
