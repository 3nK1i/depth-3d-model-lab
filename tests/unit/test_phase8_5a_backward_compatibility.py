from __future__ import annotations

import inspect
from pathlib import Path

from core.reconstruction import LiveSlamRequest
from tests.reconstruction_live_helpers import make_calibrated_live_camera
from core.reconstruction import ReconstructionQualityProfile, make_usb_source_metadata
from tests.reconstruction_helpers import SESSION_ID


def test_phase8_5a_default_live_request_still_disables_global_features() -> None:
    request = LiveSlamRequest(
        session_id=SESSION_ID,
        source=make_usb_source_metadata(0, width=320, height=240, frame_rate_hz=30.0),
        camera=make_calibrated_live_camera(),
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="development-live-slam-mock",
    )
    assert request.enable_loop_closure is False
    assert request.enable_global_optimization is False


def test_reconstruction_ui_remains_disabled() -> None:
    text = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_block


def test_phase8_5a_core_does_not_import_native_stella_or_pycolmap() -> None:
    root = Path("core/reconstruction")
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in source
        assert "from pycolmap" not in source
        assert "import stella_vslam" not in source
        assert "from stella_vslam" not in source
