from __future__ import annotations

from pathlib import Path

from core.reconstruction import STELLA_LIVE_BACKEND_ID
from tests.reconstruction_live_helpers import make_live_request


def test_default_live_slam_features_remain_disabled() -> None:
    request = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    assert request.enable_loop_closure is False
    assert request.enable_global_optimization is False


def test_raw_live_pose_contract_remains_reconstruction_frame_and_arbitrary_scale() -> None:
    source = Path("core/reconstruction/live_contracts.py").read_text(encoding="utf-8")
    assert 'LIVE_SLAM_RECONSTRUCTION_FRAME = "arbitrary_gauge_live_reconstruction"' in source
    assert "T_reconstruction_from_camera" in source


def test_reconstruction_ui_remains_disabled() -> None:
    text = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_block


def test_native_dependencies_remain_external_process_only() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in source
        assert "from pycolmap" not in source
        assert "import stella_vslam" not in source
        assert "from stella_vslam" not in source
        assert "ctypes.CDLL" not in source
        assert "ctypes.PyDLL" not in source
