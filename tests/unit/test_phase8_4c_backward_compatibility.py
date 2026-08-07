from __future__ import annotations

from pathlib import Path

from core.capture.frame import SourceFrame


ROOT = Path(__file__).resolve().parents[2]


def test_source_frame_contract_stays_frozen_for_capture_and_inference() -> None:
    assert tuple(SourceFrame.__dataclass_fields__) == ("image", "frame_id", "timestamp")


def test_reconstruction_ui_stays_disabled_in_phase8_4c() -> None:
    text = (ROOT / "app" / "widgets" / "mode_tabs.py").read_text(encoding="utf-8")
    enabled_assignment = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_assignment


def test_main_window_bridge_is_dormant_and_shutdown_order_is_safe() -> None:
    text = (ROOT / "app" / "main_window.py").read_text(encoding="utf-8")
    assert "LiveSlamCaptureController(self._controller, self)" in text
    assert "start_stella_for_active_source" not in text
    assert text.index("self._live_slam_controller.shutdown()") < text.index(
        "self._controller.shutdown()"
    )


def test_phase8_4c_never_links_stella_into_python() -> None:
    paths = [
        ROOT / "core" / "reconstruction" / "live_runtime.py",
        ROOT / "app" / "live_slam_controller.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "import stella_vslam" not in text
    assert "from stella_vslam" not in text
    assert "ctypes.CDLL" not in text
    assert "ctypes.PyDLL" not in text
