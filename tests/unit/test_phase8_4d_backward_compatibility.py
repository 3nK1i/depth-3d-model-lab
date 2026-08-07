from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    LiveCaptureRuntimeState,
    LiveSlamRecoveryPolicy,
    STELLA_LIVE_BACKEND_ID,
)


ROOT = Path(__file__).resolve().parents[2]


def test_phase8_4d_is_additive_and_reconstruction_ui_remains_disabled() -> None:
    mode_tabs = (ROOT / "app" / "widgets" / "mode_tabs.py").read_text(encoding="utf-8")
    assert "AppMode.RECONSTRUCTION" not in mode_tabs.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert LiveCaptureRuntimeState.RECOVERING.value == "recovering"
    assert LiveSlamRecoveryPolicy().max_worker_restarts == 2
    assert STELLA_LIVE_BACKEND_ID == "stella-vslam-live"


def test_phase8_4d_keeps_pycolmap_and_stella_native_linkage_out_of_python_core() -> None:
    reconstruction_root = ROOT / "core" / "reconstruction"
    for path in reconstruction_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in text
        assert "from pycolmap" not in text
        assert "import stella_vslam" not in text
        assert "from stella_vslam" not in text
        assert "ctypes.CDLL" not in text
        assert "ctypes.PyDLL" not in text
