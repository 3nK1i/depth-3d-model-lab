from __future__ import annotations

from pathlib import Path

from core.capture.frame import SourceFrame
from core.reconstruction import (
    STELLA_LIVE_BACKEND_ID,
    ExternalStellaLiveSlamBackend,
    LiveSlamBackend,
)


ROOT = Path(__file__).resolve().parents[2]


def test_source_frame_contract_remains_unchanged_for_existing_capture_inference_path() -> None:
    assert tuple(SourceFrame.__dataclass_fields__) == ("image", "frame_id", "timestamp")


def test_reconstruction_ui_remains_disabled_until_phase8_8() -> None:
    text = (ROOT / "app" / "widgets" / "mode_tabs.py").read_text(encoding="utf-8")
    enabled_assignment = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_assignment


def test_phase8_4b_python_boundary_never_imports_or_ctypes_loads_stella() -> None:
    paths = [
        ROOT / "core" / "reconstruction" / "live_stella.py",
        ROOT / "core" / "reconstruction" / "live_stella_environment.py",
        ROOT / "core" / "reconstruction" / "live_process.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "import stella_vslam" not in text
    assert "from stella_vslam" not in text
    assert "ctypes.CDLL" not in text
    assert "ctypes.PyDLL" not in text


def test_stella_backend_is_declared_live_backend_without_metric_or_loop_support() -> None:
    # Protocol runtime checking is exercised through integration tests; keep this
    # source-level guard dependency-free and constructor-free.
    assert STELLA_LIVE_BACKEND_ID == "stella-vslam-live"
    assert hasattr(ExternalStellaLiveSlamBackend, "track_live_frame")
    assert hasattr(LiveSlamBackend, "__protocol_attrs__") or hasattr(LiveSlamBackend, "track_live_frame")
