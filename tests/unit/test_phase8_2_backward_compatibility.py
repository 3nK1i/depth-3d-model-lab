from __future__ import annotations

from pathlib import Path

from core.capture.frame import SourceFrame
from core.reconstruction import (
    ImageSequenceReader,
    ReconstructionIngestionPipeline,
    ReconstructionSession,
)
from core.version import RELEASE_NAME, __version__


def test_phase82_preserves_release_and_existing_capture_contracts() -> None:
    assert __version__ == "0.7.5"
    assert RELEASE_NAME == "Phase 7.5 Stable"
    assert SourceFrame is not None
    assert ReconstructionSession is not None
    assert ImageSequenceReader is not None
    assert ReconstructionIngestionPipeline is not None


def test_phase82_keeps_reconstruction_mode_disabled() -> None:
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_line = next(line for line in source.splitlines() if line.startswith("_ENABLED_MODES"))
    assert "AppMode.RECONSTRUCTION" not in enabled_line


def test_phase82_does_not_modify_stable_capture_sources() -> None:
    files = (
        Path("core/capture/base_source.py"),
        Path("core/capture/image_source.py"),
        Path("core/capture/video_source.py"),
        Path("core/capture/camera_source.py"),
        Path("core/capture/rtsp_source.py"),
    )
    assert all(path.exists() for path in files)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "core.reconstruction" not in combined


def test_phase82_ingestion_has_no_heavy_backend_or_qt_imports() -> None:
    root = Path("core/reconstruction")
    phase82 = (
        "cache.py",
        "ingestion.py",
        "keyframes.py",
        "quality.py",
        "sampling.py",
        "sources.py",
    )
    source = "\n".join((root / name).read_text(encoding="utf-8") for name in phase82)
    forbidden = ("import PyQt6", "import torch", "import open3d", "import pycolmap")
    assert not any(token in source for token in forbidden)
