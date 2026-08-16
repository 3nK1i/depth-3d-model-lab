from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.capture.frame import SourceFrame
from core.reconstruction import (
    ExternalPycolmapSfmBackend,
    LIVE_SLAM_SCHEMA_VERSION,
    ReconstructionCamera,
    ReconstructionSession,
)
from core.version import RELEASE_NAME, __version__


def test_phase8_4a_preserves_release_identity_and_disabled_reconstruction_ui() -> None:
    assert __version__ == "0.9.0"
    assert RELEASE_NAME == "Phase 9 Digital Twin Stable"
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_line = next(line for line in source.splitlines() if line.startswith("_ENABLED_MODES"))
    assert "AppMode.RECONSTRUCTION" not in enabled_line


def test_phase8_4a_does_not_change_stable_sourceframe_contract() -> None:
    assert tuple(field.name for field in fields(SourceFrame)) == (
        "image",
        "frame_id",
        "timestamp",
    )


def test_phase8_4a_keeps_offline_pycolmap_boundary_and_capabilities() -> None:
    backend = ExternalPycolmapSfmBackend("/tmp/external-python")
    assert backend.capabilities.supports_metric_scale is False
    assert backend.capabilities.supports_resume is True
    for path in Path("core/reconstruction").glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import pycolmap")
            assert not stripped.startswith("from pycolmap")


def test_phase8_4a_foundation_does_not_link_native_stella_inside_python_core() -> None:
    root = Path("core/reconstruction")
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    forbidden = (
        "import stella_vslam",
        "from stella_vslam",
        "ctypes.CDLL",
        "import PyQt6",
    )
    assert not any(token in source for token in forbidden)


def test_phase8_4a_is_additive_to_existing_reconstruction_foundation() -> None:
    assert ReconstructionSession is not None
    assert ReconstructionCamera is not None
    assert LIVE_SLAM_SCHEMA_VERSION == 1
