from __future__ import annotations

from pathlib import Path

from core.contracts import CameraIntrinsics, FrameReference, RigidTransform3D
from core.hardware.profiles import ComputeProfile, resolve_profile
from core.reconstruction import RECONSTRUCTION_SCHEMA_VERSION, ReconstructionSession
from core.runtime.session import SessionTracker
from core.version import RELEASE_NAME, __version__


def test_phase81_preserves_v075_release_identity_and_existing_imports() -> None:
    assert __version__ == "0.7.5"
    assert RELEASE_NAME == "Phase 7.5 Stable"
    assert CameraIntrinsics is not None
    assert FrameReference is not None
    assert RigidTransform3D is not None
    assert ComputeProfile is not None
    assert resolve_profile is not None
    assert SessionTracker is not None
    assert ReconstructionSession is not None
    assert RECONSTRUCTION_SCHEMA_VERSION == 1


def test_phase81_keeps_reconstruction_mode_disabled() -> None:
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_line = next(line for line in source.splitlines() if line.startswith("_ENABLED_MODES"))
    assert "AppMode.RECONSTRUCTION" not in enabled_line


def test_phase81_foundation_has_no_heavy_backend_imports() -> None:
    root = Path("core/reconstruction")
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    forbidden = ("import open3d", "import pycolmap", "import torch", "import PyQt6")
    assert not any(token in source for token in forbidden)
