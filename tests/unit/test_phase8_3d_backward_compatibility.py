from __future__ import annotations

from pathlib import Path

from core.reconstruction import ExternalPycolmapSfmBackend
from core.version import RELEASE_NAME, __version__


def test_phase8_3d_keeps_release_identity_and_reconstruction_disabled() -> None:
    assert __version__ == "0.7.5"
    assert RELEASE_NAME == "Phase 7.5 Stable"
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    assert (
        "_ENABLED_MODES = {AppMode.LIVE_3D, AppMode.INSPECTION, "
        "AppMode.STABLE_VIDEO, AppMode.SNAPSHOT, AppMode.BENCHMARK}"
    ) in source


def test_phase8_3d_keeps_main_reconstruction_package_pycolmap_free() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import pycolmap")
            assert not stripped.startswith("from pycolmap")


def test_phase8_3d_worker_remains_external_and_dependency_isolated() -> None:
    source = Path("scripts/reconstruction_pycolmap_worker.py").read_text(encoding="utf-8")
    imports = tuple(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    assert "import pycolmap" in imports
    forbidden = (
        "import cv2",
        "from cv2",
        "import torch",
        "from torch",
        "import PyQt",
        "from PyQt",
        "import core",
        "from core",
        "import app",
        "from app",
    )
    assert not any(line.startswith(forbidden) for line in imports)


def test_phase8_3d_backend_declares_safe_stage_resume_without_metric_scale() -> None:
    backend = ExternalPycolmapSfmBackend("/tmp/external-python")
    capabilities = backend.capabilities
    assert capabilities.supports_checkpoint is True
    assert capabilities.supports_resume is True
    assert capabilities.supports_metric_scale is False
    assert capabilities.supports_pause is False
