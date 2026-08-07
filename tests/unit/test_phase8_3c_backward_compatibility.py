from __future__ import annotations

from pathlib import Path

from core.version import RELEASE_NAME, __version__


def test_phase8_3c_keeps_release_identity_and_reconstruction_disabled() -> None:
    assert __version__ == "0.7.5"
    assert RELEASE_NAME == "Phase 7.5 Stable"
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    assert '_ENABLED_MODES = {AppMode.LIVE_3D, AppMode.INSPECTION, AppMode.STABLE_VIDEO, AppMode.SNAPSHOT, AppMode.BENCHMARK}' in source


def test_main_reconstruction_package_has_no_direct_pycolmap_import() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import pycolmap")
            assert not stripped.startswith("from pycolmap")


def test_external_worker_remains_dependency_isolated() -> None:
    source = Path("scripts/reconstruction_pycolmap_worker.py").read_text(encoding="utf-8")
    import_lines = tuple(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    assert "import pycolmap" in import_lines
    forbidden_prefixes = (
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
    assert not any(
        line.startswith(forbidden_prefixes)
        for line in import_lines
    )
