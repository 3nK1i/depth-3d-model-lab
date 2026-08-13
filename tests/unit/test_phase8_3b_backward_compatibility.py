from __future__ import annotations

import importlib.util
from pathlib import Path

from core.reconstruction import ReconstructionQualityProfile
from core.version import RELEASE_NAME, __version__


def test_phase8_3b_keeps_release_identity_and_reconstruction_disabled() -> None:
    assert __version__ == "0.8.0"
    assert RELEASE_NAME == "Phase 8 Reconstruction Stable"
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    assert '_ENABLED_MODES = {AppMode.LIVE_3D, AppMode.INSPECTION, AppMode.STABLE_VIDEO, AppMode.SNAPSHOT, AppMode.BENCHMARK}' in source


def test_core_reconstruction_import_does_not_require_pycolmap() -> None:
    import core.reconstruction as reconstruction

    assert reconstruction.ReconstructionQualityProfile is ReconstructionQualityProfile
    # The main environment is allowed to have no PyCOLMAP at all; Phase 8.3B
    # only discovers it through an external interpreter.
    if importlib.util.find_spec("pycolmap") is None:
        assert reconstruction.PYCOLMAP_RECOMMENDED_VERSION == "4.1.1"


def test_pycolmap_remains_outside_main_application_dependencies() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "pycolmap" not in pyproject.casefold()
    worker_requirements = Path("requirements/reconstruction-pycolmap-cuda12.txt").read_text(
        encoding="utf-8"
    )
    assert "pycolmap-cuda12==4.1.1" in worker_requirements
