from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np

from core.models.registry import MODEL_REGISTRY
from core.models.result import GeometryResult
from core.runtime.messages import InferenceResult
from core.version import __version__


def test_depth_registry_order_and_version_are_unchanged():
    assert MODEL_REGISTRY.ids() == (
        "moge-2",
        "depth-pro",
        "unidepth-v2",
        "metric3d-v2",
        "depth-anything-v2",
        "video-depth-anything",
        "marigold",
        "development-mock",
    )
    assert __version__ == "0.7.0"


def test_geometry_and_ipc_positional_compatibility_are_unchanged():
    geometry = GeometryResult(1, 2, 3.0, np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
    assert geometry.session_id == 1 and geometry.frame_id == 2
    ipc = InferenceResult(1, 2, "mock", np.ones((2, 2)), 4.0)
    assert ipc.model_id == "mock" and ipc.latency_ms == 4.0
    assert list(inspect.signature(GeometryResult).parameters)[:5] == [
        "session_id", "frame_id", "timestamp", "rgb", "depth"
    ]
    assert list(inspect.signature(InferenceResult).parameters)[:5] == [
        "session_id", "frame_id", "model_id", "depth", "latency_ms"
    ]


def test_alignment_import_does_not_import_ultralytics_torch_or_qt():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import core.alignment; "
            "blocked=('ultralytics','torch','PyQt6','vispy'); "
            "assert not any(name in sys.modules for name in blocked)"
        ),
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])


def test_existing_runtime_and_ui_sources_do_not_reference_alignment_package():
    root = Path(__file__).resolve().parents[2]
    production_roots = [root / "app", root / "core" / "inference", root / "core" / "runtime"]
    for directory in production_roots:
        for path in directory.rglob("*.py"):
            assert "core.alignment" not in path.read_text(encoding="utf-8")
