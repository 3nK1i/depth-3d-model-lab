from __future__ import annotations

import ast
from pathlib import Path

from core.models.registry import MODEL_REGISTRY
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.version import __version__


def test_existing_registries_phase75_version_and_runtime_sources_are_preserved():
    assert __version__ == "0.9.0"
    assert tuple(MODEL_REGISTRY.ids()) == (
        "moge-2", "depth-pro", "unidepth-v2", "metric3d-v2",
        "depth-anything-v2", "video-depth-anything", "marigold", "development-mock",
    )
    assert tuple(PERCEPTION_MODEL_REGISTRY.ids()) == (
        "ultralytics-yolo-seg", "ultralytics-yolo-detect",
        "ultralytics-sam2-prompted", "development-perception-mock",
    )
    for path in (
        "core/inference/worker.py", "core/runtime/messages.py",
        "app/inference_controller.py", "app/main_window.py",
    ):
        ast.parse(Path(path).read_text(encoding="utf-8"))


def test_object_geometry_package_does_not_import_forbidden_runtime_frameworks():
    forbidden = {"torch", "PyQt6", "vispy", "ultralytics"}
    for path in Path("core/object_geometry").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        assert not (imports & forbidden), (path, imports & forbidden)
