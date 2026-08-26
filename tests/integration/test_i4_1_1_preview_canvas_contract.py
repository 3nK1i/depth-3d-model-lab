"""Static producer/consumer contract for the real I4.1.1 preview canvas.

This test intentionally does not instantiate Qt/VisPy.  It prevents the preview
from drifting onto PointCloudCanvas while calling ReconstructionCanvas-only APIs.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANVAS_SOURCE = ROOT / "app/widgets/viewport/vispy_canvas.py"
PREVIEW_SOURCE = ROOT / "scripts/run_i4_1_1_orbit_diagnosis_preview.py"


def _class_methods(source: Path, class_name: str) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Missing class {class_name!r} in {source}")


def _preview_tree() -> ast.Module:
    return ast.parse(PREVIEW_SOURCE.read_text(encoding="utf-8"), filename=str(PREVIEW_SOURCE))


def test_preview_uses_reconstruction_canvas_not_point_cloud_canvas() -> None:
    tree = _preview_tree()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "app.widgets.viewport.vispy_canvas":
            imported.update(alias.name for alias in node.names)
    assert "ReconstructionCanvas" in imported
    assert "PointCloudCanvas" not in imported


def test_reconstruction_canvas_exposes_every_preview_api() -> None:
    methods = _class_methods(CANVAS_SOURCE, "ReconstructionCanvas")
    required = {
        "set_dense_rgb_points",
        "update_sparse_landmarks",
        "set_sparse_landmarks_visible",
        "set_canonical_camera_trajectory",
        "set_canonical_camera_frustums_visible",
        "set_canonical_camera_trajectory_visible",
        "set_grid_visible",
        "fit_scene",
        "close",
    }
    assert required <= methods


def test_point_cloud_canvas_does_not_satisfy_reconstruction_preview_contract() -> None:
    methods = _class_methods(CANVAS_SOURCE, "PointCloudCanvas")
    assert "update_sparse_landmarks" not in methods
    assert "set_canonical_camera_trajectory" not in methods


def test_preview_loads_dense_geometry_through_canonical_dense_api() -> None:
    source = PREVIEW_SOURCE.read_text(encoding="utf-8")
    assert "canvas.set_dense_rgb_points(" in source
    assert "canvas.update_points(" not in source


def test_preview_keeps_target_and_camera_diagnostic_layers() -> None:
    source = PREVIEW_SOURCE.read_text(encoding="utf-8")
    assert "canvas.update_sparse_landmarks(" in source
    assert "canvas.set_canonical_camera_trajectory(" in source
