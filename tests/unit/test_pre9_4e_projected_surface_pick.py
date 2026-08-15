from __future__ import annotations

import numpy as np
import pytest

from app.reconstruction_interactive_measurement import pick_projected_surface_point
from app.reconstruction_visualization import (
    reconstruction_point_to_viewer,
    viewer_point_to_reconstruction,
)


def test_projected_triangle_pick_returns_interpolated_surface_point() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    mapped = np.asarray([[10.0, 10.0, 0.2, 1.0], [110.0, 10.0, 0.2, 1.0], [10.0, 110.0, 0.2, 1.0]])
    picked = pick_projected_surface_point(
        vertices_viewer=vertices,
        faces=faces,
        mapped_homogeneous=mapped,
        click_xy=(35.0, 35.0),
    )
    assert picked is not None
    np.testing.assert_allclose(picked.point_viewer, (2.5, 2.5, 0.0), atol=1e-8)
    assert sum(picked.barycentric) == pytest.approx(1.0)


def test_projected_pick_chooses_frontmost_overlapping_triangle() -> None:
    vertices = np.asarray(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 1], [10, 0, 1], [0, 10, 1]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    mapped = np.asarray(
        [
            [0, 0, 0.7, 1], [100, 0, 0.7, 1], [0, 100, 0.7, 1],
            [0, 0, 0.2, 1], [100, 0, 0.2, 1], [0, 100, 0.2, 1],
        ],
        dtype=np.float64,
    )
    picked = pick_projected_surface_point(
        vertices_viewer=vertices,
        faces=faces,
        mapped_homogeneous=mapped,
        click_xy=(20.0, 20.0),
    )
    assert picked is not None
    assert picked.face_index == 1
    assert picked.point_viewer[2] == pytest.approx(1.0)


def test_projected_pick_returns_none_for_background_click() -> None:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    mapped = np.asarray([[0, 0, 0, 1], [10, 0, 0, 1], [0, 10, 0, 1]], dtype=np.float64)
    assert pick_projected_surface_point(
        vertices_viewer=vertices,
        faces=faces,
        mapped_homogeneous=mapped,
        click_xy=(20.0, 20.0),
    ) is None


def test_viewer_reconstruction_transform_round_trip_is_exact() -> None:
    point = np.asarray([1.25, -2.5, 3.75], dtype=np.float64)
    viewer = reconstruction_point_to_viewer(point)
    restored = viewer_point_to_reconstruction(viewer)
    np.testing.assert_allclose(restored, point, atol=1e-12)


def test_real_reconstruction_canvas_declares_measurement_signals_on_the_correct_class() -> None:
    """Guard the real VisPy class, not only the Qt-only test double in conftest."""
    import ast
    from pathlib import Path

    source_path = Path(__file__).resolve().parents[2] / "app" / "widgets" / "viewport" / "vispy_canvas.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }
    reconstruction = classes["ReconstructionCanvas"]
    point_cloud = classes["PointCloudCanvas"]

    def assigned_names(node: ast.ClassDef) -> set[str]:
        names: set[str] = set()
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                names.add(statement.target.id)
        return names

    reconstruction_names = assigned_names(reconstruction)
    point_cloud_names = assigned_names(point_cloud)
    assert "measurement_point_picked" in reconstruction_names
    assert "measurement_cancel_requested" in reconstruction_names
    assert "measurement_point_picked" not in point_cloud_names
    assert "measurement_cancel_requested" not in point_cloud_names
