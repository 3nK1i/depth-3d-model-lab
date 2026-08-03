"""Dependency-free PLY export round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry.export import export_ply, read_ply_for_verification


def test_binary_ply_round_trip(tmp_path) -> None:
    points = np.array([[1.25, -2.5, 3.0], [4.0, 5.5, 6.75]], dtype=np.float64)
    colors = np.array([[255, 0, 10], [20, 30, 40]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"

    export_ply(path, points, colors)
    restored_points, restored_colors = read_ply_for_verification(path)

    np.testing.assert_allclose(restored_points, points.astype(np.float32))
    np.testing.assert_array_equal(restored_colors, colors)
    assert path.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")


def test_empty_cloud_exports_and_reads_back(tmp_path) -> None:
    path = tmp_path / "empty.ply"
    export_ply(path, np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8))
    points, colors = read_ply_for_verification(path)
    assert points.shape == (0, 3)
    assert colors.shape == (0, 3)


@pytest.mark.parametrize(
    ("points", "colors"),
    [
        (np.zeros((2, 2), np.float32), np.zeros((2, 2), np.uint8)),
        (np.zeros((2, 3), np.float32), np.zeros((1, 3), np.uint8)),
    ],
)
def test_invalid_export_shapes_are_rejected(tmp_path, points, colors) -> None:
    with pytest.raises(ValueError):
        export_ply(tmp_path / "bad.ply", points, colors)
