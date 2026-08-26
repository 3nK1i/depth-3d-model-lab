from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_visual_ab import (
    MVS_VISUAL_AB_VIEWS,
    SharedVisualFrame,
    build_visual_ab_summary,
    descriptive_cloud_stats,
    read_colmap_fused_ply,
)


def _frame() -> SharedVisualFrame:
    return SharedVisualFrame(
        lower_reconstruction=(-1.0, -1.0, -1.0),
        upper_reconstruction=(1.0, 1.0, 1.0),
        center_reconstruction=(0.0, 0.0, 0.0),
        center_viewer=(0.0, 0.0, -0.0),
        robust_diagonal=float(np.sqrt(12.0)),
        camera_distance=float(np.sqrt(12.0) * 1.45),
    )


def _write_binary_ply(path: Path) -> None:
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data = np.zeros(2, dtype=dtype)
    data["x"] = [1.0, 4.0]; data["y"] = [2.0, 5.0]; data["z"] = [3.0, 6.0]
    data["red"] = [10, 40]; data["green"] = [20, 50]; data["blue"] = [30, 60]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_bytes(header.encode("ascii") + data.tobytes())


def test_colmap_binary_fused_ply_reader_handles_normals_and_rgb(tmp_path: Path) -> None:
    path = tmp_path / "fused.ply"
    _write_binary_ply(path)
    points, colors = read_colmap_fused_ply(path)
    np.testing.assert_allclose(points, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(colors, [[10, 20, 30], [40, 50, 60]])


def test_colmap_ascii_fused_ply_reader_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "fused_ascii.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        "1 2 3 10 20 30\n4 5 6 40 50 60\n",
        encoding="ascii",
    )
    points, colors = read_colmap_fused_ply(path)
    np.testing.assert_allclose(points, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(colors, [[10, 20, 30], [40, 50, 60]])


def test_descriptive_stats_are_not_quality_gates() -> None:
    points = np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [4, 0, 0]], dtype=np.float64)
    stats = descriptive_cloud_stats(points, _frame())
    assert stats["point_count"] == 4
    assert stats["quality_gate"] is False
    assert stats["inside_shared_core_anchor_fraction"] == pytest.approx(0.75)
    assert stats["radial_p99_from_shared_center"] > 1.0


def test_fixed_visual_views_are_explicit_and_deterministic() -> None:
    assert MVS_VISUAL_AB_VIEWS == {
        "oblique": (32.0, 24.0),
        "front": (0.0, 0.0),
        "side": (90.0, 0.0),
        "top": (0.0, 89.0),
    }


def test_summary_never_selects_automatic_winner() -> None:
    class Sources:
        shared_frame = _frame()
        da3_i4_dir = Path("/da3")
        da3_diagnostics_npz = Path("/da3.npz")
        mvs_a2_dir = Path("/mvs")
        mvs_fused_ply = Path("/fused.ply")
        source_i4_1_7_dir = Path("/i4")
        frozen_sha256 = {"x": "a" * 64}
    summary = build_visual_ab_summary(
        sources=Sources(),
        da3_stats={"point_count": 1},
        mvs_stats={"point_count": 1},
        rendered_views=["oblique"],
        artifacts={"renders": {}},
        point_budget=100,
    )
    assert summary["status"] == "ready_for_manual_visual_ab_review"
    assert summary["decision"]["automatic_winner_selected"] is False
    assert summary["decision"]["manual_visual_verdict_required"] is True
    assert summary["authority"]["dense_provider_promoted"] is False
    assert summary["authority"]["metric_scale_claimed"] is False


def test_summary_rejects_invalid_view_or_budget() -> None:
    class Sources:
        shared_frame = _frame()
        da3_i4_dir = Path("/da3")
        da3_diagnostics_npz = Path("/da3.npz")
        mvs_a2_dir = Path("/mvs")
        mvs_fused_ply = Path("/fused.ply")
        source_i4_1_7_dir = Path("/i4")
        frozen_sha256 = {}
    with pytest.raises(ReconstructionValidationError):
        build_visual_ab_summary(sources=Sources(), da3_stats={}, mvs_stats={}, rendered_views=["bad"], artifacts={}, point_budget=1)
    with pytest.raises(ReconstructionValidationError):
        build_visual_ab_summary(sources=Sources(), da3_stats={}, mvs_stats={}, rendered_views=["oblique"], artifacts={}, point_budget=0)
