from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_visual_ab import (
    SharedVisualFrame,
    descriptive_cloud_stats,
    read_colmap_fused_ply,
    require_clean_output,
)


def test_non_empty_a3_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "a3"
    output.mkdir()
    marker = output / "manual_review.png"
    marker.write_bytes(b"preserve")
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite visual A/B evidence"):
        require_clean_output(output)
    assert marker.read_bytes() == b"preserve"


def test_truncated_binary_ply_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" + b"\x00" * 10
    )
    with pytest.raises(ReconstructionValidationError, match="truncated"):
        read_colmap_fused_ply(path)


def test_ply_without_rgb_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "no_rgb.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n" + b"\x00" * 12
    )
    with pytest.raises(ReconstructionValidationError, match="missing required properties"):
        read_colmap_fused_ply(path)


def test_nonfinite_ply_geometry_fails_closed(tmp_path: Path) -> None:
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data = np.zeros(1, dtype=dtype); data["x"] = np.nan
    path = tmp_path / "nan.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_bytes(header.encode("ascii") + data.tobytes())
    with pytest.raises(ReconstructionValidationError, match="non-finite"):
        read_colmap_fused_ply(path)


def test_descriptive_stats_are_deterministic_for_identical_input() -> None:
    rng = np.random.default_rng(42)
    points = rng.normal(size=(5000, 3))
    frame = SharedVisualFrame(
        lower_reconstruction=(-1.0, -1.0, -1.0), upper_reconstruction=(1.0, 1.0, 1.0),
        center_reconstruction=(0.0, 0.0, 0.0), center_viewer=(0.0, 0.0, -0.0),
        robust_diagonal=float(np.sqrt(12.0)), camera_distance=5.0,
    )
    assert descriptive_cloud_stats(points, frame) == descriptive_cloud_stats(points.copy(), frame)


def test_decision_matrix_is_not_encoded_as_numeric_threshold_tuning() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "core/reconstruction/mvs_visual_ab.py").read_text(encoding="utf-8")
    assert "automatic_winner_selected" in text
    assert "quality_gate\": False" in text
    assert "mvs_clean_da3_smeared" in text
    assert "random threshold" not in text.lower()
