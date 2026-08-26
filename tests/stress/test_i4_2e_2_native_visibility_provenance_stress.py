from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import pytest


def _worker():
    path = Path(__file__).resolve().parents[2] / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
    spec = importlib.util.spec_from_file_location("i4_2e_2_visibility_worker_stress", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_vis(path: Path, rows: list[list[int]], *, declared_points: int | None = None, trailing: bytes = b"") -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(rows) if declared_points is None else declared_points))
        for row in rows:
            handle.write(struct.pack("<I", len(row)))
            if row:
                handle.write(struct.pack(f"<{len(row)}I", *row))
        handle.write(trailing)


def test_i42e2_visibility_rejects_point_count_mismatch(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[0], [1]], declared_points=3)
    with pytest.raises(RuntimeError, match="point count does not match"):
        worker._read_points_visibility(path, expected_points=2, expected_views=2)


def test_i42e2_visibility_rejects_empty_support(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[], [0, 1]])
    with pytest.raises(RuntimeError, match="empty support"):
        worker._read_points_visibility(path, expected_points=2, expected_views=2)


def test_i42e2_visibility_rejects_duplicate_image_index(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[0, 0], [1]])
    with pytest.raises(RuntimeError, match="duplicate image indices"):
        worker._read_points_visibility(path, expected_points=2, expected_views=2)


def test_i42e2_visibility_rejects_out_of_range_image_index(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[0], [2]])
    with pytest.raises(RuntimeError, match="out-of-range"):
        worker._read_points_visibility(path, expected_points=2, expected_views=2)


def test_i42e2_visibility_rejects_truncated_payload(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    path.write_bytes(struct.pack("<QII", 1, 2, 0))  # claims 2 indices, contains only 1
    with pytest.raises(RuntimeError, match="truncated"):
        worker._read_points_visibility(path, expected_points=1, expected_views=2)


def test_i42e2_visibility_rejects_trailing_bytes(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[0], [1]], trailing=b"x")
    with pytest.raises(RuntimeError, match="trailing bytes"):
        worker._read_points_visibility(path, expected_points=2, expected_views=2)


def test_i42e2_visibility_accepts_valid_subset_of_campaign_view_indices(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[0], [1], [1]])
    offsets, indices = worker._read_points_visibility(path, expected_points=3, expected_views=3)
    assert offsets.tolist() == [0, 1, 2, 3]
    assert indices.tolist() == [0, 1, 1]
