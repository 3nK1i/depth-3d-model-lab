from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import pytest


def _worker():
    path = Path(__file__).resolve().parents[2] / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
    spec = importlib.util.spec_from_file_location("i4_2e_2a_visibility_worker_stress", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_vis(path: Path, rows: list[list[int]]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(rows)))
        for row in rows:
            handle.write(struct.pack("<I", len(row)))
            if row:
                handle.write(struct.pack(f"<{len(row)}I", *row))


def test_i42e2a_single_observed_campaign_view_is_valid_native_visibility(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[4], [4], [4]])
    offsets, indices = worker._read_points_visibility(path, expected_points=3, expected_views=6)
    assert offsets.tolist() == [0, 1, 2, 3]
    assert indices.tolist() == [4, 4, 4]


def test_i42e2a_highest_in_range_index_is_valid_without_full_coverage(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[46], [0, 46]])
    offsets, indices = worker._read_points_visibility(path, expected_points=2, expected_views=47)
    assert offsets.tolist() == [0, 1, 3]
    assert indices.tolist() == [46, 0, 46]


def test_i42e2a_index_equal_to_expected_views_still_fails_closed(tmp_path: Path) -> None:
    worker = _worker(); path = tmp_path / "x.vis"
    _write_vis(path, [[47]])
    with pytest.raises(RuntimeError, match="out-of-range"):
        worker._read_points_visibility(path, expected_points=1, expected_views=47)
