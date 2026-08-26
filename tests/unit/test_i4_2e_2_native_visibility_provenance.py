from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import numpy as np


def _worker():
    path = Path(__file__).resolve().parents[2] / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
    spec = importlib.util.spec_from_file_location("i4_2e_2_visibility_worker_unit", path)
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


def test_i42e2_native_visibility_parser_matches_colmap_binary_layout(tmp_path: Path) -> None:
    worker = _worker()
    path = tmp_path / "fused.ply.vis"
    rows = [[0, 2, 4], [1, 3], [0, 1, 2, 3, 4]]
    _write_vis(path, rows)
    offsets, indices = worker._read_points_visibility(path, expected_points=3, expected_views=5)
    assert offsets.tolist() == [0, 3, 5, 10]
    assert indices.tolist() == [0, 2, 4, 1, 3, 0, 1, 2, 3, 4]


def test_i42e2_native_visibility_indices_are_zero_based_mvs_indices(tmp_path: Path) -> None:
    worker = _worker()
    path = tmp_path / "fused.ply.vis"
    _write_vis(path, [[0], [1], [2], [3]])
    offsets, indices = worker._read_points_visibility(path, expected_points=4, expected_views=4)
    assert offsets[0] == 0
    assert indices.min() == 0
    assert indices.max() == 3
    np.testing.assert_array_equal(np.unique(indices), np.arange(4, dtype=np.int32))


def test_i42e2_visibility_parser_sorts_each_support_set_deterministically(tmp_path: Path) -> None:
    worker = _worker()
    path = tmp_path / "fused.ply.vis"
    _write_vis(path, [[3, 1, 2, 0]])
    _, indices = worker._read_points_visibility(path, expected_points=1, expected_views=4)
    assert indices.tolist() == [0, 1, 2, 3]


def test_i42e2_worker_has_no_dense_reconstruction_api_dependency() -> None:
    source = (Path(__file__).resolve().parents[2] / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py").read_text(encoding="utf-8")
    assert "import pycolmap" not in source
    assert "pycolmap.stereo_fusion(" not in source
    assert "point.track.elements" not in source
    assert "_read_points_visibility(" in source


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for xyz, rgb in zip(points, colors):
            handle.write(struct.pack("<fffBBB", float(xyz[0]), float(xyz[1]), float(xyz[2]), *map(int, rgb)))


def test_i42e2_worker_end_to_end_exports_support_from_existing_ply_vis(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    worker = root / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
    points = np.column_stack((np.arange(47, dtype=np.float32), np.zeros(47, dtype=np.float32), np.ones(47, dtype=np.float32)))
    colors = np.tile(np.asarray([[10, 20, 30]], dtype=np.uint8), (47, 1))
    ply = tmp_path / "fused.ply"
    vis = tmp_path / "fused.ply.vis"
    support = tmp_path / "support.npz"
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    _write_ply(ply, points, colors)
    _write_vis(vis, [[index] for index in range(47)])
    request.write_text(json.dumps({
        "schema_version": 2,
        "action": "temple_ring_export_native_fusion_visibility",
        "raw_fused_ply": str(ply),
        "native_visibility_path": str(vis),
        "support_npz": str(support),
        "expected_raw_ply_bytes": ply.stat().st_size,
        "expected_raw_vertices": 47,
        "expected_views": 47,
        "producer_pycolmap_version": "4.1.1",
        "authority": {"stereo_fusion_replay": False},
    }), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(worker), "--request", str(request), "--result", str(result)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["native_visibility_consumed"] is True
    assert payload["stereo_fusion_replayed_for_support_evidence"] is False
    with np.load(support, allow_pickle=False) as data:
        assert data["points_reconstruction"].shape == (47, 3)
        assert data["visibility_offsets"].tolist() == list(range(48))
        assert data["visibility_image_indices"].tolist() == list(range(47))
