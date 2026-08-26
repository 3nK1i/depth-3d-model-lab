from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest


def _worker():
    path = Path(__file__).resolve().parents[2] / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
    spec = importlib.util.spec_from_file_location("i4_2e_1_worker_stress_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_i42e1_worker_rejects_truncated_binary_vertex_payload(tmp_path: Path) -> None:
    worker = _worker()
    path = tmp_path / "truncated.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        + struct.pack("<fffBBB", 1.0, 2.0, 3.0, 1, 2, 3)
    )
    with pytest.raises(RuntimeError, match="truncated"):
        worker._read_colmap_fused_ply(path)


def test_i42e1_worker_rejects_missing_rgb_property(tmp_path: Path) -> None:
    worker = _worker()
    path = tmp_path / "missing_rgb.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nend_header\n"
        "1 2 3 4 5\n",
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="missing required properties"):
        worker._read_colmap_fused_ply(path)


def test_i42e1_worker_rejects_nonfinite_ply_geometry(tmp_path: Path) -> None:
    worker = _worker()
    path = tmp_path / "nonfinite.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        "nan 2 3 4 5 6\n",
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        worker._read_colmap_fused_ply(path)


def test_i42e1_worker_digest_canonicalises_signed_zero() -> None:
    worker = _worker()
    colors = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    plus = np.asarray([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]], dtype=np.float64)
    minus = np.asarray([[-0.0, 1.0, 2.0], [3.0, -0.0, 4.0]], dtype=np.float64)
    assert worker._canonical_cloud_digest(plus, colors) == worker._canonical_cloud_digest(minus, colors)
