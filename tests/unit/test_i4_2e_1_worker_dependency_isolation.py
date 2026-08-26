from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np


def _worker_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"


def _load_worker(*, block_project_and_cv2: bool = False):
    path = _worker_path()
    spec = importlib.util.spec_from_file_location("i4_2e_1_worker_isolation_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if not block_project_and_cv2:
        spec.loader.exec_module(module)
        return module

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2" or name == "core" or name.startswith("core."):
            raise AssertionError(f"isolated I4.2E worker attempted forbidden import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import
    return module


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for xyz, rgb in zip(points, colors):
            handle.write(struct.pack("<fffBBB", float(xyz[0]), float(xyz[1]), float(xyz[2]), *map(int, rgb)))


def test_i42e1_worker_imports_with_project_package_and_cv2_blocked() -> None:
    module = _load_worker(block_project_and_cv2=True)
    assert callable(module._read_colmap_fused_ply)
    assert callable(module._canonical_cloud_digest)


def test_i42e1_worker_local_binary_ply_reader_round_trips_xyz_rgb(tmp_path: Path) -> None:
    module = _load_worker()
    points = np.asarray([[1.25, -2.5, 3.75], [4.0, 5.5, -6.25]], dtype=np.float32)
    colors = np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"
    _write_binary_ply(path, points, colors)
    got_points, got_colors = module._read_colmap_fused_ply(path)
    np.testing.assert_allclose(got_points, points.astype(np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(got_colors, colors)


def test_i42e1_worker_local_digest_is_order_independent_and_sensitive() -> None:
    module = _load_worker()
    points = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
    colors = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    digest = module._canonical_cloud_digest(points, colors)
    assert digest == module._canonical_cloud_digest(points[::-1], colors[::-1])
    changed = points.copy(); changed[0, 0] = 0.25
    assert digest != module._canonical_cloud_digest(changed, colors)
    assert len(digest) == hashlib.sha256().digest_size * 2


def test_i42e1_worker_digest_matches_application_i42e_semantics() -> None:
    module = _load_worker()
    source_path = Path(__file__).resolve().parents[2] / "core/reconstruction/temple_ring_detail_qualification.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {"_xyz", "_rgb", "canonical_cloud_digest"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes} == wanted

    class ReconstructionValidationError(RuntimeError):
        pass

    namespace = {
        "np": np,
        "hashlib": hashlib,
        "ReconstructionValidationError": ReconstructionValidationError,
    }
    module_ast = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    exec(compile(module_ast, str(source_path), "exec"), namespace)
    application_digest = namespace["canonical_cloud_digest"]

    points = np.asarray([[0.0, -0.0, 2.5], [1.25, 4.0, -7.0], [2.0, 2.0, 2.0]], dtype=np.float64)
    colors = np.asarray([[9, 8, 7], [1, 2, 3], [255, 0, 128]], dtype=np.uint8)
    assert module._canonical_cloud_digest(points, colors) == application_digest(points, colors)
