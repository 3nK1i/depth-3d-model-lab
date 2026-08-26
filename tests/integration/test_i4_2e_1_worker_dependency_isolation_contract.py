from __future__ import annotations

from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_i42e1_support_worker_has_no_project_or_opencv_import_dependency() -> None:
    source = (_root() / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py").read_text(encoding="utf-8")
    assert "from core." not in source
    assert "import core." not in source
    assert "import cv2" not in source
    assert "from cv2" not in source
    assert "def _read_colmap_fused_ply(" in source
    assert "def _canonical_cloud_digest(" in source


def test_i42e1_support_worker_preserves_isolation_after_native_visibility_correction() -> None:
    source = (_root() / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py").read_text(encoding="utf-8")
    assert '"project_reconstruction_package_required": False' in source
    assert '"opencv_required": False' in source
    assert '"pycolmap_import_required": False' in source
    assert "_read_points_visibility(" in source
    for forbidden in (
        "pycolmap.stereo_fusion(", "pycolmap.patch_match_stereo(", "incremental_mapping(",
        "bundle_adjustment(", "pycolmap.undistort_images(", "poisson_meshing(", "delaunay_meshing(",
    ):
        assert forbidden not in source


def test_i42e1_test_runner_probes_worker_with_isolated_python_and_import_guard() -> None:
    source = (_root() / "scripts/run_i4_2e_tests.sh").read_text(encoding="utf-8")
    assert "EIA_PYCOLMAP_PYTHON" in source
    assert "runpy.run_path" in source
    assert '{"cv2", "pycolmap", "core"}' in source
    assert 'name.startswith("core.")' in source
    assert "I4.2E.2 isolated native-visibility probe: PASS" in source
