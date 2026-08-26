from __future__ import annotations

from pathlib import Path


def test_a2_worker_contains_only_dense_mvs_operations_and_no_camera_solver() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (root / "scripts/reconstruction_pycolmap_mvs_worker.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_i4_1_9a_a2_house_mvs.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/mvs_execution.py").read_text(encoding="utf-8")
    combined = worker + runner + core
    assert "pycolmap.undistort_images(" in worker
    assert "pycolmap.patch_match_stereo(" in worker
    assert "pycolmap.stereo_fusion(" in worker
    assert "incremental_mapping(" not in combined
    assert "global_mapping(" not in combined
    assert "bundle_adjustment(" not in combined
    assert "triangulate_points(" not in combined
    assert "poisson_meshing(" not in combined
    assert "delaunay_meshing(" not in combined


def test_a2_uses_pycolmap_source_neighbor_control_without_global_view_cap() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (root / "scripts/reconstruction_pycolmap_mvs_worker.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/mvs_execution.py").read_text(encoding="utf-8")
    assert "num_patch_match_src_images=source_neighbors" in worker
    assert 'MVS_HOUSE_EXPECTED_REGISTERED_VIEWS = 68' in core
    assert 'MVS_HOUSE_DEFAULT_SOURCE_NEIGHBORS = 10' in core
    assert '"permanent_global_view_cap": False' in core


def test_a2_fusion_is_geometric_and_does_not_claim_metric_or_mesh_authority() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (root / "scripts/reconstruction_pycolmap_mvs_worker.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/mvs_execution.py").read_text(encoding="utf-8")
    assert 'input_type="geometric"' in worker
    assert 'output_type="ply"' in worker
    assert '"metric_scale_claimed": False' in core
    assert '"measurement_authoritative": False' in core
    assert '"surface_mesh_generated": False' in core
