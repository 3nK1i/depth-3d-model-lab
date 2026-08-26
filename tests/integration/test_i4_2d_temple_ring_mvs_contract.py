from __future__ import annotations

from pathlib import Path


def test_i42d_worker_is_dense_only_and_contains_no_camera_solver_or_gt_loader() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (root / "scripts/reconstruction_pycolmap_temple_ring_mvs_worker.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_i4_2d_temple_ring_mvs.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/temple_ring_mvs.py").read_text(encoding="utf-8")
    combined = worker + runner + core
    assert "pycolmap.undistort_images(" in worker
    assert "pycolmap.patch_match_stereo(" in worker
    assert "pycolmap.stereo_fusion(" in worker
    for forbidden in (
        "incremental_mapping(", "global_mapping(", "bundle_adjustment(", "triangulate_points(",
        "poisson_meshing(", "delaunay_meshing(", "templeR_ang.txt",
    ):
        assert forbidden not in combined


def test_i42d_runs_all_47_references_without_global_view_cap() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (root / "scripts/reconstruction_pycolmap_temple_ring_mvs_worker.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/temple_ring_mvs.py").read_text(encoding="utf-8")
    assert "num_patch_match_src_images=source_neighbors" in worker
    assert "TEMPLE_RING_EXPECTED_VIEWS = 47" in core
    assert "TEMPLE_RING_DEFAULT_SOURCE_NEIGHBORS = 10" in core
    assert '"permanent_global_view_cap": False' in core


def test_i42d_raw_fusion_precedes_manifold_and_keeps_authority_false() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (root / "scripts/reconstruction_pycolmap_temple_ring_mvs_worker.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/temple_ring_mvs.py").read_text(encoding="utf-8")
    assert 'input_type="geometric"' in worker
    assert 'output_type="ply"' in worker
    assert '"production_manifold_applied": False' in core
    assert '"production_filter_policy_changed": False' in core
    assert '"dense_provider_promoted": False' in core
    assert '"i4_3_provider_architecture_authorized": False' in core
    assert '"metric_scale_claimed": False' in core
    assert '"measurement_authoritative": False' in core
    assert '"surface_mesh_generated": False' in core
