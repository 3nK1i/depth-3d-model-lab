from __future__ import annotations

from pathlib import Path


def test_a3_is_visual_only_and_contains_no_reconstruction_execution() -> None:
    root = Path(__file__).resolve().parents[2]
    core = (root / "core/reconstruction/mvs_visual_ab.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_i4_1_9a_a3_visual_ab.py").read_text(encoding="utf-8")
    combined = core + runner
    forbidden = (
        "incremental_mapping(",
        "bundle_adjustment(",
        "undistort_images(",
        "patch_match_stereo(",
        "stereo_fusion(",
        "triangulate_points(",
        "poisson_meshing(",
        "delaunay_meshing(",
    )
    assert all(token not in combined for token in forbidden)
    assert "read_colmap_fused_ply" in runner
    assert "load_da3_cloud" in runner


def test_a3_applies_one_shared_camera_to_both_methods_and_forbids_independent_fit() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/run_i4_1_9a_a3_visual_ab.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/mvs_visual_ab.py").read_text(encoding="utf-8")
    assert runner.count("_apply_shared_camera(") >= 2
    assert "canvas.fit_scene()" not in runner
    assert '"independent_auto_fit_forbidden": True' in core
    assert '"shared_camera_parameters": True' in core


def test_a3_preserves_manual_decision_and_authority_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    core = (root / "core/reconstruction/mvs_visual_ab.py").read_text(encoding="utf-8")
    assert '"automatic_winner_selected": False' in core
    assert '"manual_visual_verdict_required": True' in core
    assert '"metric_scale_claimed": False' in core
    assert '"measurement_authoritative": False' in core
    assert '"surface_mesh_generated": False' in core
    assert '"dense_provider_promoted": False' in core
