from __future__ import annotations

from pathlib import Path


def _sources() -> tuple[str, str, str]:
    root = Path(__file__).resolve().parents[2]
    worker = (root / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_i4_2e_temple_ring_detail_qualification.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/temple_ring_detail_qualification.py").read_text(encoding="utf-8")
    return worker, runner, core


def test_i42e_worker_consumes_native_visibility_without_replaying_heavy_mvs() -> None:
    worker, runner, _ = _sources()
    assert "_read_points_visibility(" in worker
    assert '"native_visibility_consumed": True' in worker
    assert '"stereo_fusion_replayed_for_support_evidence": False' in worker
    assert '"stereo_fusion_replay": False' in runner
    for forbidden in (
        "pycolmap.stereo_fusion(", "pycolmap.patch_match_stereo(", "incremental_mapping(",
        "bundle_adjustment(", "pycolmap.undistort_images(", "poisson_meshing(", "delaunay_meshing(",
    ):
        assert forbidden not in worker


def test_i42e_exact_d12_predicate_is_reused_without_changing_production_policy() -> None:
    _, runner, core = _sources()
    assert "DenseConfidenceFilterPolicy" in core
    assert "filter_dense_supported_geometry" in core
    assert "DenseManifoldRefinementPolicy" in core
    assert "build_scene_relative_manifold_evidence" in core
    assert '"exact_production_predicate": True' in core
    assert '"production_filter_policy_changed": False' in runner
    assert '"d12_counterfactual_only": True' in runner


def test_i42e_native_visibility_is_required_and_bound_to_raw_fused_point_order() -> None:
    worker, runner, core = _sources()
    assert "fused.ply.vis" in worker
    assert "point_order_bound_to_fused_ply" in worker
    assert "visibility_indices_are_mvs_indices_not_colmap_image_ids" in worker
    assert "native_visibility_path" in runner
    assert "validate_fusion_visibility_evidence" in core
    assert "point.track.elements" not in worker


def test_i42e_does_not_authorize_i43_metric_measurement_or_mesh() -> None:
    worker, runner, core = _sources()
    combined = worker + runner + core
    for literal in (
        '"i4_3_provider_architecture_authorized": False',
        '"metric_scale_claimed": False',
        '"measurement_authoritative": False',
        '"surface_mesh_generated": False',
    ):
        assert literal in combined
