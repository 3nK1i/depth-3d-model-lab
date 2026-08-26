from __future__ import annotations

from pathlib import Path


def test_i42e_preview_uses_same_sample_and_same_camera_for_pre_post() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/run_i4_2e_temple_ring_detail_preview.py").read_text(encoding="utf-8")
    assert '"same_sample_indices_pre_post": True' in source
    assert '"same_camera_pre_post": True' in source
    assert '"independent_auto_fit": False' in source
    assert "post_points = pre_points[post_keep]" in source


def test_i42e_preview_keeps_manual_review_and_authority_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/run_i4_2e_temple_ring_detail_preview.py").read_text(encoding="utf-8")
    assert '"automatic_winner_selected": False' in source
    assert '"manual_visual_review_required": True' in source
    assert '"production_filter_policy_changed": False' in source
    assert '"i4_3_provider_architecture_authorized": False' in source
    assert '"metric_scale_claimed": False' in source
    assert '"surface_mesh_generated": False' in source
