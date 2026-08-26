from __future__ import annotations

from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_i42e2_i4_2d_producer_writes_ply_output_that_emits_native_visibility_sidecar() -> None:
    source = (_root() / "scripts/reconstruction_pycolmap_temple_ring_mvs_worker.py").read_text(encoding="utf-8")
    assert "pycolmap.stereo_fusion(" in source
    assert 'output_type="ply"' in source


def test_i42e2_runner_consumes_existing_fused_ply_vis_and_never_requests_replay() -> None:
    source = (_root() / "scripts/run_i4_2e_temple_ring_detail_qualification.py").read_text(encoding="utf-8")
    assert 'native_visibility = Path(f"{raw_ply}.vis")' in source
    assert '"action": "temple_ring_export_native_fusion_visibility"' in source
    assert '"stereo_fusion_replay": False' in source
    assert '"native_i4_2d_fusion_visibility_consumed_for_support_evidence": True' in source
    assert "raw_ply_sha_before" in source and "raw_ply_sha_after" in source
    assert "native_visibility_sha_before" in source and "native_visibility_sha_after" in source
    assert "support_replay/replay_fused.ply" not in source


def test_i42e2_worker_contract_is_dependency_light_and_fail_closed() -> None:
    source = (_root() / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py").read_text(encoding="utf-8")
    assert "import cv2" not in source
    assert "from core." not in source
    assert "import pycolmap" not in source
    assert 'producer_pycolmap_version != "4.1.1"' in source
    assert "unexpected trailing bytes" in source
    assert "duplicate image indices" in source
    assert "point count does not match fused.ply" in source
    assert '"native_visibility_indices_within_expected_range": True' in source
    assert '"native_visibility_full_view_coverage_required": False' in source
    assert "does not cover the exact expected MVS image-index set" not in source


def test_i42e2_native_visibility_is_not_promoted_to_metric_or_measurement_authority() -> None:
    source = (
        (_root() / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py").read_text(encoding="utf-8")
        + (_root() / "scripts/run_i4_2e_temple_ring_detail_qualification.py").read_text(encoding="utf-8")
    )
    for literal in (
        '"metric_scale_claimed": False',
        '"measurement_authoritative": False',
        '"certified_metrology": False',
        '"surface_mesh_generated": False',
        '"i4_3_provider_architecture_authorized": False',
    ):
        assert literal in source
