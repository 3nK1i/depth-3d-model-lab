from __future__ import annotations

from pathlib import Path


def test_a31_finalizer_requires_explicit_human_confirmation_and_persists_new_evidence() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_i4_1_9a_a3_1_finalize_manual_verdict.py").read_text(encoding="utf-8")
    assert "--confirm-manual-verdict" in text
    assert "--verdict" in text
    assert "mvs_better_but_imperfect" in text
    assert "i4_1_9a_a3_1_closure.json" in text
    assert "require_absent_output" in text


def test_a31_closure_is_diagnostic_persistence_not_reconstruction_or_provider_promotion() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "core/reconstruction/mvs_visual_ab_closure.py").read_text(encoding="utf-8")
    for forbidden_call in (
        "incremental_mapping(",
        "bundle_adjustment(",
        "undistort_images(",
        "patch_match_stereo(",
        "stereo_fusion(",
        "fuse_dense_frames(",
    ):
        assert forbidden_call not in text
    assert '"dense_provider_promoted": False' in text
    assert '"i4_3_provider_architecture_authorized": False' in text
    assert '"metric_scale_claimed": False' in text
    assert '"surface_mesh_generated": False' in text


def test_a31_runner_keeps_unit_integration_stress_and_previous_a3_coverage() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_i4_1_9a_a3_1_tests.sh").read_text(encoding="utf-8")
    assert "test_i4_1_9a_mvs_visual_ab.py" in text
    assert "test_i4_1_9a_mvs_visual_ab_closure.py" in text
    assert "test_i4_1_9a_mvs_visual_ab_closure_contract.py" in text
    assert "test_i4_1_9a_mvs_visual_ab_closure_stress.py" in text
    assert "focused|stress|all" in text
