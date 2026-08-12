from __future__ import annotations

from scripts.run_phase8_9d_gpu_production_profile import _policy_report, _synthetic_hardware, readiness_matrix


def test_d_readiness_matrix_never_claims_measurements() -> None:
    matrix = readiness_matrix()
    assert tuple(matrix) == ("rtx2060", "rtx4090", "rtx5090")
    for report in matrix.values():
        assert report["measured"] is False
        assert report["evidence"] == "policy_readiness_only_not_measured"


def test_d_rtx2060_readiness_uses_safe_efficient_profiles() -> None:
    report = _policy_report(_synthetic_hardware("rtx2060"), evidence="policy_readiness_only_not_measured")
    assert report["automatic_compute_profile"]["effective"] == "efficient"
    assert report["automatic_reconstruction_profile"]["effective"] == "efficient"
    assert report["automatic_compute_profile"]["device"] == "cuda:0"


def test_d_rtx4090_readiness_is_automatic_without_product_branching() -> None:
    report = _policy_report(_synthetic_hardware("rtx4090"), evidence="policy_readiness_only_not_measured")
    assert report["automatic_compute_profile"]["effective"] == "ultra"
    assert report["automatic_reconstruction_profile"]["effective"] == "high_quality"
    assert report["automatic_compute_profile"]["precision"] == "fp16"
    assert report["automatic_reconstruction_profile"]["gpu_budget_gib"] <= 22.0


def test_d_rtx5090_readiness_is_automatic_without_manual_config() -> None:
    report = _policy_report(_synthetic_hardware("rtx5090"), evidence="policy_readiness_only_not_measured")
    assert report["automatic_compute_profile"]["effective"] == "ultra"
    assert report["automatic_reconstruction_profile"]["effective"] == "high_quality"
    assert report["automatic_compute_profile"]["point_budget"] == 1_000_000
    assert report["automatic_reconstruction_profile"]["max_active_keyframes"] == 800
