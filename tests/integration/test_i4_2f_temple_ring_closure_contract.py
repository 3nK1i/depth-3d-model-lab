from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core/reconstruction/temple_ring_closure.py"
RUNNER = ROOT / "scripts/run_i4_2f_finalize_closure.py"
RUNNER_TESTS = ROOT / "scripts/run_i4_2f_tests.sh"


def test_i42f_runner_requires_explicit_manual_confirmation_and_verdict() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'p.add_argument("--verdict"' in text
    assert "required=True" in text
    assert 'p.add_argument("--confirm-manual-verdict", action="store_true")' in text
    assert "build_i4_2f_closure" in text


def test_i42f_closure_is_evidence_only_and_does_not_run_heavy_reconstruction() -> None:
    text = (CORE.read_text(encoding="utf-8") + RUNNER.read_text(encoding="utf-8")).lower()
    forbidden_calls = (
        "incremental_mapping(", "extract_features(", "match_exhaustive(", "patch_match_stereo(",
        "stereo_fusion(", "bundle_adjustment(", "undistort_images(", "run_temple_ring_mvs(",
    )
    for token in forbidden_calls:
        assert token not in text
    assert '"dense_provider_promoted": false' in text
    assert '"production_filter_policy_changed": false' in text


def test_i42f_test_runner_keeps_full_i42_stack_plus_unit_integration_stress() -> None:
    text = RUNNER_TESTS.read_text(encoding="utf-8")
    assert "test_i4_2a_temple_ring_preflight.py" in text
    assert "test_i4_2e_2a_native_visibility_coverage_stress.py" in text
    assert "test_i4_2f_temple_ring_closure.py" in text
    assert "test_i4_2f_temple_ring_closure_contract.py" in text
    assert "test_i4_2f_temple_ring_closure_stress.py" in text
    assert "I4.2F backward isolated native-visibility probe: PASS" in text
    assert "expected_views=6" in text
