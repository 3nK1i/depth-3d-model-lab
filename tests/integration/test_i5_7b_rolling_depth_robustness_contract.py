from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.reconstruction.i5_rolling_depth_robustness_replay import I57B_SCENARIO_IDS, I57BReplayPolicy
from scripts.run_i5_7b_rolling_depth_robustness_replay import _verify_recovery_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_runner_is_real_rolling_depth_only_and_reuses_frozen_sfm():
    source = (ROOT / "scripts/run_i5_7b_rolling_depth_robustness_replay.py").read_text(encoding="utf-8")
    assert "RollingDepthProductionProviderAdapter" in source
    assert "offline_sfm_result_from_dict" in source
    assert "analyze_i5_6a_sfm_report" in source
    assert "ExternalPycolmapSfmBackend" not in source
    assert "MvsProductionProviderAdapter" not in source
    assert "VideoDepthAnythingProductionProviderAdapter" not in source
    assert "DenseProviderSelectionMode.EXPLICIT_PROVIDER" in source
    assert "allow_fallback=False" in source


def test_runner_requires_pinned_runtime_and_immutable_output_directory():
    source = (ROOT / "scripts/run_i5_7b_rolling_depth_robustness_replay.py").read_text(encoding="utf-8")
    assert "probe_rolling_depth_environment" in source
    assert "runtime_lock_valid" in source
    assert "checkpoint_ready" in source
    assert "output.mkdir(parents=True, exist_ok=False)" in source
    assert "i5_7b_attempt_failure.json" in source
    assert "retry_until_pass" in source


def test_policy_and_runner_freeze_six_scenarios_without_adaptive_reduction():
    policy = I57BReplayPolicy()
    assert policy.scenario_ids == I57B_SCENARIO_IDS
    assert policy.all_registered_views is True
    assert policy.adaptive_view_reduction is False
    assert policy.permanent_global_view_cap is None


def test_runner_has_narrow_hash_verified_castle_control_recovery_without_rerun():
    source = (ROOT / "scripts/run_i5_7b_rolling_depth_robustness_replay.py").read_text(encoding="utf-8")
    assert '"recovery-production-native"' in source
    assert "--recovery-attempt-dir" in source
    assert "--recovery-manifest-sha256" in source
    assert "--recovery-console-log-sha256" in source
    assert "I57B_RECOVERY_EXCEPTION_MESSAGE" in source
    assert "_verify_recovery_manifest" in source
    assert "build_i5_7b_recovered_control_record" in source
    assert "scenario_ids = I57B_SCENARIO_IDS[1:]" in source
    assert '"provider_execution_rerun": False' in source
    assert 'provider_dir / "provider/input_manifest.json"' in source
    assert 'provider_dir / "provider/worker_output/rolling_depth_evidence.npz"' in source


def test_recovery_does_not_copy_or_modify_original_provider_evidence():
    source = (ROOT / "scripts/run_i5_7b_rolling_depth_robustness_replay.py").read_text(encoding="utf-8")
    assert "shutil" not in source
    assert ".replace(" not in source
    assert "unlink(" not in source
    assert "rmtree(" not in source


def test_recovery_manifest_verification_is_immutable_and_detects_tamper(tmp_path: Path):
    attempt = tmp_path / "i5_7b_rolling_depth_production_native_20260908_110816"
    attempt.mkdir()
    evidence = attempt / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    manifest = tmp_path / "attempt.sha256_manifest.txt"
    manifest.write_text(f"{evidence_sha}  {evidence.resolve()}\n", encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    entries = _verify_recovery_manifest(manifest, manifest_sha, attempt)
    assert entries[evidence.resolve()] == evidence_sha

    evidence.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError):
        _verify_recovery_manifest(manifest, manifest_sha, attempt)


def test_recovery_manifest_rejects_paths_outside_preserved_attempt(tmp_path: Path):
    attempt = tmp_path / "i5_7b_rolling_depth_production_native_20260908_110816"
    attempt.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    sha = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest = tmp_path / "attempt.sha256_manifest.txt"
    manifest.write_text(f"{sha}  {outside.resolve()}\n", encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):
        _verify_recovery_manifest(manifest, manifest_sha, attempt)
