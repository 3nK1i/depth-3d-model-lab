from __future__ import annotations

import json
from pathlib import Path

from scripts.gaussian_d3_finalization import finalize_gaussian_d3, write_finalization_outputs


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _d2() -> dict:
    return {
        "phase": "8.7D2", "pass": True, "canonical_geometry_authoritative": False,
        "certified_metrology": False, "reconstruction_ui_enabled": False,
        "qualifications": {"gsplat": {"qualified": True, "status": "qualified"}, "graphdeco": {"qualified": True, "status": "qualified"}},
    }


def _result(kind: str) -> dict:
    return {
        "backend_kind": kind,
        "quality": {"measured": True, "psnr_db": 10.0, "ssim": 0.5, "lpips": 0.4},
        "performance": {"detailed_measured": True, "peak_training_vram_mib": 100.0, "peak_render_vram_mib": 90.0, "training_duration_s": 1.0, "render_duration_s": 0.01, "render_fps": 100.0},
        "hardware": {"measured": True, "profile_compatible": True, "gpu_name": "NVIDIA GeForce RTX 2060", "driver_version": "x", "cuda_runtime_version": "12.8"},
    }


def _step2() -> dict:
    return {"phase": "8.7D3", "step": 2, "pass": True, "quality_comparison_ready": True, "performance_metrics_complete": False, "canonical_geometry_authoritative": False, "certified_metrology": False, "report": {"results": [_result("gsplat_external"), _result("graphdeco_reference_external")]}}


def _step3() -> dict:
    return {"phase": "8.7D3", "step": 3, "pass": True, "quality_comparison_ready": True, "performance_metrics_complete": True, "hardware_evidence_complete": True, "hardware_consistent_across_backends": True, "comparison_ready": True, "measured_hardware_profile": "rtx_2060_6gb", "rtx_5090_readiness_profile": {"profile": "rtx_5090_32gb"}, "canonical_geometry_authoritative": False, "certified_metrology": False, "report": {"results": [_result("gsplat_external"), _result("graphdeco_reference_external")]}}


def _step4() -> dict:
    evidence = []
    for backend in ("gsplat_external", "graphdeco_reference_external"):
        for scenario in ("cancellation", "timeout", "crash", "partial_artifact"):
            evidence.append({
                "backend_kind": backend, "scenario": scenario, "passed": True,
                "process_group_clean": True, "recovery_result_valid": True,
                "recovery_lifecycle": {"state": "recovered", "recovery_succeeded": True},
                "partial_artifacts_detected": scenario == "partial_artifact",
                "partial_artifacts_cleaned": scenario == "partial_artifact",
            })
    return {"phase": "8.7D3", "step": 4, "pass": True, "backend_process_scope_only": True, "engine_level_recovery_deferred_to_phase8_7g": True, "canonical_geometry_authoritative": False, "certified_metrology": False, "report": {"evidence": evidence}}


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "core" / "reconstruction").mkdir(parents=True)
    (tmp_path / "core" / "reconstruction" / "safe.py").write_text("import json\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app.yaml").write_text("reconstruction_mode_enabled: false\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='safe'\n", encoding="utf-8")
    return tmp_path


def test_finalization_closes_d3_only_when_all_evidence_passes(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"; evidence.mkdir()
    result = finalize_gaussian_d3(
        d2_summary_path=_write(evidence / "d2.json", _d2()),
        step2_summary_path=_write(evidence / "s2.json", _step2()),
        step3_summary_path=_write(evidence / "s3.json", _step3()),
        step4_summary_path=_write(evidence / "s4.json", _step4()),
        expected_hardware_profile="rtx_2060_6gb", repository_root=_repo(tmp_path / "repo"),
    )
    assert result.passed is True
    assert result.summary["d3_closed"] is True
    assert result.summary["next_phase"] == "8.7E1"
    assert result.summary["canonical_geometry_authoritative"] is False
    assert result.summary["certified_metrology"] is False


def test_finalization_fails_closed_on_partial_artifact_cleanup_regression(tmp_path: Path) -> None:
    step4 = _step4()
    step4["report"]["evidence"][3]["partial_artifacts_cleaned"] = False
    evidence = tmp_path / "evidence"; evidence.mkdir()
    result = finalize_gaussian_d3(
        d2_summary_path=_write(evidence / "d2.json", _d2()), step2_summary_path=_write(evidence / "s2.json", _step2()),
        step3_summary_path=_write(evidence / "s3.json", _step3()), step4_summary_path=_write(evidence / "s4.json", step4),
        expected_hardware_profile="rtx_2060_6gb", repository_root=_repo(tmp_path / "repo"),
    )
    assert result.passed is False
    assert result.summary["d3_closed"] is False
    assert result.summary["next_phase"] is None


def test_finalization_rejects_hardware_mislabel(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"; evidence.mkdir()
    result = finalize_gaussian_d3(
        d2_summary_path=_write(evidence / "d2.json", _d2()), step2_summary_path=_write(evidence / "s2.json", _step2()),
        step3_summary_path=_write(evidence / "s3.json", _step3()), step4_summary_path=_write(evidence / "s4.json", _step4()),
        expected_hardware_profile="rtx_5090_32gb", repository_root=_repo(tmp_path / "repo"),
    )
    assert result.passed is False
    assert "step3_hardware_profile_mismatch" in result.summary["gates"]["d3_step3_performance"]["reasons"]


def test_finalization_isolation_audit_rejects_heavy_core_import(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "core" / "reconstruction" / "bad.py").write_text("import gsplat\n", encoding="utf-8")
    evidence = tmp_path / "evidence"; evidence.mkdir()
    result = finalize_gaussian_d3(
        d2_summary_path=_write(evidence / "d2.json", _d2()), step2_summary_path=_write(evidence / "s2.json", _step2()),
        step3_summary_path=_write(evidence / "s3.json", _step3()), step4_summary_path=_write(evidence / "s4.json", _step4()),
        expected_hardware_profile="rtx_2060_6gb", repository_root=repo,
    )
    assert result.passed is False
    assert result.summary["gates"]["production_isolation"]["pass"] is False


def test_finalization_writes_json_and_markdown(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"; evidence.mkdir()
    result = finalize_gaussian_d3(
        d2_summary_path=_write(evidence / "d2.json", _d2()), step2_summary_path=_write(evidence / "s2.json", _step2()),
        step3_summary_path=_write(evidence / "s3.json", _step3()), step4_summary_path=_write(evidence / "s4.json", _step4()),
        expected_hardware_profile="rtx_2060_6gb", repository_root=_repo(tmp_path / "repo"),
    )
    json_path, md_path = write_finalization_outputs(result, tmp_path / "out")
    assert json.loads(json_path.read_text())["d3_closed"] is True
    assert "Smoke-scale backend comparison" in md_path.read_text()
