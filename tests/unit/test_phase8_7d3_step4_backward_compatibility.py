from __future__ import annotations

from pathlib import Path

from core.reconstruction import GaussianBenchmarkRunState, GaussianLifecycleEvidence
from scripts.gaussian_benchmark_workers.common import apply_lifecycle_fault_injection

ROOT = Path(__file__).resolve().parents[2]


def test_existing_lifecycle_completed_contract_is_unchanged() -> None:
    evidence = GaussianLifecycleEvidence(
        state=GaussianBenchmarkRunState.COMPLETED,
        worker_exit_code=0,
    )
    assert evidence.as_dict() == {
        "state": "completed",
        "worker_exit_code": 0,
        "cancellation_requested": False,
        "recovery_attempted": False,
        "recovery_succeeded": False,
        "error": "",
    }


def test_normal_worker_request_has_no_fault_side_effect(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    apply_lifecycle_fault_injection(request={}, result_path=result)
    assert not result.exists()


def test_step4_does_not_modify_external_backend_setup_or_ui_contracts() -> None:
    runner = (ROOT / "scripts" / "run_gaussian_backend_robustness_smoke.py").read_text()
    assert "engine_level_recovery_deferred_to_phase8_7g" in runner
    assert "canonical_geometry_authoritative" in runner
    assert "certified_metrology" in runner
    assert "setup_gaussian_backends.sh" not in runner
    assert "reconstruction_ui_enabled" not in runner


def test_normal_real_scene_runner_uses_process_group_supervision() -> None:
    source = (ROOT / "scripts" / "run_gaussian_real_scene_benchmark.py").read_text()
    assert "run_supervised_process" in source
    assert "subprocess.run(" not in source


def test_both_real_backend_workers_bind_fault_helper_explicitly() -> None:
    for name in ("gsplat_worker.py", "graphdeco_worker.py"):
        source = (ROOT / "scripts" / "gaussian_benchmark_workers" / name).read_text()
        assert "apply_lifecycle_fault_injection = _BENCH_COMMON.apply_lifecycle_fault_injection" in source
        assert "apply_lifecycle_fault_injection(request=request, result_path=result_path)" in source
