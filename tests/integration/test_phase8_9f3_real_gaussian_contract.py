from __future__ import annotations

from pathlib import Path

from core.reconstruction import GaussianBackendKind, GaussianSceneRole


ROOT = Path(__file__).resolve().parents[2]


def test_existing_gaussian_contract_remains_appearance_only() -> None:
    assert GaussianSceneRole.APPEARANCE_ONLY.value == "appearance_only"
    assert GaussianBackendKind.GSPLAT_EXTERNAL.value == "gsplat_external"


def test_existing_real_benchmark_worker_is_external_to_core() -> None:
    worker = ROOT / "scripts/gaussian_benchmark_workers/gsplat_worker.py"
    runner = ROOT / "scripts/run_gaussian_real_scene_benchmark.py"
    assert worker.is_file()
    assert runner.is_file()
    core_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "core/reconstruction").glob("*.py")
    )
    assert "from gsplat" not in core_text
    assert "import gsplat" not in core_text


def test_f3_runner_reuses_existing_contracts_not_new_backend() -> None:
    text = (ROOT / "scripts/run_phase8_9f3_real_gaussian_campaign.py").read_text(encoding="utf-8")
    assert "run_gaussian_real_scene_benchmark.py" in text
    assert "build_gaussian_seed" in text
    assert "materialize_appearance_dataset" in text
    assert "validation_only_bridge" in text
    assert "new_gaussian_algorithm_created" in text
