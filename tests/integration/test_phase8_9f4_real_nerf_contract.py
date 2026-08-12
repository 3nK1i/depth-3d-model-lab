from __future__ import annotations

from pathlib import Path

from core.reconstruction import NerfBackendKind, NerfSceneRole

ROOT = Path(__file__).resolve().parents[2]


def test_existing_nerf_contract_remains_appearance_only() -> None:
    assert NerfSceneRole.APPEARANCE_ONLY.value == "appearance_only"
    assert NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value == "nerfstudio_nerfacto_external"


def test_existing_real_nerf_runners_remain_external_to_core() -> None:
    assert (ROOT / "scripts/run_nerf_real_scene_validation.py").is_file()
    assert (ROOT / "scripts/run_nerf_performance_hardening.py").is_file()
    assert (ROOT / "scripts/nerf_workers/nerfstudio_e3_worker.py").is_file()
    assert (ROOT / "scripts/nerf_workers/nerfstudio_e4_worker.py").is_file()
    core_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "core/reconstruction").glob("*.py")
    )
    assert "from nerfstudio" not in core_text
    assert "import nerfstudio" not in core_text
    assert "from tinycudann" not in core_text
    assert "import tinycudann" not in core_text


def test_f4_runner_reuses_existing_e3_e4_contracts_not_new_backend() -> None:
    text = (ROOT / "scripts/run_phase8_9f4_real_nerf_campaign.py").read_text(encoding="utf-8")
    assert "run_nerf_real_scene_validation.py" in text
    assert "run_nerf_performance_hardening.py" in text
    assert "NeuralSceneDataset" in text
    assert "validation_only_bridge" in text
    assert "new_nerf_algorithm_created" in text
    assert "gaussian_used_as_nerf_geometry_input" in text
