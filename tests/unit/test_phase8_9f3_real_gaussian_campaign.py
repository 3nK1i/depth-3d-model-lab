from __future__ import annotations

import importlib.util
from pathlib import Path
from uuid import UUID

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_phase8_9f3_real_gaussian_campaign.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase8_9f3_campaign", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f3_scope_is_validation_only() -> None:
    module = _load()
    assert len(module.F3_REPO_FILES) == 5
    assert not any(path.startswith("core/") or path.startswith("app/") for path in module.F3_REPO_FILES)
    assert module.EXPECTED_HEAD == "a0366d4e889a227bcdbf19ab60c929814eb96c63"


def test_f3_uses_qualified_gsplat_pin() -> None:
    module = _load()
    assert module.EXPECTED_GSPLAT_VERSION == "1.5.3"
    assert module.EXPECTED_GSPLAT_REVISION == "937e29912570c372bed6747a5c9bf85fed877bae"
    assert module.DEFAULT_GSPLAT_PYTHON.name == "python"
    assert "depth3d-gaussian-gsplat" in str(module.DEFAULT_GSPLAT_PYTHON)


def test_benchmark_command_is_gsplat_only_and_bounded(tmp_path: Path) -> None:
    module = _load()
    args = type("Args", (), {
        "training_steps": 64,
        "render_repeats": 3,
        "gsplat_python": Path("/isolated/gsplat/python"),
        "metrics_python": Path("/isolated/metrics/python"),
        "timeout_s": 1800.0,
    })()
    bridge = {
        "dataset_manifest": str(tmp_path / "dataset.json"),
        "gaussian_seed": str(tmp_path / "seed.npz"),
        "validation_frame_id": "middle",
    }
    command = module._benchmark_command(args, tmp_path, bridge)
    joined = " ".join(command)
    assert "--backend gsplat" in joined
    assert "--training-steps 64" in joined
    assert "--render-repeats 3" in joined
    assert "--hardware-profile rtx_2060_6gb" in joined
    assert "graphdeco" not in joined.lower()
    assert "nerf" not in joined.lower()


def test_f3_authority_strings_cannot_claim_metric_or_metrology() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"gaussian": "appearance_only"' in text
    assert '"certified_metrology": False' in text
    assert '"sfm_pose_to_da3_injection": False' in text
    assert '"sfm_fused_into_gaussian_in_f3": False' in text
    assert "quality_metric_floor_invented\": False" in text


def test_f3_isolation_probe_includes_heavy_gaussian_modules() -> None:
    module = _load()
    assert "gsplat" in module.HEAVY_MODULES
    assert "diff_gaussian_rasterization" in module.HEAVY_MODULES
    assert "simple_knn" in module.HEAVY_MODULES


def test_validate_f2_rejects_missing_summary(tmp_path: Path) -> None:
    module = _load()
    with pytest.raises(RuntimeError, match="invalid or missing JSON evidence"):
        module._validate_f2(tmp_path)


def test_gsplat_checkout_missing_fails_closed(tmp_path: Path) -> None:
    module = _load()
    state = module._gsplat_checkout_state(tmp_path / "missing")
    assert state["ready"] is False
    assert state["exists"] is False


def test_f3_validation_bridge_session_id_is_contract_valid_uuid() -> None:
    module = _load()
    session_id = module._new_validation_session_id()
    assert str(UUID(session_id)) == session_id
    assert session_id != "phase8-9f3-real-gaussian-session"
