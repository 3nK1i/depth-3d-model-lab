from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_runner_reuses_unchanged_i1_dense_thresholds() -> None:
    text = (ROOT / "scripts/run_i4_1_6b_decoupled_dense_ab.py").read_text(encoding="utf-8")
    assert "from scripts.run_i1_offline_dense_campaign import _DENSE_THRESHOLDS" in text
    assert '"threshold_relaxation": False' in text


def test_runner_keeps_one_sfm_authority_and_does_not_run_fusion() -> None:
    text = (ROOT / "scripts/run_i4_1_6b_decoupled_dense_ab.py").read_text(encoding="utf-8")
    assert '"single_authoritative_sfm_frame": True' in text
    assert '"pycolmap_rerun": False' in text
    assert '"fusion_rerun": False' in text
    assert '"production_default_changed": False' in text


def test_da3_worker_preserves_single_frame_historical_inference_shape() -> None:
    text = (ROOT / "scripts/da3_workers/i4_1_6b_da3_sequential_worker.py").read_text(encoding="utf-8")
    assert "model.inference(" in text
    assert "[str(path)]" in text
    assert "single_frame_sequential_model_loaded_once" in text
