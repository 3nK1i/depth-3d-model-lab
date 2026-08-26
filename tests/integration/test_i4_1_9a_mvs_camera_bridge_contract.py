from __future__ import annotations

from pathlib import Path


def test_a1_runner_and_external_worker_are_additive_and_stop_before_mvs() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/run_i4_1_9a_a1_camera_bridge.py").read_text(encoding="utf-8")
    worker = (root / "scripts/reconstruction_pycolmap_mvs_bridge_worker.py").read_text(encoding="utf-8")
    core = (root / "core/reconstruction/mvs_camera_bridge.py").read_text(encoding="utf-8")
    assert "patch_match_stereo(" not in runner + worker + core
    assert "stereo_fusion(" not in runner + worker + core
    assert "incremental_mapping(" not in runner + worker + core
    assert "bundle_adjuster(" not in worker.lower()
    assert "bundle_adjustment(" not in worker.lower()
    assert "house_k_txt_consumed" in core
    assert "all 68 source views" in core


def test_a1_does_not_modify_existing_da3_or_i4_1_9_visual_acceptance_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "scripts/run_i4_1_9_visual_reconstruction_acceptance.py").is_file()
    assert (root / "core/reconstruction/offline_dense_i4_visual_acceptance.py").is_file()


def test_a1_runner_exposes_focused_and_stress_modes_with_all_as_default() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/run_i4_1_9a_a1_tests.sh").read_text(encoding="utf-8")
    assert 'mode="${1:-all}"' in runner
    assert "tests/unit/test_i4_1_9a_mvs_camera_bridge.py" in runner
    assert "tests/integration/test_i4_1_9a_mvs_camera_bridge_contract.py" in runner
    assert "tests/stress/test_i4_1_9a_mvs_camera_bridge_stress.py" in runner
    assert "focused|stress|all" in runner
