from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.reconstruction.live_contracts import LiveSlamObservation

ROOT = Path(__file__).resolve().parents[2]


def test_live_slam_observation_contract_is_not_rewritten_for_optimized_trajectory() -> None:
    assert [field.name for field in fields(LiveSlamObservation)] == [
        "session_id",
        "frame_id",
        "sequence_index",
        "tracking_state",
        "quality",
        "pose",
        "is_keyframe",
        "keyframe_id",
        "map_snapshot",
    ]


def test_reconstruction_ui_remains_disabled() -> None:
    source = (ROOT / "app" / "widgets" / "mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = source.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "AppMode.RECONSTRUCTION" not in enabled_block


def test_application_live_bridge_still_defaults_loop_and_global_features_off() -> None:
    source = (ROOT / "app" / "live_slam_controller.py").read_text(encoding="utf-8")
    assert "enable_loop_closure=False" in source
    assert "enable_global_optimization=False" in source


def test_phase8_5c_does_not_introduce_dense_fusion_or_metric_claims() -> None:
    source = (ROOT / "core" / "reconstruction" / "live_trajectory.py").read_text(encoding="utf-8")
    assert "unit_to_metre" in source
    assert "SfmScaleStatus.ARBITRARY" in source
    assert "DenseFusion" not in source
