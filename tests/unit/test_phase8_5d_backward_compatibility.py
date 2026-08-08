from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.reconstruction.live_contracts import LiveSlamObservation
from core.reconstruction.live_trajectory import LiveTrajectoryPoseLayer

ROOT = Path(__file__).resolve().parents[2]


def test_live_slam_observation_contract_stays_unchanged() -> None:
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


def test_existing_pose_layer_names_remain_stable() -> None:
    assert LiveTrajectoryPoseLayer.TRACKING.value == "tracking"
    assert LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME.value == "optimized_keyframe"
    assert LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED.value == "optimized_propagated"


def test_reconstruction_ui_remains_disabled() -> None:
    source = (ROOT / "app" / "widgets" / "mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = source.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "AppMode.RECONSTRUCTION" not in enabled_block


def test_application_live_bridge_keeps_loop_and_global_features_default_off() -> None:
    source = (ROOT / "app" / "live_slam_controller.py").read_text(encoding="utf-8")
    assert "enable_loop_closure=False" in source
    assert "enable_global_optimization=False" in source


def test_phase8_5d_keeps_native_backends_isolated_and_has_no_dense_fusion() -> None:
    hardening = (ROOT / "core" / "reconstruction" / "live_trajectory_hardening.py").read_text(
        encoding="utf-8"
    )
    trajectory = (ROOT / "core" / "reconstruction" / "live_trajectory.py").read_text(
        encoding="utf-8"
    )
    combined = hardening + trajectory
    assert "import stella_vslam" not in combined
    assert "from stella_vslam" not in combined
    assert "import pycolmap" not in combined
    assert "from pycolmap" not in combined
    assert "DenseFusion" not in combined
    assert "unit_to_metre" in trajectory
