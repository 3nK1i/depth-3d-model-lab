from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from app.reconstruction_da3_full_scene import (
    DA3_FULL_SCENE_PROTOCOL,
    DA3FullSceneProfile,
    _default_da3_python_path,
    _default_da3_repository_path,
    build_full_scene_da3_job,
    estimate_overlap_similarity,
    plan_overlapping_windows,
    transform_chunk_to_global_gauge,
)


def _w2c(rotation_c2w: np.ndarray, center: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = rotation_c2w.T
    output[:3, 3] = -rotation_c2w.T @ center
    return output


def test_e2_profile_is_bounded_three_view_overlap_and_arbitrary_scale_safe() -> None:
    profile = DA3FullSceneProfile()
    assert profile.window_size == 3
    assert profile.window_overlap == 2
    assert profile.maximum_keyframes == 96
    assert profile.process_res == 392
    assert "Full Scene" in profile.profile_label
    with pytest.raises(ValueError):
        DA3FullSceneProfile(window_size=4, window_overlap=2)


def test_e2_window_plan_covers_complete_selected_sequence_with_two_view_overlap() -> None:
    assert plan_overlapping_windows(3) == ((0, 1, 2),)
    assert plan_overlapping_windows(6) == (
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
    )
    with pytest.raises(ValueError):
        plan_overlapping_windows(2)


def test_e2_overlap_alignment_recovers_known_arbitrary_similarity() -> None:
    local_rotation = np.eye(3)
    global_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([3.0, 4.0, 0.5])
    scale = 2.5
    local_centres = [np.array([0.0, 0.0, 0.0]), np.array([0.8, 0.1, 0.0])]
    local_ext = [_w2c(local_rotation, centre) for centre in local_centres]
    global_ext = [
        _w2c(global_rotation @ local_rotation, scale * (global_rotation @ centre) + translation)
        for centre in local_centres
    ]
    local_depth = [np.full((16, 20), 2.0, np.float32), np.full((16, 20), 3.0, np.float32)]
    global_depth = [value * scale for value in local_depth]

    recovered_scale, recovered_rotation, recovered_translation = estimate_overlap_similarity(
        global_ext, local_ext, global_depth, local_depth
    )
    assert recovered_scale == pytest.approx(scale, rel=1e-5)
    assert recovered_rotation == pytest.approx(global_rotation, abs=1e-5)
    assert recovered_translation == pytest.approx(translation, abs=1e-5)

    transformed_ext, transformed_depth = transform_chunk_to_global_gauge(
        np.stack(local_ext),
        np.stack(local_depth),
        scale=recovered_scale,
        rotation=recovered_rotation,
        translation=recovered_translation,
    )
    assert transformed_ext == pytest.approx(np.stack(global_ext), abs=1e-5)
    assert transformed_depth == pytest.approx(np.stack(global_depth), abs=1e-5)


def test_e2_alignment_rejects_insufficient_or_unstable_overlap() -> None:
    pose = _w2c(np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="exactly two"):
        estimate_overlap_similarity([pose], [pose], [np.ones((8, 8))], [np.ones((8, 8))])
    bad_local = [np.ones((8, 8), np.float32), np.ones((8, 8), np.float32)]
    bad_global = [value * 100.0 for value in bad_local]
    with pytest.raises(ValueError, match="unstable"):
        estimate_overlap_similarity([pose, pose], [pose, pose], bad_global, bad_local)


def test_e2_build_contract_uses_e1_dense_job_and_external_da3_provenance(tmp_path: Path) -> None:
    video = tmp_path / "office.mp4"
    video.write_bytes(b"video")
    da3_python = tmp_path / "depth3d-da3-python"
    da3_python.write_text("fixture", encoding="utf-8")
    repo = tmp_path / "depth-anything-3"
    repo.mkdir()
    spec, command = build_full_scene_da3_job(
        video_path=video,
        output_root=tmp_path / "outputs",
        da3_python=da3_python,
        da3_repository=repo,
    )
    assert spec.expected_protocol == DA3_FULL_SCENE_PROTOCOL
    assert spec.kind.value == "dense_geometry"
    assert spec.scene_id == ""
    assert spec.source_label == "office.mp4"
    assert "Full Scene" in spec.profile_label
    assert command.program == Path(sys.executable).resolve()
    joined = " ".join(command.arguments)
    assert "run_phase9_6e2_full_scene_da3_job.py" in joined
    assert str(video.resolve()) in command.arguments
    assert str(da3_python.resolve()) in command.arguments
    assert str(repo.resolve()) in command.arguments
