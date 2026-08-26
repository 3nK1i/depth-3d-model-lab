from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_d3_five_view_canonical_accumulation import (
    _accumulate_five_view_clouds,
    _adjacent_chain_evidence,
    _select_five_registered_keyframes,
    _shared_track_counts,
)


def _selection_fixture():
    cameras = []
    images = []
    for index in range(10):
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = float(index) * 0.2
        camera = SimpleNamespace(
            keyframe_id=f"kf{index}",
            camera_id="cam",
            frame_id=f"frame{index}",
            camera_center_reconstruction=transform[:3, 3].copy(),
        )
        cameras.append(camera)
        images.append(
            SimpleNamespace(order=index, keyframe_id=f"kf{index}", source_sequence_index=index * 10)
        )

    pair_tracks = {
        (0, 1): 5,
        (1, 2): 60,
        (2, 3): 80,
        (3, 4): 90,
        (4, 5): 95,
        (5, 6): 15,
        (6, 7): 10,
        (7, 8): 10,
        (8, 9): 10,
    }
    landmarks = []
    track_index = 0
    for (left, right), count in pair_tracks.items():
        for _ in range(count):
            landmarks.append(
                SimpleNamespace(
                    observations=(
                        SimpleNamespace(keyframe_id=f"kf{left}"),
                        SimpleNamespace(keyframe_id=f"kf{right}"),
                    ),
                    position_reconstruction=np.asarray(
                        [float(track_index % 17), float((track_index // 17) % 13), 1.0 + 0.01 * track_index],
                        dtype=np.float64,
                    ),
                    color_rgb=None,
                )
            )
            track_index += 1
    return SimpleNamespace(
        cameras=tuple(cameras),
        workspace=SimpleNamespace(images=tuple(images)),
        landmarks=tuple(landmarks),
    )


def test_overlap_aware_selection_expands_qualified_triplet_into_best_five_window() -> None:
    selected = _select_five_registered_keyframes(
        _selection_fixture(),
        qualified_keyframe_ids=("kf3", "kf4", "kf5"),
    )
    # The 1..5 window has the strongest weakest adjacent shared-track support
    # while preserving the qualified 3/4/5 chain.
    assert [item[0].keyframe_id for item in selected] == ["kf1", "kf2", "kf3", "kf4", "kf5"]


def test_explicit_five_selection_is_sorted_by_workspace_order() -> None:
    selected = _select_five_registered_keyframes(
        _selection_fixture(),
        explicit_keyframe_ids=("kf8", "kf2", "kf6", "kf1", "kf4"),
    )
    assert [item[0].keyframe_id for item in selected] == ["kf1", "kf2", "kf4", "kf6", "kf8"]


def test_qualified_triplet_must_be_registered_consecutive_and_ordered() -> None:
    fixture = _selection_fixture()
    with pytest.raises(ReconstructionValidationError, match="consecutive registered workspace order"):
        _select_five_registered_keyframes(
            fixture,
            qualified_keyframe_ids=("kf3", "kf5", "kf6"),
        )
    with pytest.raises(ReconstructionValidationError, match="consecutive registered workspace order"):
        _select_five_registered_keyframes(
            fixture,
            qualified_keyframe_ids=("kf5", "kf4", "kf3"),
        )
    with pytest.raises(ReconstructionValidationError, match="registered/materialized"):
        _select_five_registered_keyframes(
            fixture,
            qualified_keyframe_ids=("kf3", "kf4", "missing"),
        )


def test_shared_track_and_chain_evidence_report_overlap_and_baseline() -> None:
    fixture = _selection_fixture()
    selected = _select_five_registered_keyframes(
        fixture,
        qualified_keyframe_ids=("kf3", "kf4", "kf5"),
    )
    shared = _shared_track_counts(fixture)
    evidence = _adjacent_chain_evidence(selected, shared_counts=shared, scene_scale=10.0)
    assert len(evidence) == 4
    assert evidence[0]["shared_sparse_landmarks"] == 60
    assert evidence[0]["shared_over_min_observations"] > 0.0
    assert evidence[0]["camera_baseline_reconstruction_units"] == pytest.approx(0.2)
    assert evidence[0]["camera_baseline_scene_fraction"] == pytest.approx(0.02)


def test_five_view_accumulation_concatenates_without_merging_and_retains_source() -> None:
    clouds = []
    for view_index in range(5):
        points = np.asarray(
            [[float(view_index), 0.0, 1.0], [float(view_index), 1.0, 1.0]], dtype=np.float64
        )
        colors = np.full((2, 3), 20 + view_index, dtype=np.uint8)
        clouds.append((points, colors))
    points, colors, source = _accumulate_five_view_clouds(clouds)
    assert points.shape == (10, 3)
    assert colors.shape == (10, 3)
    assert source.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_five_view_accumulation_rejects_wrong_count_or_shape() -> None:
    point = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    color = np.asarray([[1, 2, 3]], dtype=np.uint8)
    with pytest.raises(ReconstructionValidationError, match="exactly five"):
        _accumulate_five_view_clouds([(point, color)] * 4)
    with pytest.raises(ReconstructionValidationError, match="matching Nx3"):
        _accumulate_five_view_clouds(
            [(point, color)] * 4 + [(point, np.asarray([[1, 2]], dtype=np.uint8))]
        )
