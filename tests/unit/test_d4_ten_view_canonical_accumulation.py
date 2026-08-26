from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_d4_ten_view_canonical_accumulation import (
    _accumulate_ten_view_clouds,
    _normalize_adjacent_pairs,
    _select_ten_registered_keyframes,
    _window_contains_excluded_pair,
)


def _selection_fixture():
    cameras = []
    images = []
    for index in range(22):
        center = np.asarray([float(index) * 0.15, 0.02 * (index % 3), 0.0], dtype=np.float64)
        cameras.append(
            SimpleNamespace(
                keyframe_id=f"kf{index}",
                camera_id="cam",
                frame_id=f"frame{index}",
                camera_center_reconstruction=center,
            )
        )
        images.append(
            SimpleNamespace(order=index, keyframe_id=f"kf{index}", source_sequence_index=index * 10)
        )

    landmarks = []
    track_index = 0
    for left in range(21):
        # Strong, non-zero adjacent overlap everywhere; make later windows a bit
        # stronger so the exclusion test proves known dense-quality evidence can
        # override sparse-only ranking.
        count = 40 + left
        for _ in range(count):
            landmarks.append(
                SimpleNamespace(
                    observations=(
                        SimpleNamespace(keyframe_id=f"kf{left}"),
                        SimpleNamespace(keyframe_id=f"kf{left + 1}"),
                    ),
                    position_reconstruction=np.asarray(
                        [
                            float(track_index % 23) * 0.1,
                            float((track_index // 23) % 17) * 0.1,
                            1.0 + 0.005 * track_index,
                        ],
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


def test_qualified_five_with_excluded_dense_pair_selects_only_safe_ten_window() -> None:
    selected = _select_ten_registered_keyframes(
        _selection_fixture(),
        qualified_keyframe_ids=("kf11", "kf12", "kf13", "kf14", "kf15"),
        excluded_adjacent_pairs=(("kf15", "kf16"),),
    )
    assert [item[0].keyframe_id for item in selected] == [f"kf{i}" for i in range(6, 16)]


def test_explicit_ten_selection_is_sorted_by_workspace_order() -> None:
    selected = _select_ten_registered_keyframes(
        _selection_fixture(),
        explicit_keyframe_ids=("kf12", "kf2", "kf10", "kf4", "kf8", "kf1", "kf6", "kf11", "kf3", "kf9"),
    )
    assert [item[0].keyframe_id for item in selected] == [
        "kf1",
        "kf2",
        "kf3",
        "kf4",
        "kf6",
        "kf8",
        "kf9",
        "kf10",
        "kf11",
        "kf12",
    ]


def test_qualified_five_must_be_registered_consecutive_and_ordered() -> None:
    fixture = _selection_fixture()
    with pytest.raises(ReconstructionValidationError, match="consecutive registered workspace order"):
        _select_ten_registered_keyframes(
            fixture,
            qualified_keyframe_ids=("kf11", "kf12", "kf14", "kf15", "kf16"),
        )
    with pytest.raises(ReconstructionValidationError, match="consecutive registered workspace order"):
        _select_ten_registered_keyframes(
            fixture,
            qualified_keyframe_ids=("kf15", "kf14", "kf13", "kf12", "kf11"),
        )
    with pytest.raises(ReconstructionValidationError, match="registered/materialized"):
        _select_ten_registered_keyframes(
            fixture,
            qualified_keyframe_ids=("kf11", "kf12", "kf13", "kf14", "missing"),
        )


def test_excluded_adjacent_pair_is_order_independent_and_detected() -> None:
    normalized = _normalize_adjacent_pairs((("kf16", "kf15"),))
    assert normalized == frozenset({("kf15", "kf16")})
    window = tuple(SimpleNamespace(keyframe_id=f"kf{i}") for i in range(10, 18))
    assert _window_contains_excluded_pair(window, normalized) is True
    assert _window_contains_excluded_pair(window, frozenset({("kf18", "kf19")})) is False


def test_exclusion_can_make_qualified_ten_selection_impossible() -> None:
    with pytest.raises(ReconstructionValidationError, match="excluded adjacent-pair evidence"):
        _select_ten_registered_keyframes(
            _selection_fixture(),
            qualified_keyframe_ids=("kf0", "kf1", "kf2", "kf3", "kf4"),
            excluded_adjacent_pairs=(("kf3", "kf4"),),
        )


def test_ten_view_accumulation_concatenates_without_merging_and_retains_source() -> None:
    clouds = []
    for view_index in range(10):
        points = np.asarray(
            [[float(view_index), 0.0, 1.0], [float(view_index), 1.0, 1.0]], dtype=np.float64
        )
        colors = np.full((2, 3), 20 + view_index, dtype=np.uint8)
        clouds.append((points, colors))
    points, colors, source = _accumulate_ten_view_clouds(clouds)
    assert points.shape == (20, 3)
    assert colors.shape == (20, 3)
    assert source.tolist() == [value for index in range(10) for value in (index, index)]


def test_ten_view_accumulation_rejects_wrong_count_or_shape() -> None:
    point = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    color = np.asarray([[1, 2, 3]], dtype=np.uint8)
    with pytest.raises(ReconstructionValidationError, match="exactly ten"):
        _accumulate_ten_view_clouds([(point, color)] * 9)
    with pytest.raises(ReconstructionValidationError, match="matching Nx3"):
        _accumulate_ten_view_clouds(
            [(point, color)] * 9 + [(point, np.asarray([[1, 2]], dtype=np.uint8))]
        )
