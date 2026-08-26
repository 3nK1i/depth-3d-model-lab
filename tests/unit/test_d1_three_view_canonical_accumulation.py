from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_d1_three_view_canonical_accumulation import (
    _accumulate_view_clouds,
    _select_three_registered_keyframes,
    _sfm_result_from_payload,
)


def _selection_fixture():
    cameras = tuple(
        SimpleNamespace(keyframe_id=f"kf{i}", camera_id="cam", frame_id=f"frame{i}")
        for i in range(9)
    )
    images = tuple(
        SimpleNamespace(order=i, keyframe_id=f"kf{i}", source_sequence_index=i * 10)
        for i in range(9)
    )
    # Observation counts are deliberately strongest at kf0/kf4/kf7, but the
    # B2/C anchor kf1 must remain selected in the first temporal third.
    desired_counts = {"kf0": 90, "kf1": 60, "kf2": 20, "kf3": 30, "kf4": 100, "kf5": 40,
                      "kf6": 50, "kf7": 110, "kf8": 10}
    landmarks = []
    for keyframe_id, count in desired_counts.items():
        for index in range(count):
            landmarks.append(
                SimpleNamespace(
                    observations=(SimpleNamespace(keyframe_id=keyframe_id),),
                    position_reconstruction=np.zeros(3),
                    color_rgb=None,
                )
            )
    return SimpleNamespace(
        cameras=cameras,
        workspace=SimpleNamespace(images=images),
        landmarks=tuple(landmarks),
    )


def test_default_selection_uses_temporal_thirds_and_preserves_b2_anchor() -> None:
    selected = _select_three_registered_keyframes(
        _selection_fixture(), b2_anchor_keyframe_id="kf1"
    )
    assert [item[0].keyframe_id for item in selected] == ["kf1", "kf4", "kf7"]
    assert [item[2] for item in selected] == [60, 100, 110]


def test_explicit_selection_is_sorted_by_workspace_order() -> None:
    selected = _select_three_registered_keyframes(
        _selection_fixture(),
        b2_anchor_keyframe_id="kf1",
        explicit_keyframe_ids=("kf8", "kf2", "kf5"),
    )
    assert [item[0].keyframe_id for item in selected] == ["kf2", "kf5", "kf8"]


def test_explicit_selection_rejects_missing_or_duplicate_views() -> None:
    fixture = _selection_fixture()
    with pytest.raises(ReconstructionValidationError, match="exactly three distinct"):
        _select_three_registered_keyframes(
            fixture,
            b2_anchor_keyframe_id="kf1",
            explicit_keyframe_ids=("kf1", "kf1", "kf2"),
        )
    with pytest.raises(ReconstructionValidationError, match="registered/materialized"):
        _select_three_registered_keyframes(
            fixture,
            b2_anchor_keyframe_id="kf1",
            explicit_keyframe_ids=("kf1", "kf2", "missing"),
        )


def test_accumulation_concatenates_without_merging_and_retains_source_view() -> None:
    clouds = []
    for view_index in range(3):
        points = np.asarray(
            [[float(view_index), 0.0, 1.0], [float(view_index), 1.0, 1.0]], dtype=np.float64
        )
        colors = np.full((2, 3), 20 + view_index, dtype=np.uint8)
        clouds.append((points, colors))
    points, colors, source = _accumulate_view_clouds(clouds)
    assert points.shape == (6, 3)
    assert colors.shape == (6, 3)
    assert source.tolist() == [0, 0, 1, 1, 2, 2]
    assert np.array_equal(points[:2], clouds[0][0])
    assert np.array_equal(points[2:4], clouds[1][0])
    assert np.array_equal(points[4:], clouds[2][0])


def test_accumulation_requires_exactly_three_matching_xyzrgb_clouds() -> None:
    point = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    color = np.asarray([[1, 2, 3]], dtype=np.uint8)
    with pytest.raises(ReconstructionValidationError, match="exactly three"):
        _accumulate_view_clouds([(point, color), (point, color)])
    with pytest.raises(ReconstructionValidationError, match="matching Nx3"):
        _accumulate_view_clouds(
            [(point, color), (point, color), (point, np.asarray([[1, 2]], dtype=np.uint8))]
        )


def _minimal_sfm_payload() -> dict:
    session_id = "11111111-1111-4111-8111-111111111111"
    images = []
    cameras = []
    for index in range(3):
        keyframe_id = f"kf{index}"
        images.append(
            {
                "order": index,
                "keyframe_id": keyframe_id,
                "frame_id": f"frame{index}",
                "source_sequence_index": index,
                "timestamp_ns": index * 1_000_000,
                "image_size": {"width": 4, "height": 3},
                "source_locator": f"source:{index}",
                "staged_image_path": f"images/{keyframe_id}.png",
                "staged_sha256": f"{index + 1:064x}",
            }
        )
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = float(index)
        cameras.append(
            {
                "estimate_id": f"est{index}",
                "camera_id": "cam0",
                "keyframe_id": keyframe_id,
                "frame_id": f"frame{index}",
                "backend_image_id": index + 1,
                "T_reconstruction_from_camera": transform.tolist(),
            }
        )
    observations = [
        {
            "observation_id": f"obs{index}",
            "landmark_id": "lm0",
            "keyframe_id": f"kf{index}",
            "feature_xy": [1.0 + 0.1 * index, 1.0],
            "feature_index": index,
            "backend_point2d_index": index,
        }
        for index in range(3)
    ]
    return {
        "schema_version": 1,
        "session_id": session_id,
        "backend_id": "pycolmap-offline-sfm",
        "backend_version": "1",
        "gauge": {
            "scale_status": "arbitrary",
            "alignment_status": "arbitrary",
            "unit_to_metre": None,
            "scale_source": "",
            "relative_scale_uncertainty": None,
        },
        "cameras": cameras,
        "camera_models": [
            {
                "camera_id": "cam0",
                "backend_camera_id": 1,
                "backend_model_name": "PINHOLE",
                "image_size": {"width": 4, "height": 3},
                "parameters": [3.0, 3.0, 1.5, 1.0],
            }
        ],
        "landmarks": [
            {
                "landmark_id": "lm0",
                "track_id": "track0",
                "position_reconstruction": [0.0, 0.0, 2.0],
                "observations": observations,
                "color_rgb": [20, 30, 40],
            }
        ],
        "statistics": {
            "input_keyframes": 3,
            "registered_keyframes": 3,
            "landmarks": 1,
            "observations": 3,
            "elapsed_ns": 1,
            "mean_reprojection_error_px": 0.5,
        },
        "workspace": {
            "schema_version": 1,
            "session_id": session_id,
            "source": {
                "source_id": "source0",
                "kind": "recorded_video",
                "path": "offline",
                "display_name": "sample.mp4",
                "source_uri": "/tmp/sample.mp4",
                "expected_image_size": None,
                "expected_frame_rate_hz": 30.0,
                "timestamp_origin": "presentation_timestamp",
                "fingerprint": "",
                "metadata": {},
            },
            "workspace_root": "/tmp/workspace",
            "paths": {
                "images_directory": "images",
                "database_path": "database.db",
                "sparse_directory": "sparse",
                "checkpoints_directory": "checkpoints",
                "reports_directory": "reports",
                "logs_directory": "logs",
            },
            "images": images,
        },
        "warnings": [],
        "metadata": {"purpose": "test"},
    }


def test_sfm_json_rehydration_preserves_validated_world_contracts() -> None:
    result = _sfm_result_from_payload(_minimal_sfm_payload())
    assert result.session_id == "11111111-1111-4111-8111-111111111111"
    assert result.gauge.scale_status.value == "arbitrary"
    assert len(result.cameras) == 3
    assert len(result.camera_models) == 1
    assert len(result.landmarks) == 1
    assert result.statistics.observations == 3
    assert result.workspace.images[2].keyframe_id == "kf2"
    assert np.allclose(result.cameras[2].camera_center_reconstruction, [2.0, 0.0, 0.0])
