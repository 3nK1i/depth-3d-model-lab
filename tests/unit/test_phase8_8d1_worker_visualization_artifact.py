from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_reconstruction_offline_job import (
    OFFLINE_VISUALIZATION_LANDMARK_CAP,
    OFFLINE_VISUALIZATION_SCHEMA,
    _sample_indices,
    _write_offline_visualization,
)


def _landmark(index: int):
    return SimpleNamespace(
        position_reconstruction=np.asarray([float(index), 0.0, 1.0], dtype=np.float64),
        color_rgb=(index % 255, 20, 30),
    )


def _camera(index: int):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(index)
    return SimpleNamespace(
        estimate_id=f"estimate-{index}",
        keyframe_id=f"kf-{index}",
        frame_id=f"frame-{index}",
        T_reconstruction_from_camera=pose,
    )


def test_d1_sampling_is_bounded_deterministic_and_preserves_endpoints() -> None:
    indices = _sample_indices(20_000, 8192)
    assert len(indices) == 8192
    assert indices[0] == 0
    assert indices[-1] == 19_999
    assert tuple(sorted(indices)) == indices
    assert len(set(indices)) == len(indices)


def test_d1_worker_writes_bounded_visualization_companion(tmp_path: Path) -> None:
    result = SimpleNamespace(
        session_id="22222222-2222-4222-8222-222222222222",
        backend_id="pycolmap-offline-sfm",
        backend_version="3.13",
        gauge=SimpleNamespace(
            scale_status=SimpleNamespace(value="arbitrary"),
            alignment_status=SimpleNamespace(value="arbitrary"),
        ),
        cameras=tuple(_camera(index) for index in range(5)),
        landmarks=tuple(_landmark(index) for index in range(OFFLINE_VISUALIZATION_LANDMARK_CAP + 100)),
        statistics=SimpleNamespace(mean_reprojection_error_px=0.51),
    )
    path = tmp_path / "offline_visualization.json"
    _write_offline_visualization(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == OFFLINE_VISUALIZATION_SCHEMA
    assert payload["scale_status"] == "arbitrary"
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["certified_metrology"] is False
    assert payload["registered_camera_total"] == 5
    assert len(payload["registered_cameras"]) == 5
    assert payload["landmark_total"] == OFFLINE_VISUALIZATION_LANDMARK_CAP + 100
    assert len(payload["landmarks_reconstruction"]) == OFFLINE_VISUALIZATION_LANDMARK_CAP
    assert len(payload["landmark_colors_rgb"]) == OFFLINE_VISUALIZATION_LANDMARK_CAP
