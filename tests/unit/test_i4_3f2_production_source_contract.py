from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.keyframes import KeyframeSelectionPolicy
from core.reconstruction.sampling import FrameSamplingMode, build_frame_sampling_plan
from scripts.run_reconstruction_offline_job import _build_source
from scripts.reconstruction_pycolmap_dense_provider_mvs_worker import validate_dense_sparse


def _resources_mid() -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="Synthetic RTX 2060",
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=2 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        cuda_ready=True,
    )


def test_keyframe_policy_allows_no_fixed_limit() -> None:
    policy = KeyframeSelectionPolicy(maximum_keyframes=None)
    assert policy.maximum_keyframes is None


def test_image_source_resource_reduction_uses_full_span_sampling(tmp_path: Path) -> None:
    import cv2

    images = tmp_path / "images"
    images.mkdir()
    for index in range(100):
        image = np.full((16, 24, 3), index % 255, dtype=np.uint8)
        assert cv2.imwrite(str(images / f"frame-{index:03d}.png"), image)
    args = Namespace(
        images_dir=images,
        images=None,
        video=None,
        recursive=False,
        image_sequence_fps=10.0,
        video_target_fps=2.0,
        max_keyframes=None,
    )
    source, sampling, kind, pre_plan = _build_source(args, _resources_mid())
    assert kind == "image_directory"
    assert pre_plan.target_sampled_frames == 64
    assert sampling.mode is FrameSamplingMode.TARGET_FPS
    assert sampling.maximum_sampled_frames == 64
    resolved = build_frame_sampling_plan(
        sampling,
        estimated_total_frames=source.estimated_total_frames,
        expected_frame_rate_hz=source.metadata.expected_frame_rate_hz,
    )
    assert resolved.full_span_cap_active is True
    assert resolved.target_sequence_indices[0] == 0
    assert resolved.target_sequence_indices[-1] == 99


def test_controller_and_worker_remove_legacy_default_80() -> None:
    root = Path(__file__).resolve().parents[2]
    controller = (root / "app/offline_reconstruction_controller.py").read_text(encoding="utf-8")
    reconstruction_controller = (root / "app/reconstruction_controller.py").read_text(encoding="utf-8")
    worker = (root / "scripts/run_reconstruction_offline_job.py").read_text(encoding="utf-8")
    assert "max_keyframes: int | None = None" in controller
    assert "max_keyframes: int | None = None" in reconstruction_controller
    assert 'parser.add_argument("--max-keyframes", type=int, default=None)' in worker
    assert "default=80" not in worker
    assert "adaptive_pre_sfm_plan.json" in worker
    assert "adaptive_dense_selection_ready.json" in worker
    assert "permanent_global_view_cap" in worker


class _Pose:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def matrix(self):
        return self._matrix


class _Image:
    def __init__(self, name: str, tx: float) -> None:
        self.name = name
        self._matrix = np.asarray(
            [[1.0, 0.0, 0.0, tx], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

    def cam_from_world(self):
        return _Pose(self._matrix)


class _Point:
    def __init__(self, xyz) -> None:
        self.xyz = np.asarray(xyz, dtype=np.float64)


class _Reconstruction:
    def __init__(self, images: dict[int, _Image], points: dict[int, _Point]) -> None:
        self._images = images
        self._points = points

    def reg_image_ids(self):
        return tuple(self._images)

    def image(self, image_id: int):
        return self._images[image_id]

    def point3D_ids(self):
        return tuple(self._points)

    def point3D(self, point_id: int):
        return self._points[point_id]

    def num_reg_images(self):
        return len(self._images)

    def num_points3D(self):
        return len(self._points)

    def num_cameras(self):
        return 1


class _FakePycolmap:
    def __init__(self, source, dense) -> None:
        self._items = {"source": source, "dense": dense}

    def Reconstruction(self, path):
        return self._items[Path(path).name]


def test_mvs_validation_accepts_selected_dense_subset_without_source_mutation(tmp_path: Path) -> None:
    source = _Reconstruction(
        {index: _Image(f"frame-{index}.png", float(index)) for index in range(4)},
        {1: _Point((0, 0, 1)), 2: _Point((1, 0, 1)), 3: _Point((2, 0, 1))},
    )
    dense = _Reconstruction(
        {0: _Image("frame-0.png", 0.0), 3: _Image("frame-3.png", 3.0)},
        {1: _Point((0, 0, 1)), 3: _Point((2, 0, 1))},
    )
    result = validate_dense_sparse(
        _FakePycolmap(source, dense),
        tmp_path / "source",
        tmp_path / "dense",
        ["frame-0.png", "frame-3.png"],
    )
    assert result["pass"] is True
    assert result["source_registered_images"] == 4
    assert result["dense_registered_images"] == 2
    assert result["checks"]["image_names_exact"] is True
    assert result["checks"]["poses_preserved"] is True
    assert result["checks"]["sparse_point_ids_preserved"] is True
    assert result["checks"]["sparse_point_positions_preserved"] is True
