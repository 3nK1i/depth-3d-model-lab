from types import SimpleNamespace

import numpy as np

from core.snapshot.export import align_point_mask, export_snapshot_package


def _result():
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 2] = 255
    depth = np.linspace(1.0, 2.0, 80, dtype=np.float32).reshape(8, 10)
    return SimpleNamespace(
        rgb=rgb, depth=depth, points=None, colors=None, normals=None,
        intrinsics=None, confidence=None, valid_mask=np.ones_like(depth, dtype=bool),
        model_id="development-mock", model_name="Development Mock", precision="FP32",
        depth_scale="synthetic", is_metric=False, intrinsics_source="none",
        confidence_kind="none", latency_ms=1.0, preprocess_ms=0.0,
        postprocess_ms=0.1, worker_total_ms=1.1, point_stride=1,
    )


def test_snapshot_export_is_complete_and_atomic(tmp_path):
    target = tmp_path / "snapshot"
    exported = export_snapshot_package(target, _result())
    assert exported == target.resolve()
    assert (target / "rgb.png").is_file()
    assert (target / "depth.npy").is_file()
    assert (target / "depth_preview.png").is_file()
    assert (target / "point_cloud.ply").is_file()
    assert (target / "snapshot_manifest.json").is_file()


def test_snapshot_export_aligns_rgb_with_strided_point_grid(tmp_path):
    from core.geometry.export import read_ply_for_verification

    height, width, stride = 12, 16, 3
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    rgb[..., 2] = 200
    depth = np.ones((height, width), dtype=np.float32)
    points = np.zeros((4, 6, 3), dtype=np.float32)
    points[..., 2] = 1.0
    result = SimpleNamespace(
        rgb=rgb,
        depth=depth,
        points=points,
        colors=None,
        normals=None,
        intrinsics=None,
        confidence=None,
        valid_mask=np.ones_like(depth, dtype=bool),
        model_id="moge-2",
        model_name="MoGe-2",
        precision="FP16",
        depth_scale="metric",
        is_metric=True,
        intrinsics_source="model_predicted",
        confidence_kind="none",
        latency_ms=1.0,
        preprocess_ms=0.0,
        postprocess_ms=0.1,
        worker_total_ms=1.1,
        point_stride=stride,
    )

    target = export_snapshot_package(tmp_path / "snapshot_stride", result)
    exported_points, exported_colors = read_ply_for_verification(target / "point_cloud.ply")

    expected_rgb = rgb[:, :, ::-1][::stride, ::stride].reshape(-1, 3)
    assert exported_points.shape == (24, 3)
    assert np.array_equal(exported_colors, expected_rgb)


def test_snapshot_export_resizes_rgb_for_different_depth_grid(tmp_path):
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.ones((4, 5), dtype=np.float32)
    points = np.zeros((2, 3, 3), dtype=np.float32)
    points[..., 2] = 1.0
    result = SimpleNamespace(
        rgb=rgb,
        depth=depth,
        points=points,
        colors=None,
        normals=None,
        intrinsics=None,
        confidence=None,
        valid_mask=np.ones_like(depth, dtype=bool),
        model_id="direct-grid",
        model_name="Direct Grid",
        precision="FP32",
        depth_scale="relative",
        is_metric=False,
        intrinsics_source="estimated",
        confidence_kind="none",
        latency_ms=1.0,
        preprocess_ms=0.0,
        postprocess_ms=0.1,
        worker_total_ms=1.1,
        point_stride=2,
    )

    target = export_snapshot_package(tmp_path / "snapshot_resized", result)
    assert (target / "point_cloud.ply").is_file()


def test_snapshot_point_mask_aligns_with_strided_grid():
    height, width, stride = 12, 16, 3
    points = np.zeros((4, 6, 3), dtype=np.float32)
    mask = np.ones((height, width), dtype=bool)
    mask[0, 0] = False

    aligned = align_point_mask(
        valid_mask=mask,
        points=points,
        point_stride=stride,
    )

    assert aligned.shape == (24,)
    assert not aligned[0]
    assert aligned[1:].all()
