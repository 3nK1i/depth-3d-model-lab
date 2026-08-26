from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_detail_qualification import (
    PRODUCTION_VOXEL_SCENE_FRACTION,
    TempleRingDetailTransferPolicy,
    build_production_scale_mvs_payload,
    canonical_cloud_digest,
    json_safe_qualification,
    qualify_d12_detail_transfer,
    validate_fusion_track_evidence,
    validate_fusion_visibility_evidence,
)


def _tracks(n: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(0, 3 * n + 1, 3, dtype=np.int64)
    flat: list[int] = []
    for i in range(n):
        flat.extend((0, 1 + (i % 46), 1 + ((i + 1) % 46)))
    return offsets, np.asarray(flat, dtype=np.int32)


def _dense_raw_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, 256)
    ys = np.linspace(0.0, 0.06, 16)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    colors = np.tile(np.asarray([[120, 160, 190]], dtype=np.uint8), (len(points), 1))
    offsets, image_ids = _tracks(len(points))
    return points, colors, offsets, image_ids


def _extended_fixture() -> dict[str, np.ndarray]:
    grid = np.asarray(
        [[x, y, z] for x in range(7) for y in range(7) for z in range(4)],
        dtype=np.float64,
    )
    tail = np.column_stack((
        np.arange(30.0, 110.0, 4.0),
        np.full(20, 3.0),
        np.full(20, 1.5),
    ))
    points = np.vstack((grid, tail))
    n_core = len(grid)
    classes = np.r_[np.full(n_core, 3, dtype=np.uint8), np.full(len(tail), 2, dtype=np.uint8)]
    support = np.full(len(points), 3, dtype=np.uint32)
    high = np.r_[np.full(n_core, 8, dtype=np.int32), np.zeros(len(tail), dtype=np.int32)]
    colors = np.tile(np.asarray([[100, 150, 200]], dtype=np.uint8), (len(points), 1))
    return {
        "points_reconstruction": points,
        "colors_rgb": colors,
        "view_observation_counts": support,
        "high_support_neighbor_count": high,
        "class_code": classes,
        "voxel_size_reconstruction_units": np.asarray(1.0, dtype=np.float64),
    }


def test_i42e_policy_is_locked_to_production_scale_defaults() -> None:
    policy = TempleRingDetailTransferPolicy()
    assert policy.voxel_scene_fraction == PRODUCTION_VOXEL_SCENE_FRACTION == 0.004
    assert policy.minimum_component_size == 64
    assert policy.core_max_dispersion_voxels == 0.70
    assert policy.extended_max_dispersion_voxels == 0.95


def test_i42e_policy_refuses_threshold_tuning() -> None:
    with pytest.raises(ReconstructionValidationError, match="voxel_scene_fraction is locked"):
        TempleRingDetailTransferPolicy(voxel_scene_fraction=0.005)
    with pytest.raises(ReconstructionValidationError, match="minimum_component_size is locked"):
        TempleRingDetailTransferPolicy(minimum_component_size=32)


def test_i42e_cloud_digest_is_order_independent() -> None:
    points = np.asarray([[1, 2, 3], [4, 5, 6], [-1, 0, 2]], dtype=np.float64)
    colors = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    assert canonical_cloud_digest(points, colors) == canonical_cloud_digest(points[::-1], colors[::-1])


def test_i42e_cloud_digest_detects_geometry_or_rgb_change() -> None:
    points = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    colors = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    changed = points.copy(); changed[0, 0] += 0.25
    assert canonical_cloud_digest(points, colors) != canonical_cloud_digest(changed, colors)
    changed_rgb = colors.copy(); changed_rgb[0, 0] += 1
    assert canonical_cloud_digest(points, colors) != canonical_cloud_digest(points, changed_rgb)


def test_i42e_track_evidence_accepts_valid_visibility_subset() -> None:
    points = np.zeros((2, 3), dtype=np.float64)
    colors = np.zeros((2, 3), dtype=np.uint8)
    offsets = np.asarray([0, 2, 4], dtype=np.int64)
    ids = np.asarray([1, 2, 1, 2], dtype=np.int32)
    xyz, rgb, validated_offsets, validated_ids = validate_fusion_track_evidence(
        points, colors, offsets, ids
    )
    np.testing.assert_array_equal(xyz, points)
    np.testing.assert_array_equal(rgb, colors)
    np.testing.assert_array_equal(validated_offsets, offsets)
    np.testing.assert_array_equal(validated_ids, ids)


def test_i42e_track_evidence_rejects_duplicate_image_per_point() -> None:
    points = np.zeros((47, 3), dtype=np.float64)
    colors = np.zeros((47, 3), dtype=np.uint8)
    offsets = np.arange(0, 95, 2, dtype=np.int64)
    ids = np.asarray([[i, i] for i in range(47)], dtype=np.int32).reshape(-1)
    with pytest.raises(ReconstructionValidationError, match="duplicate image indices"):
        validate_fusion_track_evidence(points, colors, offsets, ids)


def test_i42e_mvs_track_reconstruction_builds_large_production_scale_extended_analogue() -> None:
    points, colors, offsets, ids = _dense_raw_fixture()
    payload = build_production_scale_mvs_payload(points, colors, offsets, ids)
    assert len(payload["points_reconstruction"]) >= 256
    assert np.count_nonzero(payload["class_code"] == 3) >= 128
    assert float(payload["voxel_size_reconstruction_units"]) > 0.0
    assert payload["diagnostics"]["production_i1_e2_scale_and_classification_defaults_mirrored"] is True
    assert payload["diagnostics"]["exact_da3_f_upstream_evidence_reproduction"] is False


def test_i42e_exact_d12_predicate_rejects_far_sparse_tail_in_synthetic_object() -> None:
    result = qualify_d12_detail_transfer(_extended_fixture())
    assert result["policy"]["scenario_name"] == "d12_sparse"
    assert result["policy"]["exact_production_predicate"] is True
    assert result["counts"]["d12_rejected_voxels"] > 0
    assert result["counts"]["post_d12_extended_voxels"] < result["counts"]["pre_d12_extended_voxels"]


def test_i42e_detail_proxies_are_descriptive_not_legitimate_detail_labels() -> None:
    result = qualify_d12_detail_transfer(_extended_fixture())
    definition = result["detail_proxy_definition"]
    assert definition["legitimate_detail_label_claimed"] is False
    assert "support>=3" in definition["supported_line_like_proxy"]


def test_i42e_never_auto_selects_detail_winner_or_policy_change() -> None:
    result = qualify_d12_detail_transfer(_extended_fixture())
    assert result["manual_visual_review_required"] is True
    assert result["automatic_detail_winner_selected"] is False
    assert result["production_policy_change_authorized"] is False
    assert result["i4_3_provider_architecture_authorized"] is False


def test_i42e_preserves_non_metric_authority() -> None:
    result = qualify_d12_detail_transfer(_extended_fixture())
    assert result["metric_scale_claimed"] is False
    assert result["measurement_authoritative"] is False
    assert result["certified_metrology"] is False
    assert result["surface_mesh_generated"] is False


def test_i42e_json_safe_summary_drops_large_arrays() -> None:
    result = qualify_d12_detail_transfer(_extended_fixture())
    safe = json_safe_qualification(result)
    assert "arrays" not in safe
    json.dumps(safe, allow_nan=False)


def test_i42e_potential_loss_signal_is_not_an_automatic_fail() -> None:
    result = qualify_d12_detail_transfer(_extended_fixture())
    assert isinstance(result["potential_detail_loss_signal"], bool)
    assert result["manual_visual_review_required"] is True



def _load_support_worker():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "i4_2e_support_worker_test_module",
        root / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_visibility(path: Path, rows: list[list[int]]) -> None:
    import struct
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(rows)))
        for row in rows:
            handle.write(struct.pack("<I", len(row)))
            if row:
                handle.write(struct.pack(f"<{len(row)}I", *row))


def test_i42e_support_worker_reads_native_zero_based_visibility(tmp_path: Path) -> None:
    worker = _load_support_worker()
    path = tmp_path / "fused.ply.vis"
    _write_visibility(path, [[5, 2], [3]])
    offsets, indices = worker._read_points_visibility(path, expected_points=2, expected_views=6)
    assert offsets.tolist() == [0, 2, 3]
    assert indices.tolist() == [2, 5, 3]


def test_i42e_native_visibility_validator_accepts_valid_global_subset() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    colors = np.zeros((2, 3), dtype=np.uint8)
    offsets = np.asarray([0, 2, 3], dtype=np.int64)
    indices = np.asarray([2, 5, 3], dtype=np.int32)
    got = validate_fusion_visibility_evidence(
        points, colors, offsets, indices, expected_views=6
    )
    assert got[2].tolist() == [0, 2, 3]
    assert got[3].tolist() == [2, 5, 3]


def test_i42e_native_visibility_validator_accepts_exact_0_to_46_index_set() -> None:
    points = np.column_stack((np.arange(47, dtype=np.float64), np.zeros(47), np.zeros(47)))
    colors = np.zeros((47, 3), dtype=np.uint8)
    offsets = np.arange(48, dtype=np.int64)
    indices = np.arange(47, dtype=np.int32)
    got = validate_fusion_visibility_evidence(points, colors, offsets, indices)
    assert got[3].tolist() == list(range(47))


def test_i42e_legacy_track_helper_now_uses_native_visibility_semantics() -> None:
    points = np.column_stack((np.arange(47, dtype=np.float64), np.zeros(47), np.zeros(47)))
    colors = np.zeros((47, 3), dtype=np.uint8)
    offsets = np.arange(48, dtype=np.int64)
    indices = np.arange(47, dtype=np.int32)
    got = validate_fusion_track_evidence(points, colors, offsets, indices)
    assert got[3][0] == 0 and got[3][-1] == 46
