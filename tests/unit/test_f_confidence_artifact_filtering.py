from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from core.reconstruction.dense_confidence_filtering import (
    DenseArtifactReason,
    DenseConfidenceFilterPolicy,
    artifact_reason_histogram,
    dense_filter_confidence,
    filter_dense_supported_geometry,
)
from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_e2_consistency_aware_fusion_refinement import (
    _aggregate_saved_e1_source_frame_voxels,
)
from scripts.run_f_confidence_artifact_filtering import (
    _validate_e3_summary,
    main as f_main,
)


def _e3_summary(*, passed: bool = True, authorized: bool = True, capacity_clean: bool = True):
    orders = list(range(10))
    return {
        "phase": "E3",
        "pass": passed,
        "status": "pass" if passed else "degraded",
        "decision": {"authorizes_next_confidence_filtering_milestone": authorized},
        "fusion": {"capacity_clean": capacity_clean, "capacity_drops": 0 if capacity_clean else 1},
        "selected_workspace_orders": orders,
        "selected_keyframe_ids": [f"kf-{order}" for order in orders],
        "selected_view_count": 10,
        "frames": [{"workspace_order": order, "cloud_npz": f"cloud-{order}.npz"} for order in orders],
        "authority": {
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "canonical_persistence_promoted": False,
            "surface_mesh_generated": False,
        },
    }


def test_validate_e3_summary_requires_passing_authorized_capacity_clean_source():
    _validate_e3_summary(_e3_summary())
    with pytest.raises(ReconstructionValidationError):
        _validate_e3_summary(_e3_summary(passed=False))
    with pytest.raises(ReconstructionValidationError):
        _validate_e3_summary(_e3_summary(authorized=False))
    with pytest.raises(ReconstructionValidationError):
        _validate_e3_summary(_e3_summary(capacity_clean=False))


def test_validate_e3_summary_refuses_metric_or_persistence_promotion():
    metric = _e3_summary()
    metric["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        _validate_e3_summary(metric)
    persisted = _e3_summary()
    persisted["authority"]["canonical_persistence_promoted"] = True
    with pytest.raises(ReconstructionValidationError):
        _validate_e3_summary(persisted)


def test_policy_validates_expanded_view_range_and_threshold_ordering():
    policy = DenseConfidenceFilterPolicy(maximum_views=21)
    assert policy.maximum_views == 21
    with pytest.raises(ReconstructionValidationError):
        DenseConfidenceFilterPolicy(maximum_views=1)
    with pytest.raises(ReconstructionValidationError):
        DenseConfidenceFilterPolicy(
            maximum_views=21,
            support2_core_max_dispersion_voxels=0.8,
            core_max_dispersion_voxels=0.7,
        )


def test_confidence_uses_full_expanded_support_range_instead_of_saturating_at_five():
    policy = DenseConfidenceFilterPolicy(maximum_views=21)
    confidence = dense_filter_confidence(
        np.asarray([2, 5, 10, 21]),
        np.asarray([0.2, 0.2, 0.2, 0.2]),
        np.asarray([20, 20, 20, 20]),
        np.asarray([4, 4, 4, 4]),
        policy=policy,
    )
    assert np.all(np.diff(confidence) > 0.0)
    assert np.all((0.0 <= confidence) & (confidence <= 1.0))


def test_filter_keeps_dense_consistent_core_and_extended_geometry():
    policy = DenseConfidenceFilterPolicy(maximum_views=21, minimum_component_size=64)
    result = filter_dense_supported_geometry(
        np.asarray([5, 2, 3]),
        np.asarray([0.2, 0.3, 0.8]),
        np.asarray([20, 14, 12]),
        np.asarray([5, 3, 3]),
        np.asarray([100, 100, 100]),
        policy=policy,
    )
    assert result.core_mask.tolist() == [True, True, False]
    assert result.extended_mask.tolist() == [True, True, True]
    assert not np.any(result.artifact_reason_code)


def test_support2_extended_requires_high_support_anchor_in_formal_f_policy():
    policy = DenseConfidenceFilterPolicy(maximum_views=21, minimum_component_size=64)
    result = filter_dense_supported_geometry(
        np.asarray([2, 2]),
        np.asarray([0.3, 0.3]),
        np.asarray([20, 20]),
        np.asarray([0, 1]),
        np.asarray([100, 100]),
        policy=policy,
    )
    assert result.extended_mask.tolist() == [False, True]
    assert result.artifact_reason_code[0] & int(DenseArtifactReason.WEAK_HIGH_SUPPORT_ANCHOR)


def test_rejected_voxels_report_interpretable_artifact_reasons():
    policy = DenseConfidenceFilterPolicy(maximum_views=21, minimum_component_size=64)
    result = filter_dense_supported_geometry(
        np.asarray([3, 3, 2]),
        np.asarray([1.2, 0.2, 0.2]),
        np.asarray([20, 1, 20]),
        np.asarray([4, 4, 0]),
        np.asarray([100, 100, 4]),
        policy=policy,
    )
    hist = artifact_reason_histogram(result.artifact_reason_code)
    assert hist["excessive_dispersion"] == 1
    assert hist["weak_local_connectivity"] >= 1
    assert hist["small_component"] >= 1
    assert hist["weak_high_support_anchor"] >= 1


def test_filter_rejects_invalid_supported_counts():
    policy = DenseConfidenceFilterPolicy(maximum_views=21)
    with pytest.raises(ReconstructionValidationError):
        filter_dense_supported_geometry(
            np.asarray([1]), np.asarray([0.2]), np.asarray([10]), np.asarray([2]), np.asarray([100]), policy=policy
        )
    with pytest.raises(ReconstructionValidationError):
        filter_dense_supported_geometry(
            np.asarray([22]), np.asarray([0.2]), np.asarray([10]), np.asarray([2]), np.asarray([100]), policy=policy
        )


def test_artifact_reason_histogram_supports_detached_single_view_diagnostic_bit():
    values = np.asarray([
        int(DenseArtifactReason.SINGLE_VIEW_DETACHED),
        int(DenseArtifactReason.SMALL_COMPONENT | DenseArtifactReason.EXCESSIVE_DISPERSION),
    ], dtype=np.uint16)
    hist = artifact_reason_histogram(values)
    assert hist["single_view_detached"] == 1
    assert hist["small_component"] == 1
    assert hist["excessive_dispersion"] == 1


def test_f_preview_uses_proven_concrete_canvas_and_sampling_imports():
    preview = Path(__file__).resolve().parents[2] / "scripts" / "run_f_confidence_artifact_preview.py"
    source = preview.read_text(encoding="utf-8")
    assert "from app.widgets.viewport.vispy_canvas import PointCloudCanvas" in source
    assert "from app.widgets.viewport import PointCloudCanvas" not in source
    assert "from core.geometry.sampling import enforce_point_budget" in source
    assert "from core.visualization_sampling import enforce_point_budget" not in source


def test_f_scope_authorizes_g_but_keeps_persistence_metric_measurement_and_mesh_unpromoted():
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_f_confidence_artifact_filtering.py"
    source = script.read_text(encoding="utf-8")
    assert '"confidence_artifact_filtering_formalized": True' in source
    assert '"authorizes_next_canonical_dense_persistence_milestone": passed' in source
    assert '"canonical_persistence_promoted": False' in source
    assert '"metric_scale_claimed": False' in source
    assert '"surface_mesh_generated": False' in source


def test_f_synthetic_end_to_end_exact_accounting_and_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    e3_dir = tmp_path / "e3"
    out_dir = tmp_path / "f"
    e3_dir.mkdir()
    voxel = 0.1
    # Dense 3x3x3 block, observed identically by ten independent selected views.
    raw = []
    for x in range(3):
        for y in range(3):
            for z in range(10, 13):
                raw.append([(x + 0.2) * voxel, (y + 0.2) * voxel, (z + 0.2) * voxel])
    canonical = np.asarray(raw, dtype=np.float64)
    colors = np.tile(np.asarray([[80, 120, 160]], dtype=np.uint8), (len(canonical), 1))
    depth = canonical[:, 2].astype(np.float32).reshape(1, -1)
    pose = np.eye(4, dtype=np.float64)
    keys, fused_points, fused_colors = _aggregate_saved_e1_source_frame_voxels(
        canonical_points=canonical,
        colors=colors,
        aligned_depth_reconstruction_units=depth,
        T_reconstruction_from_camera=pose,
        voxel_size=voxel,
    )
    assert len(keys) == 27

    frames = []
    orders = list(range(10))
    keyframes = [f"kf-{i}" for i in orders]
    for order in orders:
        path = e3_dir / f"frame_{order}.npz"
        np.savez_compressed(
            path,
            points_reconstruction=canonical,
            colors_rgb=colors,
            aligned_depth_reconstruction_units=depth,
            T_reconstruction_from_camera=pose,
        )
        frames.append({"workspace_order": order, "cloud_npz": str(path)})
    all_path = e3_dir / "e3_all.npz"
    np.savez_compressed(
        all_path,
        points_reconstruction=fused_points,
        colors_rgb=fused_colors,
        view_observation_counts=np.full(27, 10, dtype=np.uint32),
        voxel_size_reconstruction_units=np.asarray(voxel),
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        selected_keyframe_ids=np.asarray(keyframes, dtype="U160"),
        sparse_landmarks_reconstruction=np.empty((0, 3), dtype=np.float64),
        sparse_landmark_colors_rgb=np.empty((0, 3), dtype=np.uint8),
    )
    summary = _e3_summary()
    summary["artifacts"] = {"all_npz": str(all_path)}
    summary["fusion"].update({
        "all_voxel_count": 27,
        "supported_voxel_count": 27,
        "voxel_size_reconstruction_units": voxel,
    })
    summary["frames"] = frames
    (e3_dir / "e3_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "run_f_confidence_artifact_filtering.py",
        "--e3-dir", str(e3_dir),
        "--output-dir", str(out_dir),
        "--minimum-component-size", "1",
        "--minimum-core-voxels", "1",
        "--minimum-core-fraction", "0",
        "--minimum-extended-fraction", "0",
        "--minimum-largest-component-fraction", "0",
        "--extended-min-radius2-neighbors", "0",
    ])
    assert f_main() == 0
    payload = json.loads((out_dir / "f_summary.json").read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["observation_accounting"]["exact_match"] is True
    assert payload["classification"]["core_voxels"] > 0
    assert payload["decision"]["authorizes_next_canonical_dense_persistence_milestone"] is True
    assert (out_dir / "f_filtered_extended.npz").is_file()
