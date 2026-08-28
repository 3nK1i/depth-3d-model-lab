from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from core.reconstruction.depth_anything_v2_dense_provider import (
    DepthAnythingV2ProductionProviderAdapter,
    align_depth_anything_v2_relative_depth_to_sfm,
    depth_anything_v2_dense_provider_descriptor,
    validate_i4_3e5_binding,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3e5_helpers import (
    FakeDepthAnythingV2WorkerRunner,
    architecture_snapshot,
    d3_snapshot,
    e1_snapshot,
    e2_snapshot,
    e3_snapshot,
    e4_snapshot,
    explicit_depth_anything_v2_selection,
    good_depth_anything_v2_environment,
    make_depth_anything_v2_context,
)


def test_i43e5_descriptor_freezes_relative_depth_semantics_without_authority_promotion(tmp_path: Path) -> None:
    descriptor = depth_anything_v2_dense_provider_descriptor(
        good_depth_anything_v2_environment(tmp_path)
    )
    assert descriptor.provider_id == "depth-anything-v2-single-image"
    assert descriptor.implementation.value == "depth_anything_v2"
    assert descriptor.qualification_state.value == "candidate"
    assert descriptor.production_qualified is False
    metadata = dict(descriptor.metadata)
    assert metadata["native_depth_scale"] == "relative"
    assert metadata["native_depth_representation"] == "relative_distance_from_inverse_depth"
    assert metadata["native_intrinsics_source"] == "estimated"
    assert metadata["native_confidence_kind"] == "none"
    assert metadata["native_direct_points_available"] is False
    assert metadata["native_normals_available"] is False
    assert metadata["adapter_projected_points_discarded"] is True
    assert metadata["per_view_alignment"] == "affine_inverse_depth_to_authoritative_sfm"
    assert metadata["relative_depth_promoted_to_metric_scale"] is False
    assert metadata["estimated_model_intrinsics_consumed_for_canonical_projection"] is False
    assert metadata["permanent_global_view_cap"] is None


def test_i43e5_inverse_depth_affine_alignment_recovers_synthetic_sfm_gauge(tmp_path: Path) -> None:
    context, _ = make_depth_anything_v2_context(tmp_path, view_count=3)
    runner = FakeDepthAnythingV2WorkerRunner(context.sfm_result)
    relative = runner._relative_depth_for_keyframe("kf-0", 48, 64)
    aligned, valid, fit = align_depth_anything_v2_relative_depth_to_sfm(
        context.sfm_result,
        keyframe_id="kf-0",
        relative_depth=relative,
        source_valid_mask=np.ones_like(relative, dtype=bool),
    )
    assert fit.slope == pytest.approx(0.72, rel=0.08)
    assert fit.intercept == pytest.approx(0.08, abs=0.03)
    assert fit.inlier_correspondences >= 8
    assert fit.p90_relative_residual < 0.1
    assert aligned.shape == (48, 64)
    assert valid.shape == aligned.shape
    assert np.all(aligned[valid] > 0.0)


def test_i43e5_inverse_depth_affine_refuses_wrong_ordering(tmp_path: Path) -> None:
    context, _ = make_depth_anything_v2_context(tmp_path, view_count=3)
    runner = FakeDepthAnythingV2WorkerRunner(context.sfm_result, corrupt_alignment=True)
    relative = runner._relative_depth_for_keyframe("kf-0", 48, 64)
    with pytest.raises(ReconstructionValidationError, match="ordering|fit|correspondences"):
        align_depth_anything_v2_relative_depth_to_sfm(
            context.sfm_result,
            keyframe_id="kf-0",
            relative_depth=relative,
        )


def test_i43e5_fake_worker_execution_preserves_sfm_authority_and_all_views(tmp_path: Path) -> None:
    context, env = make_depth_anything_v2_context(tmp_path, view_count=3)
    runner = FakeDepthAnythingV2WorkerRunner(context.sfm_result)
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context, environment=env, runner=runner
    )
    request, decision = explicit_depth_anything_v2_selection(3)
    result = adapter.execute(request, decision)
    assert result.source_view_count == 3
    assert result.selected_view_count == 3
    assert result.permanent_global_view_cap is None
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.metric_scale_claimed is False
    assert result.measurement_authoritative is False
    assert result.certified_metrology is False
    assert result.surface_mesh_generated is False
    assert result.provider_promotion_performed is False
    assert Path(result.canonical_geometry_ref).is_file()
    assert runner.last_manifest["all_registered_views_selected"] is True
    assert runner.last_manifest["variant"] == "small"
    assert runner.last_manifest["encoder"] == "vits"
    assert runner.last_manifest["authority"]["estimated_model_intrinsics_are_canonical_authority"] is False
    summary = json.loads(
        (context.output_dir / "i4_3e5_depth_anything_v2_execution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["alignment"]["method"] == "affine_inverse_depth_to_authoritative_sfm"
    assert summary["authority"]["relative_depth_promoted_to_metric_scale"] is False
    assert summary["authority"]["estimated_model_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["authority"]["adapter_projected_points_consumed_for_canonical_geometry"] is False
    assert summary["integrity"]["registered_image_hashes_preserved"] is True


def test_i43e5_refuses_relative_depth_pixel_grid_drift(tmp_path: Path) -> None:
    context, env = make_depth_anything_v2_context(tmp_path, view_count=3)
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result, output_shape=(24, 32)),
    )
    request, decision = explicit_depth_anything_v2_selection(3)
    with pytest.raises(ReconstructionValidationError, match="pixel grid"):
        adapter.execute(request, decision)


def test_i43e5_refuses_source_or_checkpoint_checksum_drift(tmp_path: Path) -> None:
    for index, runner_factory in enumerate(
        (
            lambda sfm: FakeDepthAnythingV2WorkerRunner(sfm, source_revision="c" * 40),
            lambda sfm: FakeDepthAnythingV2WorkerRunner(sfm, checkpoint_sha256="d" * 64),
        )
    ):
        context, env = make_depth_anything_v2_context(tmp_path / f"case-{index}", view_count=3)
        adapter = DepthAnythingV2ProductionProviderAdapter(
            context=context, environment=env, runner=runner_factory(context.sfm_result)
        )
        request, decision = explicit_depth_anything_v2_selection(3)
        with pytest.raises(ReconstructionValidationError, match="revision|checksum"):
            adapter.execute(request, decision)


def test_i43e5_refuses_depth_anything_semantics_drift_or_fabricated_native_outputs(tmp_path: Path) -> None:
    cases = (
        ({"native_depth_scale": "metric", "native_is_metric": True}, "relative-depth"),
        ({"depth_representation_native": "raw_metric_depth"}, "inverse-depth representation"),
        ({"intrinsics_source_native": "model_predicted"}, "estimated-intrinsics"),
        ({"confidence_kind_native": "confidence"}, "no-confidence"),
        ({"direct_points_native": True}, "fabricated"),
        ({"normals_native": True}, "fabricated"),
        ({"adapter_projected_points_discarded": False}, "discard"),
        ({"promote_native_authority": True}, "promote"),
    )
    for index, (kwargs, message) in enumerate(cases):
        context, env = make_depth_anything_v2_context(tmp_path / f"sem-{index}", view_count=3)
        adapter = DepthAnythingV2ProductionProviderAdapter(
            context=context,
            environment=env,
            runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result, **kwargs),
        )
        request, decision = explicit_depth_anything_v2_selection(3)
        with pytest.raises(ReconstructionValidationError, match=message):
            adapter.execute(request, decision)


def test_i43e5_refuses_registered_image_mutation(tmp_path: Path) -> None:
    context, env = make_depth_anything_v2_context(tmp_path, view_count=3)
    workspace_root = Path(context.sfm_result.workspace.workspace_root)
    first = sorted(context.sfm_result.workspace.images, key=lambda item: item.order)[0]
    mutate_path = workspace_root / first.staged_image_path
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result, mutate_path=mutate_path),
    )
    request, decision = explicit_depth_anything_v2_selection(3)
    with pytest.raises(ReconstructionValidationError, match="changed during provider execution"):
        adapter.execute(request, decision)


def test_i43e5_refuses_multi_camera_run_until_explicit_camera_handling_exists(tmp_path: Path) -> None:
    context, env = make_depth_anything_v2_context(
        tmp_path,
        view_count=3,
        camera_ids=("cam-a", "cam-b", "cam-a"),
    )
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result),
    )
    request, decision = explicit_depth_anything_v2_selection(3)
    with pytest.raises(ReconstructionValidationError, match="one authoritative SfM camera model"):
        adapter.execute(request, decision)


def test_i43e5_refuses_unavailable_runtime_wrong_provider_and_fallback(tmp_path: Path) -> None:
    context, _ = make_depth_anything_v2_context(tmp_path / "unavailable", view_count=3)
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context,
        environment=good_depth_anything_v2_environment(tmp_path / "unavailable", available=False),
        runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result),
    )
    request, decision = explicit_depth_anything_v2_selection(3)
    with pytest.raises(ReconstructionValidationError, match="runtime is unavailable"):
        adapter.execute(request, decision)

    context, env = make_depth_anything_v2_context(tmp_path / "wrong", view_count=3)
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result),
    )
    request, decision = explicit_depth_anything_v2_selection(3, provider_id="wrong-provider")
    with pytest.raises(ReconstructionValidationError, match="does not match"):
        adapter.execute(request, decision)

    context, env = make_depth_anything_v2_context(tmp_path / "fallback", view_count=3)
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result),
    )
    request, decision = explicit_depth_anything_v2_selection(3, allow_fallback=True)
    with pytest.raises(ReconstructionValidationError, match="forbids silent provider fallback"):
        adapter.execute(request, decision)


def test_i43e5_binding_requires_closed_predecessors_and_preserves_relative_authority(tmp_path: Path) -> None:
    snapshot = validate_i4_3e5_binding(
        i4_3a_snapshot=architecture_snapshot(),
        i4_3d3_snapshot=d3_snapshot(),
        i4_3e1_snapshot=e1_snapshot(),
        i4_3e2_snapshot=e2_snapshot(),
        i4_3e3_snapshot=e3_snapshot(),
        i4_3e4_snapshot=e4_snapshot(),
        environment=good_depth_anything_v2_environment(tmp_path),
    )
    assert snapshot["pass"] is True
    assert snapshot["provider"]["implementation"] == "depth_anything_v2"
    assert snapshot["provider"]["qualification_state"] == "candidate"
    assert snapshot["architecture"]["native_depth_scale"] == "relative"
    assert snapshot["architecture"]["native_confidence_kind"] == "none"
    assert snapshot["architecture"]["per_view_alignment"] == "affine_inverse_depth_to_authoritative_sfm"
    assert snapshot["authority"]["relative_depth_promoted_to_metric_scale"] is False
    assert snapshot["authority"]["estimated_model_intrinsics_consumed_for_canonical_projection"] is False
    assert snapshot["authority"]["adapter_projected_points_consumed_for_canonical_geometry"] is False
    assert snapshot["qualification"]["predecessor_i4_3e4_verified"] is True
    assert snapshot["qualification"]["remaining_i4_3e_providers_pending"] == ["marigold"]


def test_i43e5_binding_refuses_promoted_e4_predecessor_authority(tmp_path: Path) -> None:
    bad = e4_snapshot()
    bad["authority"]["measurement_authoritative"] = True
    with pytest.raises(ReconstructionValidationError, match="I4.3E4 predecessor"):
        validate_i4_3e5_binding(
            i4_3a_snapshot=architecture_snapshot(),
            i4_3d3_snapshot=d3_snapshot(),
            i4_3e1_snapshot=e1_snapshot(),
            i4_3e2_snapshot=e2_snapshot(),
            i4_3e3_snapshot=e3_snapshot(),
            i4_3e4_snapshot=bad,
            environment=good_depth_anything_v2_environment(tmp_path),
        )


def test_i43e5_worker_import_is_lazy() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = root / "scripts/reconstruction_depth_anything_v2_provider_worker.py"
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("i43e5_worker_lazy_probe", worker)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new_modules = set(sys.modules) - before
    for forbidden in ("torch", "cv2", "depth_anything_v2", "core.models.registry"):
        assert forbidden not in new_modules
    assert callable(module.run)
