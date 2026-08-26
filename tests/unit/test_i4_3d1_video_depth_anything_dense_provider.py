from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderQualificationState,
    DenseProviderRunStatus,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.video_depth_anything_dense_provider import (
    I43D1_PROVIDER_ID,
    VideoDepthAnythingEnvironmentProbe,
    VideoDepthAnythingProductionProviderAdapter,
    validate_i4_3d1_binding,
    video_depth_anything_dense_provider_descriptor,
)
from tests.i4_3d1_helpers import (
    CollectingProgressSink,
    FakeVDAWorkerRunner,
    explicit_vda_selection,
    good_vda_environment,
    make_vda_context,
    scrambled_temporal_metadata_sfm_result,
)


def _architecture_snapshot() -> dict:
    return {
        "phase": "I4.3A",
        "status": "pass",
        "pass": True,
        "architecture": {
            "authoritative_camera_frame_owner": "pycolmap_sfm",
            "permanent_global_view_cap": None,
            "provider_implementation_contracts": [
                {
                    "implementation": "video_depth_anything",
                    "family": "temporal_depth",
                    "integration_stage": "I4.3D",
                    "native_output": "depth_evidence",
                    "frame_binding": "align_to_authoritative_sfm",
                    "existing_adapter_expected": True,
                }
            ],
        },
    }


def test_i43d1_descriptor_is_temporal_candidate_without_authority(tmp_path: Path) -> None:
    descriptor = video_depth_anything_dense_provider_descriptor(good_vda_environment(tmp_path))
    assert descriptor.provider_id == I43D1_PROVIDER_ID
    assert descriptor.implementation is DenseProviderImplementation.VIDEO_DEPTH_ANYTHING
    assert descriptor.family is DenseProviderFamily.TEMPORAL_DEPTH
    assert descriptor.qualification_state is DenseProviderQualificationState.CANDIDATE
    assert descriptor.available is True
    assert descriptor.production_qualified is False
    data = descriptor.as_dict()
    assert data["metadata"]["permanent_global_view_cap"] is None
    assert data["metadata"]["model_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert data["metadata"]["estimated_model_intrinsics_consumed_for_canonical_projection"] is False


def test_i43d1_unavailable_runtime_has_explicit_reason(tmp_path: Path) -> None:
    descriptor = video_depth_anything_dense_provider_descriptor(good_vda_environment(tmp_path, available=False))
    assert descriptor.available is False
    assert descriptor.unavailable_reason == "not_ready"


def test_i43d1_environment_revision_contract_is_strict(tmp_path: Path) -> None:
    env = good_vda_environment(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="source revision"):
        replace(env, source_revision="short")


def test_i43d1_execute_uses_all_views_and_preserves_authoritative_frame(tmp_path: Path) -> None:
    context, env = make_vda_context(tmp_path, view_count=3)
    runner = FakeVDAWorkerRunner()
    sink = CollectingProgressSink()
    adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=env, runner=runner, progress_sink=sink)
    request, decision = explicit_vda_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    assert result.source_view_count == result.selected_view_count == 3
    assert result.adaptive_view_selection is False
    assert result.permanent_global_view_cap is None
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.metric_scale_claimed is False
    assert result.provider_promotion_performed is False
    assert runner.calls == 1
    assert [item["source_sequence_index"] for item in runner.last_manifest["sequence"]] == [0, 1, 2]
    assert len(sink.events) == result.progress_event_count
    assert sink.events[-1].progress_fraction == 1.0


def test_i43d1_execute_reorders_workspace_back_to_source_temporal_order(tmp_path: Path) -> None:
    context, env = make_vda_context(tmp_path, view_count=3)
    context = replace(context, sfm_result=scrambled_temporal_metadata_sfm_result(context.sfm_result))
    runner = FakeVDAWorkerRunner()
    adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=env, runner=runner)
    request, decision = explicit_vda_selection(3)
    adapter.execute(request, decision)
    assert [item["source_sequence_index"] for item in runner.last_manifest["sequence"]] == [0, 1, 2]


def test_i43d1_execute_rejects_duplicate_temporal_sequence_indices(tmp_path: Path) -> None:
    context, env = make_vda_context(tmp_path, view_count=3)
    images = list(context.sfm_result.workspace.images)
    images[1] = replace(images[1], source_sequence_index=images[0].source_sequence_index)
    sfm = replace(context.sfm_result, workspace=replace(context.sfm_result.workspace, images=tuple(images)))
    adapter = VideoDepthAnythingProductionProviderAdapter(context=replace(context, sfm_result=sfm), environment=env, runner=FakeVDAWorkerRunner())
    request, decision = explicit_vda_selection(3)
    with pytest.raises(ReconstructionValidationError, match="unique source_sequence_index"):
        adapter.execute(request, decision)


def test_i43d1_execute_refuses_worker_pixel_grid_drift(tmp_path: Path) -> None:
    context, env = make_vda_context(tmp_path, view_count=3)
    adapter = VideoDepthAnythingProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeVDAWorkerRunner(output_shape=(24, 32)),
    )
    request, decision = explicit_vda_selection(3)
    with pytest.raises(ReconstructionValidationError, match="authoritative SfM pixel grid"):
        adapter.execute(request, decision)


def test_i43d1_execute_refuses_runtime_revision_drift(tmp_path: Path) -> None:
    context, env = make_vda_context(tmp_path, view_count=3)
    adapter = VideoDepthAnythingProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeVDAWorkerRunner(source_revision="c" * 40),
    )
    request, decision = explicit_vda_selection(3)
    with pytest.raises(ReconstructionValidationError, match="source revision"):
        adapter.execute(request, decision)


def test_i43d1_execute_refuses_source_image_mutation(tmp_path: Path) -> None:
    context, env = make_vda_context(tmp_path, view_count=3)
    first = Path(context.sfm_result.workspace.workspace_root) / context.sfm_result.workspace.images[0].staged_image_path
    adapter = VideoDepthAnythingProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeVDAWorkerRunner(mutate_path=first),
    )
    request, decision = explicit_vda_selection(3)
    with pytest.raises(ReconstructionValidationError, match="registered staged images changed"):
        adapter.execute(request, decision)


def test_i43d1_execute_refuses_unavailable_runtime(tmp_path: Path) -> None:
    context, _ = make_vda_context(tmp_path, view_count=3)
    adapter = VideoDepthAnythingProductionProviderAdapter(
        context=context,
        environment=good_vda_environment(tmp_path, available=False),
        runner=FakeVDAWorkerRunner(),
    )
    request, decision = explicit_vda_selection(3)
    with pytest.raises(ReconstructionValidationError, match="runtime is unavailable"):
        adapter.execute(request, decision)


def test_i43d1_execute_refuses_wrong_concrete_selection(tmp_path: Path) -> None:
    context, env = make_vda_context(tmp_path, view_count=3)
    adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=env, runner=FakeVDAWorkerRunner())
    request, decision = explicit_vda_selection(3, provider_id="wrong-provider")
    with pytest.raises(ReconstructionValidationError, match="concrete provider ID"):
        adapter.execute(request, decision)


def test_i43d1_summary_records_native_metric_as_non_authoritative(tmp_path: Path) -> None:
    context, env = make_vda_context(tmp_path, view_count=3)
    adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=env, runner=FakeVDAWorkerRunner())
    request, decision = explicit_vda_selection(3)
    adapter.execute(request, decision)
    summary = json.loads((context.output_dir / "i4_3d1_vda_execution_summary.json").read_text(encoding="utf-8"))
    assert summary["authority"]["model_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert summary["authority"]["estimated_model_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["integrity"]["canonical_pose_layer"] == "offline_sfm_bundle_adjusted"
    assert summary["temporal_sequence"]["ordered_source_sequence_indices"] == [0, 1, 2]


def test_i43d1_binding_passes_and_keeps_rollingdepth_pending(tmp_path: Path) -> None:
    snapshot = validate_i4_3d1_binding(i4_3a_snapshot=_architecture_snapshot(), environment=good_vda_environment(tmp_path))
    assert snapshot["status"] == "pass"
    assert snapshot["provider"]["implementation"] == "video_depth_anything"
    assert snapshot["architecture"]["stateful_sequence"] is True
    assert snapshot["qualification"]["rolling_depth_binding_pending_i4_3d2"] is True
    assert snapshot["authority"]["model_metric_depth_promoted_to_canonical_metric_scale"] is False


def test_i43d1_binding_fails_closed_on_authority_drift(tmp_path: Path) -> None:
    snapshot = _architecture_snapshot()
    snapshot["architecture"]["authoritative_camera_frame_owner"] = "video_depth_anything"
    with pytest.raises(ReconstructionValidationError, match="camera-frame authority"):
        validate_i4_3d1_binding(i4_3a_snapshot=snapshot, environment=good_vda_environment(tmp_path))


def test_i43d1_binding_fails_closed_on_stage_drift(tmp_path: Path) -> None:
    snapshot = _architecture_snapshot()
    snapshot["architecture"]["provider_implementation_contracts"][0]["integration_stage"] = "I4.3E"
    with pytest.raises(ReconstructionValidationError, match="assigned to another stage"):
        validate_i4_3d1_binding(i4_3a_snapshot=snapshot, environment=good_vda_environment(tmp_path))
