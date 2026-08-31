from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.metric3d_v2_dense_provider import (
    Metric3DV2ProductionProviderAdapter,
    metric3d_v2_dense_provider_descriptor,
    validate_i4_3e1_binding,
)
from tests.i4_3e1_helpers import (
    CollectingProgressSink,
    FakeMetric3DWorkerRunner,
    architecture_snapshot,
    d3_snapshot,
    explicit_metric3d_selection,
    good_metric3d_environment,
    make_metric3d_context,
)


def test_i43e1_descriptor_preserves_single_image_authority_locks(tmp_path: Path) -> None:
    descriptor = metric3d_v2_dense_provider_descriptor(good_metric3d_environment(tmp_path))
    payload = descriptor.as_dict()
    assert descriptor.available is True
    assert payload["implementation"] == "metric3d_v2"
    assert payload["family"] == "single_image_depth"
    assert payload["qualification_state"] == "candidate"
    assert payload["metadata"]["native_depth_scale"] == "metric"
    assert payload["metadata"]["native_intrinsics_source"] == "estimated"
    assert payload["metadata"]["model_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert payload["metadata"]["model_estimated_intrinsics_consumed_for_canonical_projection"] is False
    assert payload["metadata"]["permanent_global_view_cap"] is None


def test_i43e1_execute_metric3d_candidate_preserves_authoritative_sfm(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    runner = FakeMetric3DWorkerRunner()
    sink = CollectingProgressSink()
    adapter = Metric3DV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=runner,
        progress_sink=sink,
    )
    request, decision = explicit_metric3d_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    assert result.source_view_count == result.selected_view_count == 3
    assert result.adaptive_view_selection is False
    assert result.permanent_global_view_cap is None
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.metric_scale_claimed is False
    assert result.measurement_authoritative is False
    assert result.provider_promotion_performed is False
    assert runner.calls == 1
    assert runner.last_manifest["authority"]["model_metric_depth_establishes_metric_scale"] is False
    assert runner.last_manifest["authority"]["model_estimated_intrinsics_are_canonical_authority"] is False
    assert len(sink.events) == result.progress_event_count
    assert sink.events[-1].progress_fraction == 1.0


def test_i43e1_execute_refuses_multiple_sfm_camera_models(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3, camera_ids=("camera-0", "camera-1", "camera-0"))
    adapter = Metric3DV2ProductionProviderAdapter(context=context, environment=env, runner=FakeMetric3DWorkerRunner())
    request, decision = explicit_metric3d_selection(3)
    with pytest.raises(ReconstructionValidationError, match="one authoritative SfM camera model"):
        adapter.execute(request, decision)


def test_i43e1_execute_refuses_worker_pixel_grid_drift(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMetric3DWorkerRunner(output_shape=(24, 32)),
    )
    request, decision = explicit_metric3d_selection(3)
    with pytest.raises(ReconstructionValidationError, match="authoritative SfM pixel grid"):
        adapter.execute(request, decision)


def test_i43e1_execute_refuses_native_semantics_drift(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMetric3DWorkerRunner(native_depth_scale="relative", native_is_metric=False),
    )
    request, decision = explicit_metric3d_selection(3)
    with pytest.raises(ReconstructionValidationError, match="native metric-depth semantics"):
        adapter.execute(request, decision)


def test_i43e1_execute_refuses_runtime_revision_drift(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMetric3DWorkerRunner(source_revision="f" * 40),
    )
    request, decision = explicit_metric3d_selection(3)
    with pytest.raises(ReconstructionValidationError, match="source revision"):
        adapter.execute(request, decision)


def test_i43e1_execute_refuses_source_image_mutation(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    first = Path(context.sfm_result.workspace.workspace_root) / context.sfm_result.workspace.images[0].staged_image_path
    adapter = Metric3DV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMetric3DWorkerRunner(mutate_path=first),
    )
    request, decision = explicit_metric3d_selection(3)
    with pytest.raises(ReconstructionValidationError, match="registered staged images changed"):
        adapter.execute(request, decision)


def test_i43e1_execute_refuses_unavailable_runtime(tmp_path: Path) -> None:
    context, _ = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(
        context=context,
        environment=good_metric3d_environment(tmp_path / "unavailable", available=False),
        runner=FakeMetric3DWorkerRunner(),
    )
    request, decision = explicit_metric3d_selection(3)
    with pytest.raises(ReconstructionValidationError, match="runtime is unavailable"):
        adapter.execute(request, decision)


def test_i43e1_execute_refuses_wrong_concrete_selection(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(context=context, environment=env, runner=FakeMetric3DWorkerRunner())
    request, decision = explicit_metric3d_selection(3, provider_id="wrong-provider")
    with pytest.raises(ReconstructionValidationError, match="provider ID"):
        adapter.execute(request, decision)


def test_i43e1_execute_refuses_explicit_fallback_permission(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(context=context, environment=env, runner=FakeMetric3DWorkerRunner())
    request, decision = explicit_metric3d_selection(3, allow_fallback=True)
    with pytest.raises(ReconstructionValidationError, match="silent provider fallback"):
        adapter.execute(request, decision)


def test_i43e1_summary_records_metric_and_intrinsics_as_non_authoritative(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(context=context, environment=env, runner=FakeMetric3DWorkerRunner())
    request, decision = explicit_metric3d_selection(3)
    adapter.execute(request, decision)
    summary = json.loads((context.output_dir / "i4_3e1_metric3d_v2_execution_summary.json").read_text(encoding="utf-8"))
    assert summary["authority"]["native_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert summary["authority"]["estimated_model_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["authority"]["metric_scale_claimed"] is False
    assert summary["integrity"]["canonical_pose_layer"] == "offline_sfm_bundle_adjusted"
    assert summary["alignment"]["accepted_views"] == 3


def test_i43e1_binding_passes_only_after_d3_closure(tmp_path: Path) -> None:
    snapshot = validate_i4_3e1_binding(
        i4_3a_snapshot=architecture_snapshot(),
        i4_3d3_snapshot=d3_snapshot(),
        environment=good_metric3d_environment(tmp_path),
    )
    assert snapshot["status"] == "pass"
    assert snapshot["provider"]["implementation"] == "metric3d_v2"
    assert snapshot["architecture"]["family"] == "single_image_depth"
    assert snapshot["architecture"]["native_confidence_range"] == "unbounded_score"
    assert snapshot["architecture"]["native_confidence_consumed_as_point_confidence"] is False
    assert snapshot["architecture"]["native_confidence_normalization_invented"] is False
    assert snapshot["authority"]["metric_scale_claimed"] is False
    assert snapshot["qualification"]["state"] == "candidate"
    assert "unidepth_v2" in snapshot["qualification"]["remaining_i4_3e_providers_pending"]


def test_i43e1_binding_fails_closed_without_d3_authorization(tmp_path: Path) -> None:
    d3 = d3_snapshot()
    d3["decision"]["i4_3e_authorized"] = False
    with pytest.raises(ReconstructionValidationError, match="not authorized"):
        validate_i4_3e1_binding(
            i4_3a_snapshot=architecture_snapshot(),
            i4_3d3_snapshot=d3,
            environment=good_metric3d_environment(tmp_path),
        )


def test_i43e1_binding_fails_closed_on_prior_authority_promotion(tmp_path: Path) -> None:
    d3 = d3_snapshot()
    d3["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="promoted authority"):
        validate_i4_3e1_binding(
            i4_3a_snapshot=architecture_snapshot(),
            i4_3d3_snapshot=d3,
            environment=good_metric3d_environment(tmp_path),
        )


def test_i43e1_worker_import_is_model_and_cuda_lazy() -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/reconstruction_single_image_depth_provider_worker.py"
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("i43e1_worker_probe", worker)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new_modules = set(sys.modules) - before
    for forbidden in ("torch", "cv2", "core.models.registry"):
        assert forbidden not in new_modules
    assert callable(module.main) and callable(module.run)
    assert callable(module._native_confidence_stats)


def test_i43e1_worker_accepts_finite_unbounded_native_confidence() -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/reconstruction_single_image_depth_provider_worker.py"
    spec = importlib.util.spec_from_file_location("i43e1_worker_confidence_probe", worker)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    low, high = module._native_confidence_stats(
        np.full((2, 3), 2.5, dtype=np.float32), expected_hw=(2, 3)
    )
    assert low == pytest.approx(2.5)
    assert high == pytest.approx(2.5)


def test_i43e1_worker_rejects_nonfinite_native_confidence() -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/reconstruction_single_image_depth_provider_worker.py"
    spec = importlib.util.spec_from_file_location("i43e1_worker_nonfinite_confidence_probe", worker)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="native confidence must be finite"):
        module._native_confidence_stats(
            np.array([[1.0, np.nan]], dtype=np.float32), expected_hw=(1, 2)
        )


def test_i43e1_preserves_unbounded_native_confidence_without_using_it_as_fusion_weight(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMetric3DWorkerRunner(confidence_value=2.5),
    )
    request, decision = explicit_metric3d_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED

    summary = json.loads((context.output_dir / "i4_3e1_metric3d_v2_execution_summary.json").read_text(encoding="utf-8"))
    assert summary["authority"]["native_confidence_consumed_as_point_confidence"] is False
    assert summary["authority"]["native_confidence_normalization_invented"] is False
    for record in summary["alignment"]["records"]:
        assert record["native_confidence_consumed_as_point_confidence"] is False
        assert record["native_confidence_range"] == "unbounded_score"
        assert record["native_confidence_min"] == pytest.approx(2.5)
        assert record["native_confidence_max"] == pytest.approx(2.5)

    with np.load(result.canonical_geometry_ref, allow_pickle=False) as archive:
        # Canonical fusion confidence remains valid [0,1] neutral weighting and
        # does not reinterpret the native Metric3D score as calibrated confidence.
        assert np.isfinite(archive["mean_confidence"]).all()
        assert np.allclose(archive["mean_confidence"], 1.0)


def test_i43e1_still_rejects_nonfinite_native_confidence(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=3)
    adapter = Metric3DV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMetric3DWorkerRunner(confidence_value=float("nan")),
    )
    request, decision = explicit_metric3d_selection(3)
    with pytest.raises(ReconstructionValidationError, match="native confidence must be finite"):
        adapter.execute(request, decision)
