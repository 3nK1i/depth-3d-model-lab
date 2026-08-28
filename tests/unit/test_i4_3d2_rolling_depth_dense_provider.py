from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderQualificationState,
    DenseProviderRunStatus,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.rolling_depth_dense_provider import (
    I43D2_PROVIDER_ID,
    RollingDepthEnvironmentProbe,
    RollingDepthProductionProviderAdapter,
    fit_rolling_depth_sequence_to_sfm,
    rolling_depth_dense_provider_descriptor,
    validate_i4_3d2_binding,
)
from tests.i4_3d2_helpers import (
    CollectingProgressSink,
    FakeRollingDepthSequenceBuilder,
    FakeRollingDepthWorkerRunner,
    explicit_rolling_selection,
    good_rolling_environment,
    make_rolling_context,
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
                    "implementation": "rolling_depth",
                    "family": "temporal_depth",
                    "integration_stage": "I4.3D",
                    "native_output": "depth_evidence",
                    "frame_binding": "align_to_authoritative_sfm",
                    "existing_adapter_expected": False,
                }
            ],
        },
    }


def _d1_snapshot() -> dict:
    return {
        "phase": "I4.3D1",
        "status": "pass",
        "pass": True,
        "provider": {"implementation": "video_depth_anything"},
        "authority": {
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def _adapter(tmp_path: Path, *, view_count: int = 3, runner=None, builder=None, environment=None):
    context, env = make_rolling_context(tmp_path, view_count=view_count)
    return (
        context,
        RollingDepthProductionProviderAdapter(
            context=context,
            environment=environment or env,
            runner=runner or FakeRollingDepthWorkerRunner(context.sfm_result),
            sequence_builder=builder or FakeRollingDepthSequenceBuilder(),
        ),
    )


def test_i43d2_descriptor_is_temporal_candidate_without_authority(tmp_path: Path) -> None:
    descriptor = rolling_depth_dense_provider_descriptor(good_rolling_environment(tmp_path))
    assert descriptor.provider_id == I43D2_PROVIDER_ID
    assert descriptor.implementation is DenseProviderImplementation.ROLLING_DEPTH
    assert descriptor.family is DenseProviderFamily.TEMPORAL_DEPTH
    assert descriptor.qualification_state is DenseProviderQualificationState.CANDIDATE
    assert descriptor.available is True
    assert descriptor.production_qualified is False
    data = descriptor.as_dict()
    assert data["metadata"]["permanent_global_view_cap"] is None
    assert data["metadata"]["sequence_level_alignment"] == "global_affine_to_authoritative_sfm"
    assert data["metadata"]["per_frame_alignment"] is False
    assert data["metadata"]["relative_depth_affine_not_metric"] is True


def test_i43d2_unavailable_runtime_has_explicit_reason(tmp_path: Path) -> None:
    descriptor = rolling_depth_dense_provider_descriptor(good_rolling_environment(tmp_path, available=False))
    assert descriptor.available is False
    assert descriptor.unavailable_reason == "not_ready"


def test_i43d2_environment_revision_contract_is_strict(tmp_path: Path) -> None:
    env = good_rolling_environment(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="source revision"):
        replace(env, source_revision="short")


def test_i43d2_global_affine_fit_recovers_single_sequence_mapping(tmp_path: Path) -> None:
    context, _ = make_rolling_context(tmp_path, view_count=3)
    runner = FakeRollingDepthWorkerRunner(context.sfm_result, scale=1.6, shift=0.35)
    builder = FakeRollingDepthSequenceBuilder()
    # Let the fake runner build its exact sparse-correspondence maps from the manifest.
    context.output_dir.mkdir(parents=True, exist_ok=True)
    context.output_dir.rmdir()
    adapter = RollingDepthProductionProviderAdapter(
        context=context,
        environment=good_rolling_environment(tmp_path),
        runner=runner,
        sequence_builder=builder,
    )
    request, decision = explicit_rolling_selection(3)
    adapter.execute(request, decision)
    summary = json.loads((context.output_dir / "i4_3d2_rolling_depth_execution_summary.json").read_text(encoding="utf-8"))
    alignment = summary["alignment"]
    assert alignment["accepted"] is True
    assert alignment["scale_to_reconstruction"] == pytest.approx(1.6, rel=2e-3)
    assert alignment["shift_to_reconstruction"] == pytest.approx(0.35, abs=5e-3)
    assert alignment["alignment_scope"] == "one_global_affine_for_entire_temporal_sequence"
    assert alignment["physical_metric_scale_claimed"] is False


def test_i43d2_execute_uses_all_views_and_preserves_authoritative_frame(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    runner = FakeRollingDepthWorkerRunner(context.sfm_result)
    builder = FakeRollingDepthSequenceBuilder()
    sink = CollectingProgressSink()
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env, runner=runner, sequence_builder=builder, progress_sink=sink
    )
    request, decision = explicit_rolling_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    assert result.source_view_count == result.selected_view_count == 3
    assert result.adaptive_view_selection is False
    assert result.permanent_global_view_cap is None
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.metric_scale_claimed is False
    assert result.provider_promotion_performed is False
    assert runner.calls == builder.calls == 1
    assert len(sink.events) == result.progress_event_count
    assert sink.events[-1].progress_fraction == 1.0


def test_i43d2_execute_reorders_workspace_back_to_source_temporal_order(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    sfm = scrambled_temporal_metadata_sfm_result(context.sfm_result)
    context = replace(context, sfm_result=sfm)
    runner = FakeRollingDepthWorkerRunner(sfm)
    builder = FakeRollingDepthSequenceBuilder()
    adapter = RollingDepthProductionProviderAdapter(context=context, environment=env, runner=runner, sequence_builder=builder)
    request, decision = explicit_rolling_selection(3)
    adapter.execute(request, decision)
    assert builder.last_source_indices == [0, 1, 2]
    assert [item["source_sequence_index"] for item in runner.last_manifest["sequence"]] == [0, 1, 2]


def test_i43d2_execute_rejects_duplicate_temporal_sequence_indices(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    images = list(context.sfm_result.workspace.images)
    images[1] = replace(images[1], source_sequence_index=images[0].source_sequence_index)
    sfm = replace(context.sfm_result, workspace=replace(context.sfm_result.workspace, images=tuple(images)))
    adapter = RollingDepthProductionProviderAdapter(
        context=replace(context, sfm_result=sfm), environment=env,
        runner=FakeRollingDepthWorkerRunner(sfm), sequence_builder=FakeRollingDepthSequenceBuilder()
    )
    request, decision = explicit_rolling_selection(3)
    with pytest.raises(ReconstructionValidationError, match="unique source_sequence_index"):
        adapter.execute(request, decision)


def test_i43d2_execute_refuses_worker_pixel_grid_drift(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env,
        runner=FakeRollingDepthWorkerRunner(context.sfm_result, output_shape=(24, 32)),
        sequence_builder=FakeRollingDepthSequenceBuilder(),
    )
    request, decision = explicit_rolling_selection(3)
    with pytest.raises(ReconstructionValidationError, match="authoritative SfM pixel grid"):
        adapter.execute(request, decision)


def test_i43d2_execute_refuses_runtime_revision_drift(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env,
        runner=FakeRollingDepthWorkerRunner(context.sfm_result, source_revision="c" * 40),
        sequence_builder=FakeRollingDepthSequenceBuilder(),
    )
    request, decision = explicit_rolling_selection(3)
    with pytest.raises(ReconstructionValidationError, match="source revision"):
        adapter.execute(request, decision)


def test_i43d2_execute_refuses_worker_per_frame_alignment(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env,
        runner=FakeRollingDepthWorkerRunner(context.sfm_result, per_frame_alignment=True),
        sequence_builder=FakeRollingDepthSequenceBuilder(),
    )
    request, decision = explicit_rolling_selection(3)
    with pytest.raises(ReconstructionValidationError, match="per-frame alignment"):
        adapter.execute(request, decision)


def test_i43d2_execute_refuses_source_image_mutation(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    first = Path(context.sfm_result.workspace.workspace_root) / context.sfm_result.workspace.images[0].staged_image_path
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env,
        runner=FakeRollingDepthWorkerRunner(context.sfm_result, mutate_path=first),
        sequence_builder=FakeRollingDepthSequenceBuilder(),
    )
    request, decision = explicit_rolling_selection(3)
    with pytest.raises(ReconstructionValidationError, match="registered staged images changed"):
        adapter.execute(request, decision)


def test_i43d2_execute_refuses_unavailable_runtime(tmp_path: Path) -> None:
    context, _ = make_rolling_context(tmp_path, view_count=3)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=good_rolling_environment(tmp_path, available=False),
        runner=FakeRollingDepthWorkerRunner(context.sfm_result), sequence_builder=FakeRollingDepthSequenceBuilder(),
    )
    request, decision = explicit_rolling_selection(3)
    with pytest.raises(ReconstructionValidationError, match="runtime is unavailable"):
        adapter.execute(request, decision)


def test_i43d2_execute_refuses_wrong_concrete_selection(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env,
        runner=FakeRollingDepthWorkerRunner(context.sfm_result), sequence_builder=FakeRollingDepthSequenceBuilder(),
    )
    request, decision = explicit_rolling_selection(3, provider_id="wrong-provider")
    with pytest.raises(ReconstructionValidationError, match="concrete provider ID"):
        adapter.execute(request, decision)


def test_i43d2_summary_records_global_affine_as_non_metric(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env,
        runner=FakeRollingDepthWorkerRunner(context.sfm_result), sequence_builder=FakeRollingDepthSequenceBuilder(),
    )
    request, decision = explicit_rolling_selection(3)
    adapter.execute(request, decision)
    summary = json.loads((context.output_dir / "i4_3d2_rolling_depth_execution_summary.json").read_text(encoding="utf-8"))
    assert summary["authority"]["sequence_affine_establishes_physical_metric_scale"] is False
    assert summary["authority"]["per_frame_alignment_performed"] is False
    assert summary["integrity"]["canonical_pose_layer"] == "offline_sfm_bundle_adjusted"
    assert summary["temporal_sequence"]["native_temporal_coalignment_preserved"] is True


def test_i43d2_binding_passes_and_retains_vda(tmp_path: Path) -> None:
    snapshot = validate_i4_3d2_binding(
        i4_3a_snapshot=_architecture_snapshot(),
        i4_3d1_snapshot=_d1_snapshot(),
        environment=good_rolling_environment(tmp_path),
    )
    assert snapshot["status"] == "pass"
    assert snapshot["provider"]["implementation"] == "rolling_depth"
    assert snapshot["architecture"]["per_frame_alignment"] is False
    assert snapshot["qualification"]["video_depth_anything_binding_retained"] is True
    assert snapshot["qualification"]["temporal_provider_closure_pending_i4_3d3"] is True
    assert snapshot["authority"]["sequence_affine_establishes_physical_metric_scale"] is False


def test_i43d2_binding_fails_closed_on_authority_drift(tmp_path: Path) -> None:
    snapshot = _architecture_snapshot()
    snapshot["architecture"]["authoritative_camera_frame_owner"] = "rolling_depth"
    with pytest.raises(ReconstructionValidationError, match="camera-frame authority"):
        validate_i4_3d2_binding(
            i4_3a_snapshot=snapshot, i4_3d1_snapshot=_d1_snapshot(), environment=good_rolling_environment(tmp_path)
        )


def test_i43d2_binding_fails_closed_if_d1_metric_authority_drifted(tmp_path: Path) -> None:
    d1 = _d1_snapshot()
    d1["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="changed authority limits"):
        validate_i4_3d2_binding(
            i4_3a_snapshot=_architecture_snapshot(), i4_3d1_snapshot=d1, environment=good_rolling_environment(tmp_path)
        )


def test_i43d2_setup_normalizes_upstream_pyav_name_without_user_site_leakage() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = (project_root / "scripts" / "setup_rolling_depth_backend.sh").read_text(encoding="utf-8")
    assert "RollingDepth dependency normalization: pyav->av" in script
    assert 'r"^(\\s*)pyav(?=(?:[<>=!~]|\\s|$))"' in script
    assert 'resolve_env_python()' in script
    assert "print(os.path.realpath(sys.executable))" in script
    assert 'conda run -n "$ENV_NAME" which python' not in script
    assert 'PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m pip install -r "$NORMALIZED_REQUIREMENTS"' in script
    assert 'python -m pip install -r "$REPO/requirements.txt"' not in script
    assert 'RollingDepth checkpoint already present; skipping weight download' in script
    assert 'CHECKPOINT_MARKER="$CHECKPOINT_DIR/model_index.json"' in script
    assert '--finalize-existing' in script
    assert 'RollingDepth finalize-existing mode: skipping dependency reinstall' in script
    assert 'RollingDepth finalize-existing mode: reusing pinned checkout revision=' in script
    assert 'PROBE_FILE="$(mktemp)"' in script
    assert 'probe_path.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")' in script
    assert 'PROBE_JSON="$(cat "$PROBE_FILE")"' in script
    assert 'PROBE_JSON="$(PYTHONNOUSERSITE=1' not in script
