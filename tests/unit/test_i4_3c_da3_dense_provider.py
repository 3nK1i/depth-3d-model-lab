from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.da3 import DA3_PROVIDER_ID
from core.reconstruction.da3_dense_provider import (
    DA3ProductionConfig,
    DA3ProductionProviderAdapter,
    DA3ProviderRunContext,
    _adapt_grid_to_sfm,
    da3_dense_provider_descriptor,
    validate_i4_3c_binding,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderQualificationState,
    DenseProviderRunStatus,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import ReconstructionProviderExecution
from tests.i4_3c_helpers import (
    CollectingProgressSink,
    FakeDA3Runner,
    explicit_da3_selection,
    good_da3_environment,
    make_context,
)


def test_i43c_config_has_no_view_cap_and_reuses_production_scene_fraction() -> None:
    config = DA3ProductionConfig()
    payload = config.as_dict()
    assert payload["permanent_global_view_cap"] is None
    assert payload["all_registered_views_selected"] is True
    assert payload["adaptive_view_selection_deferred_to"] == "I4.3F"
    assert np.isclose(config.voxel_scene_fraction, 0.004)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_correspondences": 3},
        {"maximum_relative_residual": 0.0},
        {"voxel_scene_fraction": 0.0},
        {"pixel_stride": 0},
        {"max_voxels": 0},
        {"minimum_accepted_views": 1},
        {"maximum_pixel_scale_anisotropy": 1.0},
    ],
)
def test_i43c_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ReconstructionValidationError):
        DA3ProductionConfig(**kwargs)


def test_i43c_descriptor_maps_existing_da3_runtime_to_i43a_contract(tmp_path: Path) -> None:
    descriptor = da3_dense_provider_descriptor(good_da3_environment(tmp_path))
    assert descriptor.provider_id == DA3_PROVIDER_ID
    assert descriptor.implementation is DenseProviderImplementation.DA3
    assert descriptor.family is DenseProviderFamily.MULTI_VIEW_AI_DEPTH
    assert descriptor.execution is ReconstructionProviderExecution.EXTERNAL_PROCESS
    assert descriptor.qualification_state is DenseProviderQualificationState.CANDIDATE
    assert descriptor.available
    assert descriptor.production_qualified is False
    assert descriptor.implementation_contract.integration_stage == "I4.3C"


def test_i43c_descriptor_reflects_runtime_unavailability(tmp_path: Path) -> None:
    descriptor = da3_dense_provider_descriptor(good_da3_environment(tmp_path, available=False))
    assert not descriptor.available
    assert descriptor.unavailable_reason


def test_i43c_grid_adapter_identity_is_readonly() -> None:
    source = np.ones((4, 6), dtype=np.float32)
    adapted, report = _adapt_grid_to_sfm(
        source,
        sfm_width=6,
        sfm_height=4,
        maximum_scale_anisotropy=0.02,
        field_name="depth",
    )
    np.testing.assert_array_equal(adapted, source)
    assert not adapted.flags.writeable
    assert report["mode"] == "identity"


def test_i43c_grid_adapter_accepts_uniform_resize() -> None:
    source = np.ones((24, 32), dtype=np.float32)
    adapted, report = _adapt_grid_to_sfm(
        source,
        sfm_width=64,
        sfm_height=48,
        maximum_scale_anisotropy=0.02,
        field_name="depth",
    )
    assert adapted.shape == (48, 64)
    assert report["mode"] == "aspect_preserving_resize_to_sfm_grid"
    assert report["relative_scale_anisotropy"] == 0.0


def test_i43c_grid_adapter_rejects_anisotropic_resize() -> None:
    with pytest.raises(ReconstructionValidationError, match="not sufficiently uniform"):
        _adapt_grid_to_sfm(
            np.ones((20, 32), dtype=np.float32),
            sfm_width=64,
            sfm_height=48,
            maximum_scale_anisotropy=0.02,
            field_name="depth",
        )


def test_i43c_context_requires_pycolmap_authority(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    bad_sfm = replace(context.sfm_result, backend_id="other-sfm")
    with pytest.raises(ReconstructionValidationError, match="authoritative PyCOLMAP"):
        DA3ProviderRunContext(
            sfm_result=bad_sfm,
            reconstruction_frame_id=context.reconstruction_frame_id,
            output_dir=context.output_dir,
            da3_python=context.da3_python,
            da3_repository=context.da3_repository,
            worker_script=context.worker_script,
        )


def test_i43c_adapter_executes_once_and_fuses_only_with_sfm_poses(tmp_path: Path) -> None:
    context, env = make_context(tmp_path, view_count=3)
    runner = FakeDA3Runner()
    sink = CollectingProgressSink()
    adapter = DA3ProductionProviderAdapter(
        context,
        environment_probe=env,
        worker_runner=runner,
        progress_sink=sink,
    )
    request, decision = explicit_da3_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    assert result.implementation is DenseProviderImplementation.DA3
    assert result.source_view_count == 3 and result.selected_view_count == 3
    assert result.permanent_global_view_cap is None
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.canonical_geometry_authoritative
    assert runner.calls == 1
    assert Path(result.canonical_geometry_ref).is_file()
    assert len(sink.events) >= 7
    assert all(event.provisional_geometry_authoritative is False for event in sink.events)


def test_i43c_adapter_uses_all_registered_views_not_historical_40_cap(tmp_path: Path) -> None:
    context, env = make_context(tmp_path, view_count=41)
    adapter = DA3ProductionProviderAdapter(context, environment_probe=env, worker_runner=FakeDA3Runner())
    request, decision = explicit_da3_selection(41)
    result = adapter.execute(request, decision)
    assert result.selected_view_count == 41
    assert result.adaptive_view_selection is False
    assert result.permanent_global_view_cap is None


def test_i43c_adapter_refuses_wrong_selected_provider(tmp_path: Path) -> None:
    context, env = make_context(tmp_path)
    adapter = DA3ProductionProviderAdapter(context, environment_probe=env, worker_runner=FakeDA3Runner())
    request, decision = explicit_da3_selection(3, provider_id="another-da3")
    with pytest.raises(ReconstructionValidationError, match="concrete provider ID"):
        adapter.execute(request, decision)


def test_i43c_adapter_refuses_changed_registered_image(tmp_path: Path) -> None:
    context, env = make_context(tmp_path)
    first = Path(context.sfm_result.workspace.workspace_root) / context.sfm_result.workspace.images[0].staged_image_path
    first.write_bytes(first.read_bytes() + b"mutation")
    adapter = DA3ProductionProviderAdapter(context, environment_probe=env, worker_runner=FakeDA3Runner())
    request, decision = explicit_da3_selection(3)
    with pytest.raises(ReconstructionValidationError, match="checksum changed"):
        adapter.execute(request, decision)


def test_i43c_adapter_detects_worker_mutating_registered_image(tmp_path: Path) -> None:
    context, env = make_context(tmp_path)
    first = Path(context.sfm_result.workspace.workspace_root) / context.sfm_result.workspace.images[0].staged_image_path
    adapter = DA3ProductionProviderAdapter(
        context,
        environment_probe=env,
        worker_runner=FakeDA3Runner(mutate_path=first),
    )
    request, decision = explicit_da3_selection(3)
    with pytest.raises(ReconstructionValidationError, match="registered staged images changed"):
        adapter.execute(request, decision)


def test_i43c_adapter_refuses_mixed_camera_models_instead_of_approximating(tmp_path: Path) -> None:
    context, env = make_context(tmp_path, view_count=3, camera_ids=("camera-0", "camera-1", "camera-0"))
    adapter = DA3ProductionProviderAdapter(context, environment_probe=env, worker_runner=FakeDA3Runner())
    request, decision = explicit_da3_selection(3)
    with pytest.raises(ReconstructionValidationError, match="one shared SfM camera model"):
        adapter.execute(request, decision)


def test_i43c_adapter_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    context, env = make_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    (context.output_dir / "existing.txt").write_text("evidence", encoding="utf-8")
    adapter = DA3ProductionProviderAdapter(context, environment_probe=env, worker_runner=FakeDA3Runner())
    request, decision = explicit_da3_selection(3)
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite provider evidence"):
        adapter.execute(request, decision)


def _architecture_snapshot() -> dict:
    return {
        "phase": "I4.3A",
        "status": "pass",
        "architecture": {
            "authoritative_camera_frame_owner": "pycolmap_sfm",
            "provider_implementation_contracts": [
                {"implementation": "da3", "integration_stage": "I4.3C"},
                {"implementation": "colmap_mvs", "integration_stage": "I4.3B"},
            ],
        },
    }


def test_i43c_binding_accepts_real_architecture_contract_and_available_runtime(tmp_path: Path) -> None:
    snapshot = validate_i4_3c_binding(
        i4_3a_snapshot=_architecture_snapshot(),
        environment=good_da3_environment(tmp_path),
    )
    assert snapshot["status"] == "pass"
    assert snapshot["provider"]["implementation"] == "da3"
    assert snapshot["authority"]["backend_started"] is False
    assert snapshot["authority"]["da3_pose_authority_promoted"] is False
    assert snapshot["qualification"]["mvs_vs_da3_production_comparison_pending_i4_3g"] is True


def test_i43c_binding_fails_closed_if_i43a_da3_stage_changes(tmp_path: Path) -> None:
    architecture = _architecture_snapshot()
    architecture["architecture"]["provider_implementation_contracts"][0]["integration_stage"] = "I4.3D"
    with pytest.raises(ReconstructionValidationError, match="assigned to another stage"):
        validate_i4_3c_binding(i4_3a_snapshot=architecture, environment=good_da3_environment(tmp_path))


def test_i43c_binding_fails_closed_if_camera_authority_changes(tmp_path: Path) -> None:
    architecture = _architecture_snapshot()
    architecture["architecture"]["authoritative_camera_frame_owner"] = "da3"
    with pytest.raises(ReconstructionValidationError, match="camera-frame authority"):
        validate_i4_3c_binding(i4_3a_snapshot=architecture, environment=good_da3_environment(tmp_path))
