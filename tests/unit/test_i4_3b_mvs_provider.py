from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderQualificationState,
    DenseProviderRunStatus,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_environment import MvsHostResourceProbe, PycolmapMvsEnvironmentProbe
from core.reconstruction.mvs_provider import (
    I43B_PROVIDER_ID,
    MvsProductionConfig,
    MvsProductionProviderAdapter,
    MvsProviderRunContext,
    colmap_mvs_provider_descriptor,
    validate_existing_i4_2_mvs_binding,
)
from tests.i4_3b_helpers import (
    CollectingProgressSink,
    FakeWorkerRunner,
    explicit_selection,
    good_environment,
    good_resources,
    make_sfm_result,
    write_minimal_ply,
)


def _context(tmp_path: Path, *, view_count: int = 3) -> MvsProviderRunContext:
    py = tmp_path / "python3.11"
    worker = tmp_path / "worker.py"
    py.write_text("", encoding="utf-8")
    worker.write_text("", encoding="utf-8")
    return MvsProviderRunContext(
        sfm_result=make_sfm_result(tmp_path, view_count=view_count),
        reconstruction_frame_id="sfm-frame-1",
        output_dir=tmp_path / "mvs-output",
        pycolmap_python=py,
        worker_script=worker,
    )


def test_i43b_descriptor_binds_colmap_mvs_as_candidate_not_promoted() -> None:
    descriptor = colmap_mvs_provider_descriptor(good_environment())
    assert descriptor.provider_id == I43B_PROVIDER_ID
    assert descriptor.implementation is DenseProviderImplementation.COLMAP_MVS
    assert descriptor.qualification_state is DenseProviderQualificationState.CANDIDATE
    assert descriptor.available is True
    assert descriptor.production_qualified is False
    metadata = dict(descriptor.metadata)
    assert metadata["permanent_global_view_cap"] is None
    assert metadata["provider_promotion_performed"] is False
    assert metadata["production_qualification_pending"] == "I4.3G/I4.3H"


def test_i43b_unqualified_runtime_is_unavailable() -> None:
    probe = replace(good_environment(), pycolmap_version="4.0.0")
    descriptor = colmap_mvs_provider_descriptor(probe)
    assert descriptor.available is False
    assert descriptor.production_qualified is False


def test_i43b_config_uses_all_views_without_permanent_cap() -> None:
    config = MvsProductionConfig(source_neighbors=10)
    resolved = config.resolved(selected_view_count=68)
    assert resolved["selected_view_count"] == 68
    assert resolved["source_neighbors"] == 10
    assert resolved["all_registered_views_selected"] is True
    assert resolved["permanent_global_view_cap"] is None


def test_i43b_config_clamps_neighbors_not_view_population() -> None:
    resolved = MvsProductionConfig(source_neighbors=10).resolved(selected_view_count=3)
    assert resolved["source_neighbors"] == 2
    assert resolved["selected_view_count"] == 3


def test_i43b_config_refuses_weakening_geometric_consistency() -> None:
    with pytest.raises(ReconstructionValidationError, match="locked ON"):
        MvsProductionConfig(geometric_consistency=False)


def test_i43b_context_requires_authoritative_pycolmap_sfm(tmp_path: Path) -> None:
    result = make_sfm_result(tmp_path, backend_id="other-sfm")
    with pytest.raises(ReconstructionValidationError, match="authoritative PyCOLMAP"):
        MvsProviderRunContext(
            sfm_result=result,
            reconstruction_frame_id="sfm-frame-1",
            output_dir=tmp_path / "out",
            pycolmap_python=tmp_path / "py",
            worker_script=tmp_path / "worker",
        )


def test_i43b_adapter_executes_fake_worker_in_authoritative_frame(tmp_path: Path) -> None:
    context = _context(tmp_path)
    sink = CollectingProgressSink()
    runner = FakeWorkerRunner()
    adapter = MvsProductionProviderAdapter(
        context,
        progress_sink=sink,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=runner,
    )
    request, decision = explicit_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    assert result.selected_view_count == result.source_view_count == 3
    assert result.adaptive_view_selection is False
    assert result.permanent_global_view_cap is None
    assert result.canonical_frame_id == "sfm-frame-1"
    assert result.canonical_geometry_authoritative is True
    assert result.provider_promotion_performed is False
    assert result.metric_scale_claimed is False
    assert result.surface_mesh_generated is False
    assert Path(result.canonical_geometry_ref).is_file()
    assert runner.calls == 1
    assert len(sink.events) >= 6
    assert sink.events[-1].progress_fraction == 1.0
    assert all(event.provisional_geometry_authoritative is False for event in sink.events)


def test_i43b_adapter_rejects_selection_frame_mismatch(tmp_path: Path) -> None:
    context = _context(tmp_path)
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(),
    )
    request, decision = explicit_selection(3, frame_id="wrong-frame")
    with pytest.raises(ReconstructionValidationError, match="frame ID"):
        adapter.execute(request, decision)


def test_i43b_adapter_rejects_view_count_mismatch(tmp_path: Path) -> None:
    context = _context(tmp_path)
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(),
    )
    request, decision = explicit_selection(2)
    with pytest.raises(ReconstructionValidationError, match="available_view_count"):
        adapter.execute(request, decision)


def test_i43b_adapter_fails_closed_on_insufficient_resources(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resources = MvsHostResourceProbe(workspace_path=str(tmp_path), disk_free_gib=1.0, ram_available_gib=1.0, gpus=())
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=resources,
        worker_runner=FakeWorkerRunner(),
    )
    request, decision = explicit_selection(3)
    with pytest.raises(ReconstructionValidationError, match="guardrails"):
        adapter.execute(request, decision)


def test_i43b_worker_failure_returns_non_authoritative_failed_result(tmp_path: Path) -> None:
    context = _context(tmp_path)
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(return_code=2, status="error"),
    )
    request, decision = explicit_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.FAILED
    assert result.canonical_geometry_ref == ""
    assert result.canonical_geometry_authoritative is False
    assert "synthetic worker failure" in result.failure_reason


def test_i43b_fails_closed_when_worker_mutates_authoritative_sparse(tmp_path: Path) -> None:
    context = _context(tmp_path)
    sparse = Path(context.sfm_result.workspace.workspace_root) / "sparse/0/cameras.bin"
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(mutate_path=sparse),
    )
    request, decision = explicit_selection(3)
    with pytest.raises(ReconstructionValidationError, match="sparse model changed"):
        adapter.execute(request, decision)


def test_i43b_fails_closed_on_ply_visibility_count_mismatch(tmp_path: Path) -> None:
    context = _context(tmp_path)
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(ply_points=3, vis_points=2),
    )
    request, decision = explicit_selection(3)
    with pytest.raises(ReconstructionValidationError, match="correspondence"):
        adapter.execute(request, decision)


def test_i43b_fails_closed_when_worker_mutates_registered_image(tmp_path: Path) -> None:
    context = _context(tmp_path)
    image = Path(context.sfm_result.workspace.workspace_root) / "images/frame_0000.png"
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(mutate_path=image),
    )
    request, decision = explicit_selection(3)
    with pytest.raises(ReconstructionValidationError, match="staged images changed"):
        adapter.execute(request, decision)


def test_i43b_refuses_overwriting_existing_provider_evidence(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.output_dir.mkdir(parents=True)
    (context.output_dir / "existing-evidence.json").write_text("{}", encoding="utf-8")
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(),
    )
    request, decision = explicit_selection(3)
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite provider evidence"):
        adapter.execute(request, decision)


def test_i43b_adapter_refuses_execution_when_another_provider_is_selected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(context.pycolmap_python)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(),
    )
    request, decision = explicit_selection(3, provider_id="some-other-mvs-provider")
    with pytest.raises(ReconstructionValidationError, match="concrete provider ID"):
        adapter.execute(request, decision)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_binding_inputs(tmp_path: Path):
    fused = tmp_path / "i4_2d" / "dense" / "fused.ply"
    write_minimal_ply(fused, count=5)
    i4_2d = {
        "phase": "I4.2D",
        "status": "pass",
        "pass": True,
        "authority": {
            "single_authoritative_sfm_frame": True,
            "raw_mvs_dense_reconstruction": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "held_out_ground_truth_consumed_by_i4_2d": False,
            "production_filter_policy_changed": False,
            "production_manifold_applied": False,
            "dense_provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "artifacts": {"raw_fused_ply": str(fused)},
    }
    i4_2d_path = tmp_path / "i4_2d" / "i4_2d_summary.json"
    i4_2d_path.write_text(json.dumps(i4_2d), encoding="utf-8")
    closure = {
        "phase": "I4.2F",
        "status": "pass",
        "decision": {"i4_3_provider_architecture_authorized": True},
        "evidence": {
            "upstream": {"i4_2d_summary": str(i4_2d_path)},
            "source_artifact_sha256": {
                "raw_fused_ply": _sha256(fused),
                "native_visibility": _sha256(Path(f"{fused}.vis")),
            },
        },
    }
    architecture = {
        "phase": "I4.3A",
        "status": "pass",
        "architecture": {
            "authoritative_camera_frame_owner": "pycolmap_sfm",
            "permanent_global_view_cap": None,
            "provider_implementation_contracts": [
                {"implementation": "colmap_mvs", "integration_stage": "I4.3B"}
            ],
        },
    }
    return closure, architecture, fused


def test_i43b_frozen_evidence_binding_accepts_qualified_i42_mvs(tmp_path: Path) -> None:
    closure, architecture, _ = _frozen_binding_inputs(tmp_path)
    snapshot = validate_existing_i4_2_mvs_binding(
        i4_2f_closure=closure,
        i4_3a_snapshot=architecture,
        environment=good_environment(),
    )
    assert snapshot["status"] == "pass"
    assert snapshot["provider"]["qualification_state"] == "candidate"
    assert snapshot["frozen_mvs_evidence"]["fused_points"] == 5
    assert snapshot["authority"]["backend_started"] is False
    assert snapshot["qualification"]["production_operating_envelope_pending_i4_3g"] is True


def test_i43b_frozen_evidence_binding_detects_post_freeze_geometry_mutation(tmp_path: Path) -> None:
    closure, architecture, fused = _frozen_binding_inputs(tmp_path)
    fused.write_bytes(fused.read_bytes() + b"post-freeze mutation")
    with pytest.raises(ReconstructionValidationError, match="changed after I4.2F evidence freeze"):
        validate_existing_i4_2_mvs_binding(
            i4_2f_closure=closure,
            i4_3a_snapshot=architecture,
            environment=good_environment(),
        )


def test_i43b_frozen_evidence_binding_fails_closed_on_changed_i43a_authority(tmp_path: Path) -> None:
    closure, architecture, _ = _frozen_binding_inputs(tmp_path)
    architecture["architecture"]["authoritative_camera_frame_owner"] = "another_frame_owner"
    with pytest.raises(ReconstructionValidationError, match="changed camera-frame authority"):
        validate_existing_i4_2_mvs_binding(
            i4_2f_closure=closure,
            i4_3a_snapshot=architecture,
            environment=good_environment(),
        )
