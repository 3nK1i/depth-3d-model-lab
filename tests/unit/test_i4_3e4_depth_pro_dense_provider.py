from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from core.reconstruction.depth_pro_dense_provider import (
    DepthProProductionProviderAdapter,
    depth_pro_dense_provider_descriptor,
    probe_depth_pro_environment,
    validate_i4_3e4_binding,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3e4_helpers import (
    FakeDepthProWorkerRunner,
    architecture_snapshot,
    d3_snapshot,
    e1_snapshot,
    e2_snapshot,
    e3_snapshot,
    explicit_depth_pro_selection,
    good_depth_pro_environment,
    make_depth_pro_context,
)


def test_i43e4_descriptor_freezes_depth_pro_native_semantics_without_authority_promotion(tmp_path: Path) -> None:
    descriptor = depth_pro_dense_provider_descriptor(good_depth_pro_environment(tmp_path))
    assert descriptor.provider_id == "depth-pro-single-image"
    assert descriptor.implementation.value == "depth_pro"
    assert descriptor.qualification_state.value == "candidate"
    assert descriptor.production_qualified is False
    metadata = dict(descriptor.metadata)
    assert metadata["native_depth_scale"] == "metric"
    assert metadata["native_intrinsics_source"] == "model_predicted_with_estimated_fallback"
    assert metadata["native_confidence_kind"] == "none"
    assert metadata["native_direct_points_available"] is False
    assert metadata["native_normals_available"] is False
    assert metadata["model_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert metadata["model_predicted_intrinsics_consumed_for_canonical_projection"] is False
    assert metadata["permanent_global_view_cap"] is None
    assert metadata["fixed_internal_resolution"] == 1536


def test_i43e4_fake_worker_execution_preserves_sfm_authority_and_all_views(tmp_path: Path) -> None:
    context, env = make_depth_pro_context(tmp_path, view_count=3)
    runner = FakeDepthProWorkerRunner()
    adapter = DepthProProductionProviderAdapter(context=context, environment=env, runner=runner)
    request, decision = explicit_depth_pro_selection(3)
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
    assert runner.last_manifest["checkpoint_sha256"] == env.checkpoint_sha256
    assert runner.last_manifest["authority"]["model_predicted_intrinsics_are_canonical_authority"] is False
    summary = json.loads(
        (context.output_dir / "i4_3e4_depth_pro_execution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["authority"]["native_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert summary["authority"]["model_predicted_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["integrity"]["registered_image_hashes_preserved"] is True


def test_i43e4_refuses_depth_pixel_grid_drift(tmp_path: Path) -> None:
    context, env = make_depth_pro_context(tmp_path, view_count=3)
    adapter = DepthProProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthProWorkerRunner(output_shape=(24, 32)),
    )
    request, decision = explicit_depth_pro_selection(3)
    with pytest.raises(ReconstructionValidationError, match="pixel grid"):
        adapter.execute(request, decision)


def test_i43e4_refuses_source_or_checkpoint_checksum_drift(tmp_path: Path) -> None:
    runners = (
        FakeDepthProWorkerRunner(source_revision="c" * 40),
        FakeDepthProWorkerRunner(checkpoint_sha256="d" * 64),
    )
    for index, runner in enumerate(runners):
        context, env = make_depth_pro_context(tmp_path / f"case-{index}", view_count=3)
        adapter = DepthProProductionProviderAdapter(context=context, environment=env, runner=runner)
        request, decision = explicit_depth_pro_selection(3)
        with pytest.raises(ReconstructionValidationError, match="revision|checksum"):
            adapter.execute(request, decision)


def test_i43e4_refuses_depth_pro_semantics_drift_or_fabricated_outputs(tmp_path: Path) -> None:
    cases = (
        (FakeDepthProWorkerRunner(native_depth_scale="relative", native_is_metric=False), "metric-depth"),
        (FakeDepthProWorkerRunner(intrinsics_source_native="none"), "intrinsics"),
        (FakeDepthProWorkerRunner(confidence_kind_native="confidence"), "no-confidence"),
        (FakeDepthProWorkerRunner(direct_points_native=True), "fabricated"),
        (FakeDepthProWorkerRunner(normals_native=True), "fabricated"),
    )
    for index, (runner, message) in enumerate(cases):
        context, env = make_depth_pro_context(tmp_path / f"sem-{index}", view_count=3)
        adapter = DepthProProductionProviderAdapter(context=context, environment=env, runner=runner)
        request, decision = explicit_depth_pro_selection(3)
        with pytest.raises(ReconstructionValidationError, match=message):
            adapter.execute(request, decision)


def test_i43e4_refuses_registered_image_mutation(tmp_path: Path) -> None:
    context, env = make_depth_pro_context(tmp_path, view_count=3)
    workspace_root = Path(context.sfm_result.workspace.workspace_root)
    first = sorted(context.sfm_result.workspace.images, key=lambda item: item.order)[0]
    mutate_path = workspace_root / first.staged_image_path
    adapter = DepthProProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthProWorkerRunner(mutate_path=mutate_path),
    )
    request, decision = explicit_depth_pro_selection(3)
    with pytest.raises(ReconstructionValidationError, match="changed during provider execution"):
        adapter.execute(request, decision)


def test_i43e4_refuses_multi_camera_run_until_explicit_camera_handling_exists(tmp_path: Path) -> None:
    context, env = make_depth_pro_context(
        tmp_path,
        view_count=3,
        camera_ids=("cam-a", "cam-b", "cam-a"),
    )
    adapter = DepthProProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthProWorkerRunner(),
    )
    request, decision = explicit_depth_pro_selection(3)
    with pytest.raises(ReconstructionValidationError, match="one authoritative SfM camera model"):
        adapter.execute(request, decision)


def test_i43e4_refuses_unavailable_runtime_wrong_provider_and_fallback(tmp_path: Path) -> None:
    context, _ = make_depth_pro_context(tmp_path / "unavailable", view_count=3)
    adapter = DepthProProductionProviderAdapter(
        context=context,
        environment=good_depth_pro_environment(tmp_path / "unavailable", available=False),
        runner=FakeDepthProWorkerRunner(),
    )
    request, decision = explicit_depth_pro_selection(3)
    with pytest.raises(ReconstructionValidationError, match="runtime is unavailable"):
        adapter.execute(request, decision)

    context, env = make_depth_pro_context(tmp_path / "wrong", view_count=3)
    adapter = DepthProProductionProviderAdapter(context=context, environment=env, runner=FakeDepthProWorkerRunner())
    request, decision = explicit_depth_pro_selection(3, provider_id="wrong-provider")
    with pytest.raises(ReconstructionValidationError, match="does not match"):
        adapter.execute(request, decision)

    context, env = make_depth_pro_context(tmp_path / "fallback", view_count=3)
    adapter = DepthProProductionProviderAdapter(context=context, environment=env, runner=FakeDepthProWorkerRunner())
    request, decision = explicit_depth_pro_selection(3, allow_fallback=True)
    with pytest.raises(ReconstructionValidationError, match="forbids silent provider fallback"):
        adapter.execute(request, decision)


def test_i43e4_binding_requires_closed_predecessors_and_preserves_authority(tmp_path: Path) -> None:
    snapshot = validate_i4_3e4_binding(
        i4_3a_snapshot=architecture_snapshot(),
        i4_3d3_snapshot=d3_snapshot(),
        i4_3e1_snapshot=e1_snapshot(),
        i4_3e2_snapshot=e2_snapshot(),
        i4_3e3_snapshot=e3_snapshot(),
        environment=good_depth_pro_environment(tmp_path),
    )
    assert snapshot["pass"] is True
    assert snapshot["provider"]["implementation"] == "depth_pro"
    assert snapshot["provider"]["qualification_state"] == "candidate"
    assert snapshot["architecture"]["native_confidence_kind"] == "none"
    assert snapshot["architecture"]["native_direct_points_available"] is False
    assert snapshot["architecture"]["native_normals_available"] is False
    assert snapshot["authority"]["model_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert snapshot["authority"]["model_predicted_intrinsics_consumed_for_canonical_projection"] is False
    assert snapshot["qualification"]["predecessor_i4_3e3_verified"] is True
    assert snapshot["qualification"]["remaining_i4_3e_providers_pending"] == [
        "depth_anything_v2",
        "marigold",
    ]


def test_i43e4_binding_refuses_promoted_predecessor_authority(tmp_path: Path) -> None:
    bad = e3_snapshot()
    bad["authority"]["measurement_authoritative"] = True
    with pytest.raises(ReconstructionValidationError, match="I4.3E3 predecessor"):
        validate_i4_3e4_binding(
            i4_3a_snapshot=architecture_snapshot(),
            i4_3d3_snapshot=d3_snapshot(),
            i4_3e1_snapshot=e1_snapshot(),
            i4_3e2_snapshot=e2_snapshot(),
            i4_3e3_snapshot=bad,
            environment=good_depth_pro_environment(tmp_path),
        )


def test_i43e4_worker_import_is_lazy() -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/reconstruction_depth_pro_provider_worker.py"
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("i43e4_depth_pro_worker_probe", worker)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new_modules = set(sys.modules) - before
    for forbidden in ("torch", "cv2", "depth_pro", "core.models.registry"):
        assert forbidden not in new_modules
    assert callable(module.main) and callable(module.run)


def test_i43e4_environment_probe_verifies_official_vcs_runtime_and_checkpoint(tmp_path: Path, monkeypatch) -> None:
    import core.reconstruction.depth_pro_dense_provider as module

    root = tmp_path / "project"
    requirements = root / "requirements"
    requirements.mkdir(parents=True)
    (requirements / "models-depth-pro.txt").write_text(
        "depth_pro @ git+https://github.com/apple/ml-depth-pro.git\n",
        encoding="utf-8",
    )
    checkpoint = root / "checkpoints/depth_pro.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(module, "I43E4_MINIMUM_CHECKPOINT_BYTES", 1)

    dist_root = tmp_path / "site-packages" / "depth_pro"
    runtime_file = dist_root / "depth_pro.py"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("# synthetic\n", encoding="utf-8")

    class FakeDistribution:
        metadata = {"Name": "depth-pro"}
        version = "0.0.synthetic"

        def locate_file(self, name):
            assert str(name) == "depth_pro"
            return dist_root

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/apple/ml-depth-pro.git",
                    "vcs_info": {"vcs": "git", "commit_id": "c" * 40},
                }
            )

    monkeypatch.setattr(module.importlib_metadata, "distribution", lambda name: FakeDistribution())
    fake_depth_pro = types.ModuleType("depth_pro")

    def fake_factory(*args, **kwargs):
        return args, kwargs

    fake_depth_pro.create_model_and_transforms = fake_factory
    monkeypatch.setitem(sys.modules, "depth_pro", fake_depth_pro)
    monkeypatch.setattr(module.inspect, "getfile", lambda obj: str(runtime_file))

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_count=lambda: 1,
        get_device_name=lambda index: "NVIDIA GeForce RTX 2060",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    probe = probe_depth_pro_environment(
        project_root=root,
        python_executable=python,
        checkpoint_path=checkpoint,
    )
    assert probe.available is True
    assert probe.source_revision == "c" * 40
    assert probe.source_revision_resolved is True
    assert probe.requirement_commit_pinned is False
    assert probe.official_runtime_source_matches_distribution is True
    assert probe.checkpoint_ready is True
    assert len(probe.checkpoint_sha256) == 64


def test_i43e4_environment_probe_detects_explicit_requirement_commit_pin(tmp_path: Path) -> None:
    import core.reconstruction.depth_pro_dense_provider as module

    requirement = tmp_path / "models-depth-pro.txt"
    requirement.write_text(
        "depth_pro @ git+https://github.com/apple/ml-depth-pro.git@" + "f" * 40 + "\n",
        encoding="utf-8",
    )
    assert module._depth_pro_requirement_commit_pinned(requirement) is True


def test_i43e4_refuses_worker_native_authority_promotion(tmp_path: Path) -> None:
    context, env = make_depth_pro_context(tmp_path, view_count=3)
    adapter = DepthProProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthProWorkerRunner(promote_native_authority=True),
    )
    request, decision = explicit_depth_pro_selection(3)
    with pytest.raises(ReconstructionValidationError, match="promote native Depth Pro authority"):
        adapter.execute(request, decision)
