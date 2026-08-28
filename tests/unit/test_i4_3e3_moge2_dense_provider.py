from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderQualificationState,
    DenseProviderRunStatus,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.moge2_dense_provider import (
    MoGe2ProductionConfig,
    MoGe2ProductionProviderAdapter,
    moge_2_dense_provider_descriptor,
    validate_i4_3e3_binding,
)
from tests.i4_3e3_helpers import (
    CollectingProgressSink,
    FakeMoGeWorkerRunner,
    architecture_snapshot,
    d3_snapshot,
    e1_snapshot,
    e2_snapshot,
    explicit_moge_selection,
    good_moge_environment,
    make_moge_context,
)


def test_i43e3_descriptor_freezes_moge_native_semantics_without_authority_promotion(tmp_path: Path) -> None:
    descriptor = moge_2_dense_provider_descriptor(good_moge_environment(tmp_path))
    metadata = descriptor.as_dict()["metadata"]
    assert descriptor.implementation is DenseProviderImplementation.MOGE_2
    assert descriptor.qualification_state is DenseProviderQualificationState.CANDIDATE
    assert descriptor.available is True
    assert metadata["native_depth_scale"] == "metric"
    assert metadata["native_intrinsics_source"] == "model_predicted"
    assert metadata["native_confidence_kind"] == "none"
    assert metadata["native_direct_points_available"] is True
    assert metadata["native_normals_available"] is True
    assert metadata["model_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert metadata["model_predicted_intrinsics_consumed_for_canonical_projection"] is False
    assert metadata["model_direct_points_consumed_for_canonical_geometry"] is False
    assert metadata["model_normals_consumed_for_canonical_geometry"] is False
    assert metadata["permanent_global_view_cap"] is None
    assert metadata["license_status"] == "review_required"


def test_i43e3_production_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ReconstructionValidationError, match="minimum_correspondences"):
        MoGe2ProductionConfig(minimum_correspondences=2)
    with pytest.raises(ReconstructionValidationError, match="maximum_relative_residual"):
        MoGe2ProductionConfig(maximum_relative_residual=1.0)
    with pytest.raises(ReconstructionValidationError, match="voxel_scene_fraction"):
        MoGe2ProductionConfig(voxel_scene_fraction=0.2)
    with pytest.raises(ReconstructionValidationError, match="minimum_accepted_views"):
        MoGe2ProductionConfig(minimum_accepted_views=1)


def test_i43e3_execute_uses_all_views_sfm_authority_and_no_synthetic_confidence(tmp_path: Path) -> None:
    context, env = make_moge_context(tmp_path, view_count=3)
    runner = FakeMoGeWorkerRunner()
    sink = CollectingProgressSink()
    adapter = MoGe2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=runner,
        progress_sink=sink,
    )
    request, decision = explicit_moge_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    assert result.source_view_count == 3
    assert result.selected_view_count == 3
    assert result.permanent_global_view_cap is None
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.metric_scale_claimed is False
    assert result.provider_promotion_performed is False
    assert runner.last_manifest["runtime_config"]["local_files_only"] is True
    assert runner.last_manifest["checkpoint_revision"] == "b" * 40
    assert len(sink.events) >= 5

    summary = json.loads(
        (context.output_dir / "i4_3e3_moge_2_execution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["authority"]["model_predicted_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["authority"]["model_direct_points_consumed_for_canonical_geometry"] is False
    assert summary["authority"]["model_normals_consumed_for_canonical_geometry"] is False
    assert summary["authority"]["metric_scale_claimed"] is False
    assert all(record["native_confidence_available"] is False for record in summary["alignment"]["records"])


def test_i43e3_refuses_wrong_pixel_grid(tmp_path: Path) -> None:
    context, env = make_moge_context(tmp_path, view_count=3)
    adapter = MoGe2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMoGeWorkerRunner(output_shape=(24, 32)),
    )
    request, decision = explicit_moge_selection(3)
    with pytest.raises(ReconstructionValidationError, match="pixel grid"):
        adapter.execute(request, decision)


def test_i43e3_refuses_source_or_checkpoint_revision_drift(tmp_path: Path) -> None:
    runners = (
        FakeMoGeWorkerRunner(source_revision="c" * 40),
        FakeMoGeWorkerRunner(checkpoint_repository="wrong/repo"),
        FakeMoGeWorkerRunner(checkpoint_revision="d" * 40),
    )
    for index, runner in enumerate(runners):
        context, env = make_moge_context(tmp_path / f"case-{index}", view_count=3)
        adapter = MoGe2ProductionProviderAdapter(context=context, environment=env, runner=runner)
        request, decision = explicit_moge_selection(3)
        with pytest.raises(ReconstructionValidationError, match="revision|repository"):
            adapter.execute(request, decision)


def test_i43e3_refuses_native_semantics_drift(tmp_path: Path) -> None:
    cases = (
        (FakeMoGeWorkerRunner(native_depth_scale="relative", native_is_metric=False), "metric-depth"),
        (FakeMoGeWorkerRunner(intrinsics_source_native="estimated"), "intrinsics"),
        (FakeMoGeWorkerRunner(confidence_kind_native="confidence"), "no-confidence"),
        (FakeMoGeWorkerRunner(direct_points_native=False), "geometry evidence"),
        (FakeMoGeWorkerRunner(normals_native=False), "geometry evidence"),
    )
    for index, (runner, message) in enumerate(cases):
        context, env = make_moge_context(tmp_path / f"sem-{index}", view_count=3)
        adapter = MoGe2ProductionProviderAdapter(context=context, environment=env, runner=runner)
        request, decision = explicit_moge_selection(3)
        with pytest.raises(ReconstructionValidationError, match=message):
            adapter.execute(request, decision)


def test_i43e3_refuses_registered_source_mutation(tmp_path: Path) -> None:
    context, env = make_moge_context(tmp_path, view_count=3)
    first = context.sfm_result.workspace.images[0]
    mutate_path = Path(context.sfm_result.workspace.workspace_root) / first.staged_image_path
    adapter = MoGe2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMoGeWorkerRunner(mutate_path=mutate_path),
    )
    request, decision = explicit_moge_selection(3)
    with pytest.raises(ReconstructionValidationError, match="changed during provider execution"):
        adapter.execute(request, decision)


def test_i43e3_refuses_multiple_sfm_camera_models(tmp_path: Path) -> None:
    context, env = make_moge_context(
        tmp_path,
        view_count=3,
        camera_ids=("cam-a", "cam-b", "cam-a"),
    )
    adapter = MoGe2ProductionProviderAdapter(context=context, environment=env, runner=FakeMoGeWorkerRunner())
    request, decision = explicit_moge_selection(3)
    with pytest.raises(ReconstructionValidationError, match="one authoritative SfM camera model"):
        adapter.execute(request, decision)


def test_i43e3_refuses_unavailable_runtime_wrong_provider_and_fallback(tmp_path: Path) -> None:
    context, _ = make_moge_context(tmp_path / "unavailable", view_count=3)
    adapter = MoGe2ProductionProviderAdapter(
        context=context,
        environment=good_moge_environment(tmp_path / "unavailable", available=False),
        runner=FakeMoGeWorkerRunner(),
    )
    request, decision = explicit_moge_selection(3)
    with pytest.raises(ReconstructionValidationError, match="runtime is unavailable"):
        adapter.execute(request, decision)

    context, env = make_moge_context(tmp_path / "wrong", view_count=3)
    adapter = MoGe2ProductionProviderAdapter(context=context, environment=env, runner=FakeMoGeWorkerRunner())
    request, decision = explicit_moge_selection(3, provider_id="wrong-provider")
    with pytest.raises(ReconstructionValidationError, match="provider ID"):
        adapter.execute(request, decision)

    context, env = make_moge_context(tmp_path / "fallback", view_count=3)
    adapter = MoGe2ProductionProviderAdapter(context=context, environment=env, runner=FakeMoGeWorkerRunner())
    request, decision = explicit_moge_selection(3, allow_fallback=True)
    with pytest.raises(ReconstructionValidationError, match="silent provider fallback"):
        adapter.execute(request, decision)


def test_i43e3_refuses_runtime_repository_mismatch(tmp_path: Path) -> None:
    context, env = make_moge_context(tmp_path, view_count=3)
    object.__setattr__(context, "runtime_config", tuple(sorted({**dict(context.runtime_config), "pretrained": "wrong/repo"}.items())))
    adapter = MoGe2ProductionProviderAdapter(context=context, environment=env, runner=FakeMoGeWorkerRunner())
    request, decision = explicit_moge_selection(3)
    with pytest.raises(ReconstructionValidationError, match="pretrained repository"):
        adapter.execute(request, decision)


def test_i43e3_binding_requires_closed_e1_e2_and_preserves_authority(tmp_path: Path) -> None:
    snapshot = validate_i4_3e3_binding(
        i4_3a_snapshot=architecture_snapshot(),
        i4_3d3_snapshot=d3_snapshot(),
        i4_3e1_snapshot=e1_snapshot(),
        i4_3e2_snapshot=e2_snapshot(),
        environment=good_moge_environment(tmp_path),
    )
    assert snapshot["status"] == "pass"
    assert snapshot["provider"]["implementation"] == "moge_2"
    assert snapshot["architecture"]["native_confidence_kind"] == "none"
    assert snapshot["authority"]["model_direct_points_consumed_for_canonical_geometry"] is False
    assert snapshot["authority"]["model_normals_consumed_for_canonical_geometry"] is False
    assert snapshot["qualification"]["predecessor_i4_3e1_verified"] is True
    assert snapshot["qualification"]["predecessor_i4_3e2_verified"] is True
    assert snapshot["qualification"]["remaining_i4_3e_providers_pending"] == [
        "depth_pro",
        "depth_anything_v2",
        "marigold",
    ]


def test_i43e3_binding_fails_closed_on_predecessor_authority_promotion(tmp_path: Path) -> None:
    bad_e2 = e2_snapshot()
    bad_e2["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="I4.3E2 predecessor with promoted authority"):
        validate_i4_3e3_binding(
            i4_3a_snapshot=architecture_snapshot(),
            i4_3d3_snapshot=d3_snapshot(),
            i4_3e1_snapshot=e1_snapshot(),
            i4_3e2_snapshot=bad_e2,
            environment=good_moge_environment(tmp_path),
        )


def test_i43e3_worker_import_is_model_and_cuda_lazy() -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/reconstruction_moge2_provider_worker.py"
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("i43e3_worker_probe", worker)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new_modules = set(sys.modules) - before
    for forbidden in ("torch", "cv2", "moge", "core.models.registry"):
        assert forbidden not in new_modules
    assert callable(module.main) and callable(module.run)


def test_i43e3_environment_probe_verifies_exact_vcs_runtime_and_cached_snapshot(tmp_path: Path, monkeypatch) -> None:
    import types
    import core.reconstruction.moge2_dense_provider as module

    dist_root = tmp_path / "site-packages" / "moge"
    class_file = dist_root / "model" / "v2.py"
    class_file.parent.mkdir(parents=True, exist_ok=True)
    class_file.write_text("# synthetic\n", encoding="utf-8")

    fake_package = types.ModuleType("moge")
    fake_package.__path__ = [str(dist_root)]
    fake_package.__file__ = str(dist_root / "__init__.py")
    fake_model = types.ModuleType("moge.model")
    fake_v2 = types.ModuleType("moge.model.v2")
    fake_v2.__file__ = str(class_file)
    FakeOfficialMoGeModel = type("MoGeModel", (), {})
    FakeOfficialMoGeModel.__module__ = "moge.model.v2"
    fake_v2.MoGeModel = FakeOfficialMoGeModel
    monkeypatch.setitem(sys.modules, "moge", fake_package)
    monkeypatch.setitem(sys.modules, "moge.model", fake_model)
    monkeypatch.setitem(sys.modules, "moge.model.v2", fake_v2)

    class FakeDistribution:
        metadata = {"Name": "moge"}
        version = "0.0.synthetic"

        def locate_file(self, name):
            assert str(name) == "moge"
            return dist_root

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/microsoft/MoGe.git",
                    "vcs_info": {"vcs": "git", "commit_id": "c" * 40},
                }
            )

    monkeypatch.setattr(module.importlib_metadata, "distribution", lambda name: FakeDistribution())

    hub = tmp_path / "hf-hub"
    repo = hub / "models--Ruicheng--moge-2-vits-normal"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("d" * 40, encoding="utf-8")
    snapshot = repo / "snapshots" / ("d" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(module, "_huggingface_hub_root", lambda: hub)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_count=lambda: 1,
        get_device_name=lambda index: "NVIDIA GeForce RTX 2060",
        get_device_properties=lambda index: types.SimpleNamespace(total_memory=6 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    probe = module.probe_moge_2_environment(
        project_root=tmp_path,
        python_executable=python,
        variant="auto",
    )
    assert probe.available is True
    assert probe.variant == "small"
    assert probe.source_revision == "c" * 40
    assert probe.source_revision_pinned is True
    assert probe.official_runtime_source_matches_distribution is True
    assert probe.checkpoint_repository == "Ruicheng/moge-2-vits-normal"
    assert probe.checkpoint_revision == "d" * 40
    assert probe.checkpoint_ready is True


def test_i43e3_refuses_worker_native_authority_promotion(tmp_path: Path) -> None:
    context, env = make_moge_context(tmp_path, view_count=3)
    adapter = MoGe2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMoGeWorkerRunner(promote_native_authority=True),
    )
    request, decision = explicit_moge_selection(3)
    with pytest.raises(ReconstructionValidationError, match="promote native MoGe-2 authority"):
        adapter.execute(request, decision)
