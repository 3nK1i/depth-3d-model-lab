from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path
import sys

import numpy as np
import pytest

from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.unidepth_v2_dense_provider import (
    UniDepthV2ProductionProviderAdapter,
    _select_unidepth_variant,
    probe_unidepth_v2_environment,
    unidepth_v2_dense_provider_descriptor,
    validate_i4_3e2_binding,
)
from tests.i4_3e2_helpers import (
    CollectingProgressSink,
    FakeUniDepthWorkerRunner,
    architecture_snapshot,
    d3_snapshot,
    e1_snapshot,
    explicit_unidepth_selection,
    good_unidepth_environment,
    make_unidepth_context,
)


def test_i43e2_descriptor_preserves_unidepth_specific_semantics_and_authority_locks(tmp_path: Path) -> None:
    descriptor = unidepth_v2_dense_provider_descriptor(good_unidepth_environment(tmp_path))
    payload = descriptor.as_dict()
    assert descriptor.available is True
    assert payload["implementation"] == "unidepth_v2"
    assert payload["family"] == "single_image_depth"
    assert payload["qualification_state"] == "candidate"
    assert payload["metadata"]["native_depth_scale"] == "metric"
    assert payload["metadata"]["native_intrinsics_source"] == "model_predicted_with_estimated_fallback"
    assert payload["metadata"]["native_confidence_kind"] == "uncertainty"
    assert payload["metadata"]["native_direct_points_available"] is True
    assert payload["metadata"]["model_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert payload["metadata"]["model_predicted_intrinsics_consumed_for_canonical_projection"] is False
    assert payload["metadata"]["model_direct_points_consumed_for_canonical_geometry"] is False
    assert payload["metadata"]["permanent_global_view_cap"] is None
    assert payload["metadata"]["license_status"] == "noncommercial_review_required"



def test_i43e2_environment_probe_accepts_namespace_package_when_runtime_class_is_from_pinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    source = project_root / "third_party/UniDepth"
    package_dir = source / "unidepth"
    model_file = package_dir / "models/unidepthv2/unidepthv2.py"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("class UniDepthV2: pass\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".eia_unidepth_v2_compat.json").write_text(
        json.dumps(
            {
                "commit": "8d8cfe4c7ee15297099983607febf0d4f32eb3d6",
                "purpose": "V2-only imports; model and inference implementation unchanged",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = project_root / "checkpoints/unidepth-v2/vits14"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    python = project_root / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    fake_package = types.ModuleType("unidepth")
    fake_package.__file__ = None
    fake_package.__path__ = [str(package_dir)]
    fake_models = types.ModuleType("unidepth.models")
    FakeOfficialUniDepthV2 = type("UniDepthV2", (), {})
    FakeOfficialUniDepthV2.__module__ = "unidepth.models.unidepthv2.unidepthv2"
    fake_models.UniDepthV2 = FakeOfficialUniDepthV2
    monkeypatch.setitem(sys.modules, "unidepth", fake_package)
    monkeypatch.setitem(sys.modules, "unidepth.models", fake_models)

    import core.reconstruction.unidepth_v2_dense_provider as module

    real_getfile = module.inspect.getfile

    def fake_getfile(obj):
        if obj is FakeOfficialUniDepthV2:
            return str(model_file)
        return real_getfile(obj)

    monkeypatch.setattr(module.inspect, "getfile", fake_getfile)

    class Completed:
        returncode = 0
        stdout = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6\n"
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Completed())

    probe = probe_unidepth_v2_environment(
        project_root=project_root,
        python_executable=python,
        checkpoint_path=checkpoint,
        variant="small",
    )

    assert probe.official_runtime_importable is True
    assert probe.official_runtime_source_matches_checkout is True
    assert "does not resolve to the pinned checkout" not in probe.error

def test_i43e2_auto_variant_thresholds_match_existing_runtime_policy() -> None:
    assert _select_unidepth_variant(cuda_available=False, total_vram_gib=None) == "small"
    assert _select_unidepth_variant(cuda_available=True, total_vram_gib=6.0) == "small"
    assert _select_unidepth_variant(cuda_available=True, total_vram_gib=12.0) == "base"
    assert _select_unidepth_variant(cuda_available=True, total_vram_gib=32.0) == "large"


def test_i43e2_execute_unidepth_candidate_preserves_authoritative_sfm(tmp_path: Path) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=3)
    runner = FakeUniDepthWorkerRunner(uncertainty_value=0.25)
    sink = CollectingProgressSink()
    adapter = UniDepthV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=runner,
        progress_sink=sink,
    )
    request, decision = explicit_unidepth_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    assert result.source_view_count == 3
    assert result.selected_view_count == 3
    assert result.permanent_global_view_cap is None
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.metric_scale_claimed is False
    assert result.measurement_authoritative is False
    assert result.provider_promotion_performed is False
    assert runner.last_manifest["variant"] == "small"
    assert runner.last_manifest["checkpoint_expected_revision"] == "0" * 40
    assert len(sink.events) >= 6

    with np.load(Path(result.canonical_geometry_ref), allow_pickle=False) as archive:
        mean_confidence = np.asarray(archive["mean_confidence"], dtype=np.float32)
        assert np.isfinite(mean_confidence).all()
        assert np.allclose(mean_confidence, 0.8, atol=1e-5)


def test_i43e2_accepts_estimated_intrinsics_fallback_as_noncanonical_evidence(tmp_path: Path) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=3)
    adapter = UniDepthV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeUniDepthWorkerRunner(intrinsics_source_native="estimated"),
    )
    request, decision = explicit_unidepth_selection(3)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED


def test_i43e2_refuses_wrong_pixel_grid(tmp_path: Path) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=3)
    adapter = UniDepthV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeUniDepthWorkerRunner(output_shape=(24, 32)),
    )
    request, decision = explicit_unidepth_selection(3)
    with pytest.raises(ReconstructionValidationError, match="pixel grid"):
        adapter.execute(request, decision)


def test_i43e2_refuses_negative_or_nonfinite_uncertainty(tmp_path: Path) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=3)
    adapter = UniDepthV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeUniDepthWorkerRunner(uncertainty_value=-0.1),
    )
    request, decision = explicit_unidepth_selection(3)
    with pytest.raises(ReconstructionValidationError, match="uncertainty"):
        adapter.execute(request, decision)


def test_i43e2_refuses_source_or_checkpoint_expected_revision_drift(tmp_path: Path) -> None:
    for runner in (
        FakeUniDepthWorkerRunner(source_revision="x" * 40),
        FakeUniDepthWorkerRunner(checkpoint_expected_revision="y" * 40),
    ):
        root = tmp_path / ("source" if runner.source_revision.startswith("x") else "checkpoint")
        context, env = make_unidepth_context(root, view_count=3)
        adapter = UniDepthV2ProductionProviderAdapter(context=context, environment=env, runner=runner)
        request, decision = explicit_unidepth_selection(3)
        with pytest.raises(ReconstructionValidationError, match="revision"):
            adapter.execute(request, decision)


def test_i43e2_refuses_semantics_drift_and_direct_points_authority(tmp_path: Path) -> None:
    cases = (
        (FakeUniDepthWorkerRunner(native_depth_scale="relative", native_is_metric=False), "metric-depth"),
        (FakeUniDepthWorkerRunner(intrinsics_source_native="calibrated"), "intrinsics semantics"),
        (FakeUniDepthWorkerRunner(confidence_kind_native="confidence"), "uncertainty semantics"),
        (FakeUniDepthWorkerRunner(direct_points_native=False), "direct-points"),
        (FakeUniDepthWorkerRunner(direct_points_consumed_for_canonical_geometry=True), "canonical geometry authority"),
    )
    for index, (runner, message) in enumerate(cases):
        context, env = make_unidepth_context(tmp_path / f"case-{index}", view_count=3)
        adapter = UniDepthV2ProductionProviderAdapter(context=context, environment=env, runner=runner)
        request, decision = explicit_unidepth_selection(3)
        with pytest.raises(ReconstructionValidationError, match=message):
            adapter.execute(request, decision)


def test_i43e2_refuses_registered_source_mutation(tmp_path: Path) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=3)
    first = context.sfm_result.workspace.images[0]
    mutate_path = Path(context.sfm_result.workspace.workspace_root) / first.staged_image_path
    adapter = UniDepthV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeUniDepthWorkerRunner(mutate_path=mutate_path),
    )
    request, decision = explicit_unidepth_selection(3)
    with pytest.raises(ReconstructionValidationError, match="changed during provider execution"):
        adapter.execute(request, decision)


def test_i43e2_refuses_multiple_sfm_camera_models_for_single_fusion_profile(tmp_path: Path) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=3, camera_ids=("cam-a", "cam-b", "cam-a"))
    adapter = UniDepthV2ProductionProviderAdapter(context=context, environment=env, runner=FakeUniDepthWorkerRunner())
    request, decision = explicit_unidepth_selection(3)
    with pytest.raises(ReconstructionValidationError, match="one authoritative SfM camera model"):
        adapter.execute(request, decision)


def test_i43e2_execute_refuses_unavailable_runtime_wrong_provider_and_fallback(tmp_path: Path) -> None:
    context, _ = make_unidepth_context(tmp_path / "unavailable", view_count=3)
    adapter = UniDepthV2ProductionProviderAdapter(
        context=context,
        environment=good_unidepth_environment(tmp_path / "unavailable", available=False),
        runner=FakeUniDepthWorkerRunner(),
    )
    request, decision = explicit_unidepth_selection(3)
    with pytest.raises(ReconstructionValidationError, match="runtime is unavailable"):
        adapter.execute(request, decision)

    context, env = make_unidepth_context(tmp_path / "wrong", view_count=3)
    adapter = UniDepthV2ProductionProviderAdapter(context=context, environment=env, runner=FakeUniDepthWorkerRunner())
    request, decision = explicit_unidepth_selection(3, provider_id="wrong-provider")
    with pytest.raises(ReconstructionValidationError, match="provider ID"):
        adapter.execute(request, decision)

    context, env = make_unidepth_context(tmp_path / "fallback", view_count=3)
    adapter = UniDepthV2ProductionProviderAdapter(context=context, environment=env, runner=FakeUniDepthWorkerRunner())
    request, decision = explicit_unidepth_selection(3, allow_fallback=True)
    with pytest.raises(ReconstructionValidationError, match="silent provider fallback"):
        adapter.execute(request, decision)


def test_i43e2_refuses_runtime_variant_repository_or_checkpoint_mismatch(tmp_path: Path) -> None:
    fields = {
        "variant": "vitb14",
        "repository": "wrong/repository",
        "checkpoint_dir": "checkpoints/unidepth-v2/vitb14",
    }
    for index, (field, value) in enumerate(fields.items()):
        context, env = make_unidepth_context(tmp_path / f"runtime-{index}", view_count=3)
        runtime = dict(context.runtime_config)
        runtime[field] = value
        object.__setattr__(context, "runtime_config", tuple(sorted(runtime.items())))
        adapter = UniDepthV2ProductionProviderAdapter(context=context, environment=env, runner=FakeUniDepthWorkerRunner())
        request, decision = explicit_unidepth_selection(3)
        with pytest.raises(ReconstructionValidationError, match="runtime"):
            adapter.execute(request, decision)


def test_i43e2_summary_records_model_outputs_as_evidence_only(tmp_path: Path) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=3)
    adapter = UniDepthV2ProductionProviderAdapter(context=context, environment=env, runner=FakeUniDepthWorkerRunner())
    request, decision = explicit_unidepth_selection(3)
    adapter.execute(request, decision)
    summary = json.loads((context.output_dir / "i4_3e2_unidepth_v2_execution_summary.json").read_text(encoding="utf-8"))
    assert summary["authority"]["native_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert summary["authority"]["model_predicted_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["authority"]["model_direct_points_consumed_for_canonical_geometry"] is False
    assert summary["authority"]["metric_scale_claimed"] is False
    assert summary["integrity"]["canonical_pose_layer"] == "offline_sfm_bundle_adjusted"
    assert summary["alignment"]["accepted_views"] == 3


def test_i43e2_binding_passes_only_after_e1_and_d3_closure(tmp_path: Path) -> None:
    snapshot = validate_i4_3e2_binding(
        i4_3a_snapshot=architecture_snapshot(),
        i4_3d3_snapshot=d3_snapshot(),
        i4_3e1_snapshot=e1_snapshot(),
        environment=good_unidepth_environment(tmp_path),
    )
    assert snapshot["status"] == "pass"
    assert snapshot["provider"]["implementation"] == "unidepth_v2"
    assert snapshot["architecture"]["native_confidence_kind"] == "uncertainty"
    assert snapshot["authority"]["model_direct_points_consumed_for_canonical_geometry"] is False
    assert snapshot["qualification"]["predecessor_i4_3e1_verified"] is True
    assert "moge_2" in snapshot["qualification"]["remaining_i4_3e_providers_pending"]
    assert "unidepth_v2" not in snapshot["qualification"]["remaining_i4_3e_providers_pending"]


def test_i43e2_binding_fails_closed_without_e1_or_with_prior_authority_promotion(tmp_path: Path) -> None:
    bad_e1 = e1_snapshot()
    bad_e1["pass"] = False
    with pytest.raises(ReconstructionValidationError, match="I4.3E1"):
        validate_i4_3e2_binding(
            i4_3a_snapshot=architecture_snapshot(),
            i4_3d3_snapshot=d3_snapshot(),
            i4_3e1_snapshot=bad_e1,
            environment=good_unidepth_environment(tmp_path / "missing"),
        )

    promoted = e1_snapshot()
    promoted["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="promoted authority"):
        validate_i4_3e2_binding(
            i4_3a_snapshot=architecture_snapshot(),
            i4_3d3_snapshot=d3_snapshot(),
            i4_3e1_snapshot=promoted,
            environment=good_unidepth_environment(tmp_path / "promoted"),
        )


def test_i43e2_worker_import_is_model_and_cuda_lazy() -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/reconstruction_unidepth_v2_provider_worker.py"
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("i43e2_worker_probe", worker)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new_modules = set(sys.modules) - before
    for forbidden in ("torch", "cv2", "unidepth", "core.models.registry"):
        assert forbidden not in new_modules
    assert callable(module.main) and callable(module.run)
