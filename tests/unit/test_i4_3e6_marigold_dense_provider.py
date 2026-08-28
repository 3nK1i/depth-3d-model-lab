from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from core.contracts.validation import metadata_as_dict
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.marigold_dense_provider import (
    I43E6_PROVIDER_ID,
    MarigoldProductionProviderAdapter,
    align_marigold_relative_depth_to_sfm,
    marigold_dense_provider_descriptor,
    validate_i4_3e6_binding,
)
from tests.i4_3e6_helpers import (
    FakeMarigoldWorkerRunner,
    architecture_snapshot,
    d3_snapshot,
    e1_snapshot,
    e2_snapshot,
    e3_snapshot,
    e4_snapshot,
    e5_snapshot,
    explicit_marigold_selection,
    good_marigold_environment,
    make_marigold_context,
)


def _binding(tmp_path: Path, **overrides):
    values = dict(
        i4_3a_snapshot=architecture_snapshot(),
        i4_3d3_snapshot=d3_snapshot(),
        i4_3e1_snapshot=e1_snapshot(),
        i4_3e2_snapshot=e2_snapshot(),
        i4_3e3_snapshot=e3_snapshot(),
        i4_3e4_snapshot=e4_snapshot(),
        i4_3e5_snapshot=e5_snapshot(),
        environment=good_marigold_environment(tmp_path),
    )
    values.update(overrides)
    return validate_i4_3e6_binding(**values)


def test_i43e6_descriptor_preserves_marigold_semantics(tmp_path: Path) -> None:
    descriptor = marigold_dense_provider_descriptor(good_marigold_environment(tmp_path))
    assert descriptor.provider_id == I43E6_PROVIDER_ID
    assert descriptor.implementation.value == "marigold"
    assert descriptor.qualification_state.value == "candidate"
    metadata = metadata_as_dict(descriptor.metadata)
    assert metadata["native_depth_scale"] == "relative"
    assert metadata["native_confidence_kind"] == "uncertainty_optional"
    assert metadata["permanent_global_view_cap"] is None
    assert metadata["relative_depth_promoted_to_metric_scale"] is False


@pytest.mark.parametrize("prediction_type", ["depth", "disparity"])
def test_i43e6_alignment_uses_correct_affine_domain(tmp_path: Path, prediction_type: str) -> None:
    context, _ = make_marigold_context(tmp_path, view_count=3)
    runner = FakeMarigoldWorkerRunner(context.sfm_result, prediction_type=prediction_type)
    depth = runner._relative_depth_for_keyframe(context.sfm_result.cameras[0].keyframe_id, 48, 64)
    aligned, valid, fit = align_marigold_relative_depth_to_sfm(
        context.sfm_result,
        keyframe_id=context.sfm_result.cameras[0].keyframe_id,
        relative_depth=depth,
        prediction_type=prediction_type,
        source_valid_mask=np.ones_like(depth, dtype=bool),
    )
    assert fit.domain == ("depth" if prediction_type == "depth" else "inverse_depth")
    assert fit.slope > 0.0
    assert fit.p90_relative_residual < 1e-3
    assert np.count_nonzero(valid) > 0
    assert np.isfinite(aligned[valid]).all()


def test_i43e6_alignment_rejects_semantic_and_grid_drift(tmp_path: Path) -> None:
    context, _ = make_marigold_context(tmp_path, view_count=3)
    depth = np.ones((48, 64), dtype=np.float32)
    with pytest.raises(ReconstructionValidationError, match="prediction_type"):
        align_marigold_relative_depth_to_sfm(
            context.sfm_result,
            keyframe_id=context.sfm_result.cameras[0].keyframe_id,
            relative_depth=depth,
            prediction_type="metric",
        )
    with pytest.raises(ReconstructionValidationError, match="pixel grid"):
        align_marigold_relative_depth_to_sfm(
            context.sfm_result,
            keyframe_id=context.sfm_result.cameras[0].keyframe_id,
            relative_depth=np.ones((47, 64), dtype=np.float32),
            prediction_type="depth",
        )


def test_i43e6_provider_accepts_optional_uncertainty_without_promoting_it(tmp_path: Path) -> None:
    context, env = make_marigold_context(tmp_path, view_count=3)
    adapter = MarigoldProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMarigoldWorkerRunner(context.sfm_result, uncertainty=True),
    )
    request, decision = explicit_marigold_selection(3)
    result = adapter.execute(request, decision)
    assert result.metric_scale_claimed is False
    assert result.measurement_authoritative is False
    assert result.permanent_global_view_cap is None


def test_i43e6_provider_accepts_no_uncertainty_fast_profile(tmp_path: Path) -> None:
    context, env = make_marigold_context(tmp_path, view_count=3)
    adapter = MarigoldProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMarigoldWorkerRunner(context.sfm_result, uncertainty=False),
    )
    request, decision = explicit_marigold_selection(3)
    result = adapter.execute(request, decision)
    assert result.source_view_count == 3
    assert result.provider_promotion_performed is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"native_depth_scale": "metric"}, "relative-depth semantics"),
        ({"depth_representation_native": "metric_depth"}, "affine-invariant representation"),
        ({"confidence_kind_native": "confidence"}, "uncertainty semantics"),
        ({"direct_points_native": True}, "unsupported native Marigold geometry"),
        ({"normals_native": True}, "unsupported native Marigold geometry"),
        ({"adapter_projected_points_discarded": False}, "did not explicitly discard"),
        ({"promote_native_authority": True}, "promote Marigold evidence authority"),
    ],
)
def test_i43e6_refuses_worker_semantics_or_authority_drift(tmp_path: Path, kwargs: dict, match: str) -> None:
    context, env = make_marigold_context(tmp_path, view_count=3)
    adapter = MarigoldProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMarigoldWorkerRunner(context.sfm_result, **kwargs),
    )
    request, decision = explicit_marigold_selection(3)
    with pytest.raises(ReconstructionValidationError, match=match):
        adapter.execute(request, decision)


def test_i43e6_refuses_checkpoint_revision_drift(tmp_path: Path) -> None:
    context, env = make_marigold_context(tmp_path, view_count=3)
    adapter = MarigoldProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMarigoldWorkerRunner(context.sfm_result, checkpoint_revision="d" * 40),
    )
    request, decision = explicit_marigold_selection(3)
    with pytest.raises(ReconstructionValidationError, match="checkpoint revision drifted"):
        adapter.execute(request, decision)


def test_i43e6_refuses_source_image_mutation(tmp_path: Path) -> None:
    context, env = make_marigold_context(tmp_path, view_count=3)
    root = Path(context.sfm_result.workspace.workspace_root)
    first = sorted(context.sfm_result.workspace.images, key=lambda item: item.order)[0]
    mutate_path = root / first.staged_image_path
    adapter = MarigoldProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMarigoldWorkerRunner(context.sfm_result, mutate_path=mutate_path),
    )
    request, decision = explicit_marigold_selection(3)
    with pytest.raises(ReconstructionValidationError, match="changed during provider execution"):
        adapter.execute(request, decision)


def test_i43e6_refuses_multi_camera_and_fallback(tmp_path: Path) -> None:
    context, env = make_marigold_context(tmp_path / "multi", view_count=3, camera_ids=("a", "b", "a"))
    adapter = MarigoldProductionProviderAdapter(context=context, environment=env, runner=FakeMarigoldWorkerRunner(context.sfm_result))
    request, decision = explicit_marigold_selection(3)
    with pytest.raises(ReconstructionValidationError, match="one authoritative SfM camera model"):
        adapter.execute(request, decision)

    context, env = make_marigold_context(tmp_path / "fallback", view_count=3)
    adapter = MarigoldProductionProviderAdapter(context=context, environment=env, runner=FakeMarigoldWorkerRunner(context.sfm_result))
    request, decision = explicit_marigold_selection(3, allow_fallback=True)
    with pytest.raises(ReconstructionValidationError, match="forbids silent provider fallback"):
        adapter.execute(request, decision)


def test_i43e6_binding_closes_provider_adapter_set_without_promotion(tmp_path: Path) -> None:
    snapshot = _binding(tmp_path)
    assert snapshot["pass"] is True
    assert snapshot["provider"]["implementation"] == "marigold"
    assert snapshot["provider"]["qualification_state"] == "candidate"
    assert snapshot["architecture"]["native_depth_scale"] == "relative"
    assert snapshot["architecture"]["native_confidence_kind"] == "uncertainty_optional"
    assert snapshot["qualification"]["predecessor_i4_3e5_verified"] is True
    assert snapshot["qualification"]["remaining_i4_3e_providers_pending"] == []
    assert snapshot["qualification"]["i4_3e_provider_adapters_complete"] is True
    assert snapshot["authority"]["provider_promoted"] is False
    assert snapshot["next"].startswith("I4.3F")


def test_i43e6_binding_requires_closed_e5_and_clean_authority(tmp_path: Path) -> None:
    bad = e5_snapshot(); bad["authority"]["measurement_authoritative"] = True
    with pytest.raises(ReconstructionValidationError, match="I4.3E5 predecessor"):
        _binding(tmp_path, i4_3e5_snapshot=bad)
    unavailable = good_marigold_environment(tmp_path / "bad", available=False)
    with pytest.raises(ReconstructionValidationError, match="runtime is not ready"):
        _binding(tmp_path / "bad", environment=unavailable)


def test_i43e6_worker_import_is_lazy() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = root / "scripts/reconstruction_marigold_provider_worker.py"
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("i43e6_worker_lazy_probe", worker)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new_modules = set(sys.modules) - before
    for forbidden in ("torch", "cv2", "diffusers", "core.models.registry"):
        assert forbidden not in new_modules
    assert callable(module.run)
