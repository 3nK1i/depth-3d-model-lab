"""I6.8.5C routing: selection never changes the actual method or authority."""
from dataclasses import replace
from pathlib import Path

import pytest

from app.reconstruction_method_execution import start_eligible_reconstruction_method
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_method_selection_contract import (
    MethodEligibilityContext, MethodSource, ReconstructionMethod,
)


def _context(method: ReconstructionMethod, *, source=MethodSource.RECORDED_VIDEO, **overrides):
    values = dict(
        source=source, source_ready=True, sparse_admitted=True,
        canonical_scene_available=False, surface_seed_verified=False,
        active_job=False, runtime_verified_methods=(method,),
    )
    values.update(overrides)
    return MethodEligibilityContext(**values)


def _launch(method, context):
    launched = []
    def job(name):
        def launch():
            launched.append(name)
            return f'{name}-run-id'
        return launch
    result = start_eligible_reconstruction_method(
        method, context=context,
        start_colmap_mvs=job('colmap_mvs'),
        start_da3_quick=job('da3_quick'),
        start_da3_full_scene=job('da3_full_scene'),
    )
    return result, launched


@pytest.mark.parametrize('method,source', [
    (ReconstructionMethod.COLMAP_MVS, MethodSource.IMAGE_SEQUENCE),
    (ReconstructionMethod.COLMAP_MVS, MethodSource.RECORDED_VIDEO),
    (ReconstructionMethod.DA3_QUICK, MethodSource.IMAGE_SEQUENCE),
    (ReconstructionMethod.DA3_QUICK, MethodSource.RECORDED_VIDEO),
    (ReconstructionMethod.DA3_FULL_SCENE, MethodSource.RECORDED_VIDEO),
])
def test_exact_method_calls_exactly_one_existing_action(method, source):
    result, launched = _launch(method, _context(method, source=source))
    assert result == f'{method.value}-run-id'
    assert launched == [method.value]


@pytest.mark.parametrize('method,source,change,reason', [
    (ReconstructionMethod.COLMAP_MVS, MethodSource.IMAGE_SEQUENCE, {'sparse_admitted': False}, 'admitted_sparse_required'),
    (ReconstructionMethod.DA3_QUICK, MethodSource.RECORDED_VIDEO, {'source_ready': False}, 'source_not_ready'),
    (ReconstructionMethod.DA3_QUICK, MethodSource.RECORDED_VIDEO, {'active_job': True}, 'another_job_active'),
    (ReconstructionMethod.DA3_FULL_SCENE, MethodSource.IMAGE_SEQUENCE, {}, 'incompatible_source_workflow'),
    (ReconstructionMethod.DA3_FULL_SCENE, MethodSource.LIVE_SOURCE, {}, 'incompatible_source_workflow'),
    (ReconstructionMethod.DA3_QUICK, MethodSource.IMAGE_SEQUENCE, {'runtime_verified_methods': ()}, 'runtime_not_verified'),
    (ReconstructionMethod.COLMAP_MVS, MethodSource.IMAGE_SEQUENCE, {'runtime_verified_methods': (ReconstructionMethod.DA3_QUICK,)}, 'runtime_not_verified'),
    (ReconstructionMethod.DA3_FULL_SCENE, MethodSource.RECORDED_VIDEO, {'runtime_verified_methods': (ReconstructionMethod.DA3_QUICK,)}, 'runtime_not_verified'),
])
def test_ineligible_never_invokes_any_backend(method, source, change, reason):
    called = []
    def should_never_run():
        called.append(True)
        raise AssertionError('unsafe launch')
    with pytest.raises(ReconstructionValidationError, match=reason):
        start_eligible_reconstruction_method(
            method, context=_context(method, source=source, **change),
            start_colmap_mvs=should_never_run,
            start_da3_quick=should_never_run,
            start_da3_full_scene=should_never_run,
        )
    assert called == []


@pytest.mark.parametrize('method,expected', [
    (ReconstructionMethod.VOXEL_POINT_FUSION, 'no_standalone_entrypoint'),
    (ReconstructionMethod.GAUSSIAN_TRAINING, 'ui_execution_integration_pending'),
    (ReconstructionMethod.TSDF, 'backend_implementation_required'),
])
def test_unwired_methods_never_fall_back_to_mvs(method, expected):
    calls = []
    def forbidden():
        calls.append('should-not-run')
        return 'bad'
    with pytest.raises(ReconstructionValidationError, match=expected):
        start_eligible_reconstruction_method(
            method, context=_context(method, canonical_scene_available=True,
                                     surface_seed_verified=True),
            start_colmap_mvs=forbidden, start_da3_quick=forbidden,
            start_da3_full_scene=forbidden,
        )
    assert calls == []


def test_auto_or_unknown_method_is_rejected_without_side_effects():
    calls = []
    def launch():
        calls.append('unexpected')
        return 'bad'
    with pytest.raises(ReconstructionValidationError, match='explicit'):
        start_eligible_reconstruction_method(
            'auto', context=_context(ReconstructionMethod.COLMAP_MVS),  # type: ignore[arg-type]
            start_colmap_mvs=launch, start_da3_quick=launch,
            start_da3_full_scene=launch,
        )
    assert calls == []


def test_launcher_error_is_not_retried_with_another_method():
    calls = []
    def failed():
        calls.append('da3_full_scene')
        raise RuntimeError('DA3 qualification failed')
    def wrong():
        calls.append('other')
        return 'incorrect'
    with pytest.raises(RuntimeError, match='DA3 qualification failed'):
        start_eligible_reconstruction_method(
            ReconstructionMethod.DA3_FULL_SCENE,
            context=_context(ReconstructionMethod.DA3_FULL_SCENE),
            start_colmap_mvs=wrong, start_da3_quick=wrong,
            start_da3_full_scene=failed,
        )
    assert calls == ['da3_full_scene']


def test_production_bridge_is_pure_and_page_binds_existing_launchers():
    root = Path(__file__).resolve().parents[2]
    bridge = (root / 'app/reconstruction_method_execution.py').read_text()
    page = (root / 'app/widgets/reconstruction_page.py').read_text()
    assert 'PyQt6' not in bridge and 'subprocess' not in bridge
    assert 'start_eligible_reconstruction_method(' in page
    assert 'controller.start_explicit_post_sparse_dense()' in page
    assert 'start_da3_quick=controller.start_selected_da3_canonical' in page
    assert 'spec, command = build_full_scene_da3_job(' in page
    assert 'return self.start_digital_twin_processing_job(spec, command)' in page
    assert 'surface_seed_verified=False' in page
    assert 'runtime_verified_methods=runtime_verified_methods' in page
    assert 'QPushButton("Build Full Scene")' in page
    assert 'QPushButton("Integrate DA3")' in page
    assert "QPushButton('Build dense (COLMAP MVS)')" in page
