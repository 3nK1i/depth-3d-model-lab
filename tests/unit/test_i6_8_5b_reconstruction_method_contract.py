"""I6.8.5B test-first safety and eligibility contracts, without any job launch."""
from dataclasses import FrozenInstanceError, replace

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_method_selection_contract import (
    MethodEligibilityContext, MethodImplementation, MethodOutput,
    MethodSource, MethodSpecification, ReconstructionMethod,
    assess_reconstruction_method, method_specification, reconstruction_method_catalog,
)


def _context(method, *, source=MethodSource.IMAGE_SEQUENCE, **kwargs):
    defaults = dict(source=source, source_ready=True,
                    runtime_verified_methods=(method,), sparse_admitted=True,
                    canonical_scene_available=True, surface_seed_verified=True)
    defaults.update(kwargs)
    return MethodEligibilityContext(**defaults)


def test_exact_catalog_methods_are_unique_explicit_and_frozen():
    catalog = reconstruction_method_catalog()
    assert len(catalog) == len(ReconstructionMethod) == 6
    assert {spec.method for spec in catalog} == set(ReconstructionMethod)
    assert all(spec.entrypoint for spec in catalog if spec.implementation is MethodImplementation.EXISTING_ACTION)
    assert all(not spec.entrypoint for spec in catalog if spec.implementation is not MethodImplementation.EXISTING_ACTION)
    with pytest.raises(FrozenInstanceError):
        catalog[0].entrypoint = 'unsafe'  # type: ignore[misc]


@pytest.mark.parametrize('method', list(ReconstructionMethod))
def test_no_method_is_enabled_from_unverified_default_context(method):
    result = assess_reconstruction_method(method, MethodEligibilityContext(MethodSource.NONE))
    assert not result.executable_via_existing_action
    assert result.reason_codes
    assert not result.metric_scale_qualified_by_selection
    assert not result.canonical_camera_authority_changed


@pytest.mark.parametrize('method', list(ReconstructionMethod))
def test_no_method_can_run_concurrently(method):
    result = assess_reconstruction_method(method, _context(method, active_job=True))
    assert not result.executable_via_existing_action
    assert 'another_job_active' in result.reason_codes


def test_colmap_requires_admitted_sparse_runtime_and_supported_source():
    method = ReconstructionMethod.COLMAP_MVS
    assert assess_reconstruction_method(method, _context(method)).executable_via_existing_action
    missing = assess_reconstruction_method(method, _context(method, sparse_admitted=False,
                                                             runtime_verified_methods=()))
    assert missing.reason_codes == ('admitted_sparse_required', 'runtime_not_verified')
    wrong = assess_reconstruction_method(method, _context(method, source=MethodSource.LIVE_SOURCE))
    assert 'incompatible_source_workflow' in wrong.reason_codes


def test_da3_quick_is_distinct_from_full_scene_and_no_sfm_required():
    quick = ReconstructionMethod.DA3_QUICK
    assert assess_reconstruction_method(quick, _context(quick, sparse_admitted=False)).executable_via_existing_action
    assert method_specification(quick).output is MethodOutput.SUPPORTING_DEPTH_GEOMETRY
    assert 'default 3' in method_specification(quick).limitation
    full = ReconstructionMethod.DA3_FULL_SCENE
    assert not assess_reconstruction_method(full, _context(full)).executable_via_existing_action
    assert 'incompatible_source_workflow' in assess_reconstruction_method(full, _context(full)).reason_codes
    assert assess_reconstruction_method(full, _context(full, source=MethodSource.RECORDED_VIDEO)).executable_via_existing_action
    assert method_specification(full).output is MethodOutput.FULL_SCENE_GEOMETRY


@pytest.mark.parametrize('method,blocking_reason', [
    (ReconstructionMethod.VOXEL_POINT_FUSION, 'no_standalone_entrypoint'),
    (ReconstructionMethod.GAUSSIAN_TRAINING, 'ui_execution_integration_pending'),
    (ReconstructionMethod.TSDF, 'backend_implementation_required'),
])
def test_unwired_and_future_methods_never_advertise_executable(method, blocking_reason):
    result = assess_reconstruction_method(method, _context(method))
    assert not result.executable_via_existing_action
    assert blocking_reason in result.reason_codes


def test_tsdf_is_mandatory_i7_and_not_voxel_point_fusion():
    tsdf = method_specification(ReconstructionMethod.TSDF)
    voxel = method_specification(ReconstructionMethod.VOXEL_POINT_FUSION)
    assert tsdf.milestone == 'I7'
    assert tsdf.output is MethodOutput.TSDF_SURFACE_MESH
    assert tsdf.implementation is MethodImplementation.IMPLEMENTATION_REQUIRED
    assert voxel.output is MethodOutput.VOXEL_FUSED_POINTS
    assert 'not TSDF' in voxel.limitation


def test_gaussian_requires_canonical_scene_and_verified_surface_seed_but_still_unwired():
    method = ReconstructionMethod.GAUSSIAN_TRAINING
    result = assess_reconstruction_method(method, _context(method, canonical_scene_available=False,
                                                             surface_seed_verified=False))
    assert {'canonical_scene_required', 'verified_surface_seed_required'} <= set(result.reason_codes)
    assert not assess_reconstruction_method(method, _context(method)).executable_via_existing_action
    assert method_specification(method).output is MethodOutput.GAUSSIAN_APPEARANCE


def test_no_implicit_auto_fallback_or_unknown_method():
    with pytest.raises(ReconstructionValidationError, match='explicit'):
        method_specification('auto')  # type: ignore[arg-type]
    with pytest.raises(ReconstructionValidationError, match='explicit'):
        assess_reconstruction_method('da3', _context(ReconstructionMethod.DA3_QUICK))  # type: ignore[arg-type]


def test_runtime_readiness_is_per_method_and_not_gpu_inferred():
    full = ReconstructionMethod.DA3_FULL_SCENE
    ctx = _context(full, source=MethodSource.RECORDED_VIDEO,
                   runtime_verified_methods=(ReconstructionMethod.DA3_QUICK,))
    result = assess_reconstruction_method(full, ctx)
    assert not result.executable_via_existing_action
    assert 'runtime_not_verified' in result.reason_codes


def test_catalog_does_not_offer_stage_pause_or_resume():
    assert all(not spec.pause_supported and not spec.resume_supported for spec in reconstruction_method_catalog())
    base = method_specification(ReconstructionMethod.COLMAP_MVS)
    with pytest.raises(ReconstructionValidationError, match='I6.9'):
        replace(base, pause_supported=True)
    with pytest.raises(ReconstructionValidationError, match='I6.9'):
        replace(base, resume_supported=True)


def test_pending_methods_cannot_falsely_advertise_entrypoint_or_cancellation():
    future = method_specification(ReconstructionMethod.TSDF)
    with pytest.raises(ReconstructionValidationError, match='entry point'):
        replace(future, entrypoint='core.reconstruction.dense_fusion')
    with pytest.raises(ReconstructionValidationError, match='lifecycle'):
        replace(future, cancellation_supported=True)


def test_verified_runtime_does_not_override_missing_input_or_prerequisites():
    method = ReconstructionMethod.COLMAP_MVS
    empty = assess_reconstruction_method(method, _context(method, source_ready=False, sparse_admitted=False))
    assert not empty.executable_via_existing_action
    assert 'source_not_ready' in empty.reason_codes
    assert 'admitted_sparse_required' in empty.reason_codes


def test_contract_import_is_dependency_light_and_never_links_legacy_launchers():
    import core.reconstruction.reconstruction_method_selection_contract as contract
    from pathlib import Path
    source = Path(contract.__file__).read_text()
    assert 'PyQt6' not in source and 'import torch' not in source
    assert 'subprocess' not in source and 'QProcess' not in source
    assert 'def start(' not in source and 'def launch(' not in source
