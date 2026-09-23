"""I6.8.5B: read-only reconstruction *workflow* catalog and eligibility contract.

This is deliberately distinct from the I6.4 dense-provider intent/Auto catalog:
a workflow includes an existing entry point, prerequisites, and an output kind;
it does not execute work, authorize artifacts, choose fallback providers, or
claim metric scale. I6.8.5C/D may consume the assessment when wiring UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.reconstruction.errors import ReconstructionValidationError


class ReconstructionMethod(str, Enum):
    COLMAP_MVS = 'colmap_mvs'
    DA3_QUICK = 'da3_quick'
    DA3_FULL_SCENE = 'da3_full_scene'
    VOXEL_POINT_FUSION = 'voxel_point_fusion'
    GAUSSIAN_TRAINING = 'gaussian_training'
    TSDF = 'tsdf'


class MethodSource(str, Enum):
    NONE = 'none'
    IMAGE_SEQUENCE = 'image_sequence'
    RECORDED_VIDEO = 'recorded_video'
    LIVE_SOURCE = 'live_source'


class MethodImplementation(str, Enum):
    EXISTING_ACTION = 'existing_action'
    INTERNAL_ONLY = 'internal_only'
    INTEGRATION_PENDING = 'integration_pending'
    IMPLEMENTATION_REQUIRED = 'implementation_required'


class MethodOutput(str, Enum):
    DENSE_POINTS = 'dense_points'
    SUPPORTING_DEPTH_GEOMETRY = 'supporting_depth_geometry'
    FULL_SCENE_GEOMETRY = 'full_scene_geometry'
    VOXEL_FUSED_POINTS = 'voxel_fused_points'
    GAUSSIAN_APPEARANCE = 'gaussian_appearance'
    TSDF_SURFACE_MESH = 'tsdf_surface_mesh'


@dataclass(frozen=True, slots=True)
class MethodSpecification:
    method: ReconstructionMethod
    label: str
    implementation: MethodImplementation
    sources: tuple[MethodSource, ...]
    output: MethodOutput
    entrypoint: str
    requires_sparse: bool = False
    requires_canonical_scene: bool = False
    requires_surface_seed: bool = False
    cancellation_supported: bool = False
    # An opaque backend's in-progress stage is not automatically resumable.
    pause_supported: bool = False
    resume_supported: bool = False
    milestone: str = 'I6.8.5'
    limitation: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.method, ReconstructionMethod):
            raise TypeError('method must be a ReconstructionMethod.')
        if not isinstance(self.implementation, MethodImplementation):
            raise TypeError('implementation must be a MethodImplementation.')
        if not isinstance(self.output, MethodOutput):
            raise TypeError('output must be a MethodOutput.')
        if not self.label.strip() or not self.milestone.strip():
            raise ReconstructionValidationError('Method label and milestone must be non-empty.')
        if not self.sources or len(set(self.sources)) != len(self.sources) or any(
            not isinstance(source, MethodSource) or source is MethodSource.NONE
            for source in self.sources
        ):
            raise ReconstructionValidationError('Method sources must be concrete, unique workflow types.')
        if self.implementation is MethodImplementation.EXISTING_ACTION and not self.entrypoint.strip():
            raise ReconstructionValidationError('An existing action must identify its existing entry point.')
        if self.implementation is not MethodImplementation.EXISTING_ACTION and self.entrypoint:
            raise ReconstructionValidationError('Pending/internal methods cannot advertise a selectable entry point.')
        if self.implementation is not MethodImplementation.EXISTING_ACTION and (
            self.cancellation_supported or self.pause_supported or self.resume_supported
        ):
            raise ReconstructionValidationError('Unwired methods cannot advertise lifecycle support.')
        if self.pause_supported or self.resume_supported:
            raise ReconstructionValidationError('I6.9 must qualify pause/resume explicitly before advertising it.')


# Existing entry points are documentation references, NOT dynamic imports/dispatch.
# In particular, voxel point fusion is not a TSDF integrator, and the existing
# Gaussian optimization module is not a selectable Reconstruction-page job.
_METHODS: tuple[MethodSpecification, ...] = (
    MethodSpecification(
        ReconstructionMethod.COLMAP_MVS, 'COLMAP MVS',
        MethodImplementation.EXISTING_ACTION,
        (MethodSource.IMAGE_SEQUENCE, MethodSource.RECORDED_VIDEO),
        MethodOutput.DENSE_POINTS,
        'app.reconstruction_dense_job_controller.ExplicitDenseJobController.start',
        requires_sparse=True, cancellation_supported=True,
        limitation='Requires the completed admitted sparse session; Auto is not production-enabled.',
    ),
    MethodSpecification(
        ReconstructionMethod.DA3_QUICK, 'DA3 quick integration',
        MethodImplementation.EXISTING_ACTION,
        (MethodSource.IMAGE_SEQUENCE, MethodSource.RECORDED_VIDEO),
        MethodOutput.SUPPORTING_DEPTH_GEOMETRY,
        'app.da3_canonical_integration_controller.DA3CanonicalIntegrationController.start',
        cancellation_supported=True,
        limitation='Bounded 2–8-frame integration, default 3; not full 57-image coverage.',
    ),
    MethodSpecification(
        ReconstructionMethod.DA3_FULL_SCENE, 'DA3 full scene',
        MethodImplementation.EXISTING_ACTION,
        (MethodSource.RECORDED_VIDEO,),
        MethodOutput.FULL_SCENE_GEOMETRY,
        'app.reconstruction_da3_full_scene.build_full_scene_da3_job',
        cancellation_supported=True,
        limitation='Current full-scene action accepts recorded video only; image-directory wiring is pending.',
    ),
    MethodSpecification(
        ReconstructionMethod.VOXEL_POINT_FUSION, 'Predicted depth + voxel point fusion',
        MethodImplementation.INTERNAL_ONLY,
        (MethodSource.IMAGE_SEQUENCE, MethodSource.RECORDED_VIDEO),
        MethodOutput.VOXEL_FUSED_POINTS, '',
        requires_canonical_scene=True,
        limitation='Existing bounded point accumulator is not a standalone user job and is not TSDF.',
    ),
    MethodSpecification(
        ReconstructionMethod.GAUSSIAN_TRAINING, 'Gaussian appearance training',
        MethodImplementation.INTEGRATION_PENDING,
        (MethodSource.IMAGE_SEQUENCE, MethodSource.RECORDED_VIDEO),
        MethodOutput.GAUSSIAN_APPEARANCE, '',
        requires_canonical_scene=True, requires_surface_seed=True,
        limitation='Optimization and persisted viewer exist; Reconstruction-page training launch is not wired.',
    ),
    MethodSpecification(
        ReconstructionMethod.TSDF, 'True depth prediction + TSDF fusion',
        MethodImplementation.IMPLEMENTATION_REQUIRED,
        (MethodSource.IMAGE_SEQUENCE, MethodSource.RECORDED_VIDEO),
        MethodOutput.TSDF_SURFACE_MESH, '',
        requires_canonical_scene=True,
        milestone='I7',
        limitation='Mandatory I7 implementation and qualification: signed-distance integration, surface extraction, canonical provenance; voxel point fusion does not satisfy it.',
    ),
)


def reconstruction_method_catalog() -> tuple[MethodSpecification, ...]:
    """Stable inventory; must not be confused with the I6.4 provider catalog."""
    return _METHODS


def method_specification(method: ReconstructionMethod) -> MethodSpecification:
    if not isinstance(method, ReconstructionMethod):
        raise ReconstructionValidationError('Choose one explicit known reconstruction method; Auto is not enabled.')
    for candidate in _METHODS:
        if candidate.method is method:
            return candidate
    raise ReconstructionValidationError('Unknown reconstruction method.')


@dataclass(frozen=True, slots=True)
class MethodEligibilityContext:
    """Caller-verified facts; the contract does not probe GPU/files or start jobs."""
    source: MethodSource
    source_ready: bool = False
    sparse_admitted: bool = False
    canonical_scene_available: bool = False
    surface_seed_verified: bool = False
    active_job: bool = False
    runtime_verified_methods: tuple[ReconstructionMethod, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, MethodSource):
            raise TypeError('source must be MethodSource.')
        if len(set(self.runtime_verified_methods)) != len(self.runtime_verified_methods) or any(
            not isinstance(method, ReconstructionMethod) for method in self.runtime_verified_methods
        ):
            raise ReconstructionValidationError('Verified runtime methods must be unique known methods.')


@dataclass(frozen=True, slots=True)
class MethodEligibility:
    method: ReconstructionMethod
    executable_via_existing_action: bool
    reason_codes: tuple[str, ...]
    output: MethodOutput
    # No method selection or output creates camera, scale, QC or metrology authority.
    metric_scale_qualified_by_selection: bool = False
    canonical_camera_authority_changed: bool = False


def assess_reconstruction_method(
    method: ReconstructionMethod, context: MethodEligibilityContext,
) -> MethodEligibility:
    """Fail-closed *advisory* eligibility; C/D must recheck at actual launch."""
    if not isinstance(context, MethodEligibilityContext):
        raise TypeError('context must be MethodEligibilityContext.')
    spec = method_specification(method)
    reasons: list[str] = []
    if spec.implementation is not MethodImplementation.EXISTING_ACTION:
        reasons.append({
            MethodImplementation.INTERNAL_ONLY: 'no_standalone_entrypoint',
            MethodImplementation.INTEGRATION_PENDING: 'ui_execution_integration_pending',
            MethodImplementation.IMPLEMENTATION_REQUIRED: 'backend_implementation_required',
        }[spec.implementation])
    if context.active_job:
        reasons.append('another_job_active')
    if not context.source_ready or context.source is MethodSource.NONE:
        reasons.append('source_not_ready')
    elif context.source not in spec.sources:
        reasons.append('incompatible_source_workflow')
    if spec.requires_sparse and not context.sparse_admitted:
        reasons.append('admitted_sparse_required')
    if spec.requires_canonical_scene and not context.canonical_scene_available:
        reasons.append('canonical_scene_required')
    if spec.requires_surface_seed and not context.surface_seed_verified:
        reasons.append('verified_surface_seed_required')
    if spec.implementation is MethodImplementation.EXISTING_ACTION and (
        method not in context.runtime_verified_methods
    ):
        reasons.append('runtime_not_verified')
    return MethodEligibility(method, not reasons, tuple(reasons), spec.output)
