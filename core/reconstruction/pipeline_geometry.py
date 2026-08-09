"""Phase 8.7F3 canonical geometry, semantics and inspection pipeline adapters.

This module binds already-qualified reconstruction components to the neutral F1
orchestrator without changing their authority semantics.  Depth/camera outputs
remain supporting evidence; dense geometry and the surface mesh are canonical;
semantic geometry and inspection/QC retain their own non-metrology authority.

Heavy DA3 runtime dependencies are intentionally absent.  Qualified DA3 NPZ/JSON
artifacts can be converted into dependency-light evidence through helpers in
this module and then consumed by the in-process canonical geometry stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
)
from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.models.metadata import IntrinsicsSource
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.camera_geometry import (
    CameraGeometryMethod,
    CameraGeometryMode,
    CameraGeometryProfile,
)
from core.reconstruction.da3 import (
    DA3_INTERNAL_POSE_CONVENTION,
    DA3_PROVIDER_ID,
    DA3ResultManifest,
    validate_da3_npz_result,
)
from core.reconstruction.dense_fusion import (
    DenseDepthAlignment,
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseFusionResult,
    DenseSceneProfile,
    explicit_depth_scale_alignment,
    fuse_dense_frames,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.geometry_quality import (
    DenseGeometryEvidence,
    DenseGeometryQualityPolicy,
    DenseGeometryQualityReport,
    assess_dense_geometry_quality,
)
from core.reconstruction.inspection_qc import (
    InspectionReport,
    ReferenceAssembly,
    RegistrationInspectionPolicy,
    inspect_semantic_geometry,
)
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineArtifact,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunResult,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionStageExecutionContext,
)
from core.reconstruction.scale_reference import ScaleReferenceProfile
from core.reconstruction.semantic_geometry import (
    SemanticAssociationPolicy,
    SemanticFrameObservation,
    SemanticGeometryResult,
    SemanticGeometryRole,
    build_semantic_3d_geometry,
)
from core.reconstruction.sfm_coordinates import validate_rigid_matrix
from core.reconstruction.surface_mesh import (
    SurfaceMeshResult,
    SurfaceMeshingPolicy,
    generate_quality_aware_surface_mesh,
)
from core.reconstruction.validation import require_nonnegative_int, require_uuid

PIPELINE_GEOMETRY_SCHEMA_VERSION = 1
PIPELINE_EVIDENCE_PROVIDER_ID = "phase8-f3-bound-evidence"
PIPELINE_DENSE_PROVIDER_ID = "canonical-dense-fusion"
PIPELINE_SURFACE_PROVIDER_ID = "quality-aware-surface-mesh"
PIPELINE_SEMANTICS_PROVIDER_ID = "semantic-geometry-core"
PIPELINE_INSPECTION_PROVIDER_ID = "registration-inspection-qc"
PIPELINE_F3_ADAPTER_VERSION = "phase8-7f3-1"

_T = TypeVar("_T")


def _readonly(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    array.setflags(write=False)
    return array


def _frame_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(require_nonempty(value, field_name="frame_id") for value in values)
    if not result or len(set(result)) != len(result):
        raise ReconstructionValidationError("Pipeline evidence requires unique non-empty frame IDs.")
    return result


@dataclass(frozen=True, slots=True)
class PipelineIngestionEvidence:
    """RGB/input evidence used to materialize canonical dense-fusion frames."""

    session_id: str
    frame_ids: tuple[str, ...]
    timestamps_ns: tuple[int, ...]
    rgb_frames: tuple[np.ndarray, ...]
    camera_profile: CameraGeometryProfile
    scale_profile: ScaleReferenceProfile
    reconstruction_frame_id: str
    scene_profile: DenseSceneProfile
    source_label: str = ""
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = PIPELINE_GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        ids = _frame_ids(self.frame_ids)
        timestamps = tuple(require_nonnegative_int(v, field_name="timestamp_ns") for v in self.timestamps_ns)
        frames = tuple(_readonly(v, np.uint8) for v in self.rgb_frames)
        if len(ids) != len(timestamps) or len(ids) != len(frames):
            raise ReconstructionValidationError("Ingestion frame IDs/timestamps/RGB arrays must have equal length.")
        if not isinstance(self.camera_profile, CameraGeometryProfile):
            raise TypeError("camera_profile must be CameraGeometryProfile.")
        if not isinstance(self.scale_profile, ScaleReferenceProfile):
            raise TypeError("scale_profile must be ScaleReferenceProfile.")
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        size = self.camera_profile.camera.image_size
        for frame in frames:
            if frame.ndim != 3 or frame.shape != (size.height, size.width, 3):
                raise ReconstructionValidationError("RGB evidence must match the declared camera pixel space.")
        if int(self.schema_version) != PIPELINE_GEOMETRY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported F3 pipeline-geometry schema version.")
        object.__setattr__(self, "frame_ids", ids)
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "rgb_frames", frames)
        object.__setattr__(
            self,
            "reconstruction_frame_id",
            require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id"),
        )
        object.__setattr__(self, "source_label", str(self.source_label).strip())
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        object.__setattr__(self, "schema_version", PIPELINE_GEOMETRY_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        size = self.camera_profile.camera.image_size
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "frame_ids": list(self.frame_ids),
            "frame_count": len(self.frame_ids),
            "image_size": size.as_dict(),
            "camera_profile_id": self.camera_profile.profile_id,
            "scale_profile_id": self.scale_profile.profile_id,
            "metric_scale_available": self.scale_profile.is_metric,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "scene_profile": self.scene_profile.value,
            "source_label": self.source_label,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class PipelineCameraTrajectoryEvidence:
    session_id: str
    frame_ids: tuple[str, ...]
    T_reconstruction_from_camera: tuple[np.ndarray, ...]
    pose_layers: tuple[str, ...]
    source_provider_id: str
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = PIPELINE_GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        ids = _frame_ids(self.frame_ids)
        poses = tuple(
            validate_rigid_matrix(value, field_name=f"T_reconstruction_from_camera[{index}]")
            for index, value in enumerate(self.T_reconstruction_from_camera)
        )
        layers = tuple(require_nonempty(value, field_name="pose_layer") for value in self.pose_layers)
        if len(ids) != len(poses) or len(ids) != len(layers):
            raise ReconstructionValidationError("Trajectory frame IDs/poses/layers must have equal length.")
        if int(self.schema_version) != PIPELINE_GEOMETRY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported F3 pipeline-geometry schema version.")
        object.__setattr__(self, "frame_ids", ids)
        object.__setattr__(self, "T_reconstruction_from_camera", poses)
        object.__setattr__(self, "pose_layers", layers)
        object.__setattr__(
            self,
            "source_provider_id",
            require_nonempty(self.source_provider_id, field_name="source_provider_id"),
        )
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        object.__setattr__(self, "schema_version", PIPELINE_GEOMETRY_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "frame_ids": list(self.frame_ids),
            "frame_count": len(self.frame_ids),
            "pose_convention": DA3_INTERNAL_POSE_CONVENTION,
            "pose_layers": list(self.pose_layers),
            "source_provider_id": self.source_provider_id,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class PipelineDepthEvidence:
    session_id: str
    frame_ids: tuple[str, ...]
    depth_frames: tuple[np.ndarray, ...]
    depth_unit: DenseDepthUnit
    depth_alignments: tuple[DenseDepthAlignment, ...]
    confidence_frames: tuple[np.ndarray | None, ...]
    source_provider_id: str
    geometry_scale: str
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = PIPELINE_GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        ids = _frame_ids(self.frame_ids)
        depths = tuple(_readonly(value, np.float32) for value in self.depth_frames)
        alignments = tuple(self.depth_alignments)
        confidences: list[np.ndarray | None] = []
        if not isinstance(self.depth_unit, DenseDepthUnit):
            raise TypeError("depth_unit must be DenseDepthUnit.")
        if len(ids) != len(depths) or len(ids) != len(alignments) or len(ids) != len(self.confidence_frames):
            raise ReconstructionValidationError("Depth evidence arrays must have equal frame counts.")
        for index, (depth, alignment, confidence) in enumerate(
            zip(depths, alignments, self.confidence_frames)
        ):
            if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth <= 0.0):
                raise ReconstructionValidationError(f"depth_frames[{index}] must be finite positive HxW.")
            if not isinstance(alignment, DenseDepthAlignment):
                raise TypeError("depth_alignments must contain DenseDepthAlignment values.")
            if confidence is None:
                confidences.append(None)
            else:
                array = _readonly(confidence, np.float32)
                if array.shape != depth.shape or not np.isfinite(array).all():
                    raise ReconstructionValidationError("Depth confidence must be finite and match depth shape.")
                if np.any(array < 0.0) or np.any(array > 1.0):
                    raise ReconstructionValidationError("Pipeline depth confidence must lie in [0, 1].")
                confidences.append(array)
        if int(self.schema_version) != PIPELINE_GEOMETRY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported F3 pipeline-geometry schema version.")
        object.__setattr__(self, "frame_ids", ids)
        object.__setattr__(self, "depth_frames", depths)
        object.__setattr__(self, "depth_alignments", alignments)
        object.__setattr__(self, "confidence_frames", tuple(confidences))
        object.__setattr__(
            self,
            "source_provider_id",
            require_nonempty(self.source_provider_id, field_name="source_provider_id"),
        )
        object.__setattr__(self, "geometry_scale", require_nonempty(self.geometry_scale, field_name="geometry_scale"))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        object.__setattr__(self, "schema_version", PIPELINE_GEOMETRY_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "frame_ids": list(self.frame_ids),
            "frame_count": len(self.frame_ids),
            "depth_unit": self.depth_unit.value,
            "geometry_scale": self.geometry_scale,
            "source_provider_id": self.source_provider_id,
            "confidence_available": all(value is not None for value in self.confidence_frames),
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class CanonicalDenseGeometryPayload:
    dense: DenseFusionResult
    quality: DenseGeometryQualityReport
    camera_profile: CameraGeometryProfile
    scale_profile: ScaleReferenceProfile

    def __post_init__(self) -> None:
        if not isinstance(self.dense, DenseFusionResult) or not isinstance(self.quality, DenseGeometryQualityReport):
            raise TypeError("dense/quality must be dense reconstruction result objects.")
        if not isinstance(self.camera_profile, CameraGeometryProfile) or not isinstance(self.scale_profile, ScaleReferenceProfile):
            raise TypeError("camera_profile/scale_profile must use reconstruction contracts.")
        if self.dense.session_id != self.quality.session_id:
            raise ReconstructionValidationError("Dense and quality payload session IDs differ.")
        if self.dense.camera_profile_id != self.camera_profile.profile_id:
            raise ReconstructionValidationError("Dense payload camera profile mismatch.")
        if self.dense.scale_profile_id != self.scale_profile.profile_id:
            raise ReconstructionValidationError("Dense payload scale profile mismatch.")
        if self.dense.certified_metrology or self.quality.certified_metrology:
            raise ReconstructionValidationError("F3 canonical dense payload cannot claim certified metrology.")


class PipelinePayloadStore:
    """In-process typed payload registry behind F1's serializable artifact references."""

    def __init__(self) -> None:
        self._payloads: dict[tuple[str, type[Any]], Any] = {}
        self._counter = 0

    def put(self, payload: Any, *, payload_ref: str | None = None) -> str:
        if payload_ref is None:
            payload_ref = f"memory://phase8-7f3/{self._counter}"
            self._counter += 1
        ref = require_nonempty(payload_ref, field_name="payload_ref")
        key = (ref, type(payload))
        if key in self._payloads:
            raise ReconstructionValidationError(f"Payload already registered for {ref} and {type(payload).__name__}.")
        self._payloads[key] = payload
        return ref

    def get(self, payload_ref: str, expected_type: type[_T]) -> _T:
        ref = require_nonempty(payload_ref, field_name="payload_ref")
        value = self._payloads.get((ref, expected_type))
        if value is None:
            available = sorted(kind.__name__ for (stored_ref, kind) in self._payloads if stored_ref == ref)
            raise ReconstructionValidationError(
                f"No {expected_type.__name__} payload registered for {ref}; available={available}."
            )
        return value

    def contains(self, payload_ref: str, expected_type: type[Any]) -> bool:
        return (str(payload_ref), expected_type) in self._payloads

    def contains_ref(self, payload_ref: str) -> bool:
        ref = require_nonempty(payload_ref, field_name="payload_ref")
        return any(stored_ref == ref for stored_ref, _kind in self._payloads)

    @property
    def payload_refs(self) -> tuple[str, ...]:
        return tuple(sorted({stored_ref for stored_ref, _kind in self._payloads}))

    def find_unique(self, expected_type: type[_T], *, predicate: Callable[[_T], bool] | None = None) -> _T:
        values = [
            value
            for (_ref, kind), value in self._payloads.items()
            if kind is expected_type and (predicate is None or predicate(value))
        ]
        if len(values) != 1:
            raise ReconstructionValidationError(
                f"Expected exactly one {expected_type.__name__} payload; found {len(values)}."
            )
        return values[0]


class BoundEvidenceProviderExecutor:
    """Bind previously qualified source/pose/depth evidence into F1 artifacts.

    This is an integration adapter, not a development mock.  The caller decides
    which provider identity is exposed so replayed DA3/SfM evidence remains
    explicit in pipeline provenance.
    """

    _TYPE_BY_STAGE: Mapping[ReconstructionPipelineStage, type[Any]] = {
        ReconstructionPipelineStage.SOURCE_INGESTION: PipelineIngestionEvidence,
        ReconstructionPipelineStage.CAMERA_POSE: PipelineCameraTrajectoryEvidence,
        ReconstructionPipelineStage.DEPTH_EVIDENCE: PipelineDepthEvidence,
    }

    def __init__(
        self,
        *,
        store: PipelinePayloadStore,
        provider_id: str,
        payload_ref_by_stage: Mapping[ReconstructionPipelineStage, str],
    ) -> None:
        self.store = store
        self.provider_id = require_nonempty(provider_id, field_name="provider_id")
        self.payload_ref_by_stage = dict(payload_ref_by_stage)

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != self.provider_id:
            raise ReconstructionValidationError("Bound evidence provider identity mismatch.")
        stage = context.planned_stage.stage
        expected_type = self._TYPE_BY_STAGE.get(stage)
        if expected_type is None:
            raise ReconstructionValidationError("Bound evidence executor only serves ingestion/camera/depth stages.")
        ref = self.payload_ref_by_stage.get(stage, "")
        payload = self.store.get(ref, expected_type)
        if payload.session_id != context.request.session_id:
            raise ReconstructionValidationError("Bound evidence session does not match pipeline request.")
        contract = context.planned_stage.contract
        metadata: dict[str, JsonValue] = {
            "binding_only": True,
            "development_mock": False,
            "certified_metrology": False,
        }
        if hasattr(payload, "as_dict"):
            summary = payload.as_dict()
            metadata["frame_count"] = int(summary.get("frame_count", 0))
            if "geometry_scale" in summary:
                metadata["geometry_scale"] = str(summary["geometry_scale"])
            if "source_provider_id" in summary:
                metadata["source_provider_id"] = str(summary["source_provider_id"])
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:{stage.value}:{self.provider_id}",
            session_id=context.request.session_id,
            stage=stage,
            kind=contract.output_kind,
            authority=contract.output_authority,
            provider_id=self.provider_id,
            payload_ref=ref,
            metadata=metadata,
        )


def bound_evidence_provider_descriptor(
    *,
    provider_id: str = PIPELINE_EVIDENCE_PROVIDER_ID,
    provider_version: str = PIPELINE_F3_ADAPTER_VERSION,
    roles: Sequence[ReconstructionProviderRole] = (
        ReconstructionProviderRole.INGESTION,
        ReconstructionProviderRole.CAMERA_POSE,
        ReconstructionProviderRole.DEPTH,
    ),
    priority: int = 400,
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = (),
) -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version=provider_version,
        adapter_version=PIPELINE_F3_ADAPTER_VERSION,
        roles=tuple(roles),
        execution=ReconstructionProviderExecution.IN_PROCESS,
        priority=priority,
        metadata={
            "binding_only": True,
            "development_mock": False,
            "certified_metrology": False,
            **metadata_as_dict(normalize_metadata(metadata)),  # type: ignore[arg-type]
        },
    )


def canonical_geometry_provider_descriptors(*, priority: int = 500) -> tuple[ReconstructionProviderDescriptor, ...]:
    common = {
        "phase": "8.7F3",
        "development_mock": False,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
    }
    return (
        ReconstructionProviderDescriptor(
            provider_id=PIPELINE_DENSE_PROVIDER_ID,
            provider_version="phase8-6d-core",
            adapter_version=PIPELINE_F3_ADAPTER_VERSION,
            roles=(ReconstructionProviderRole.DENSE_GEOMETRY,),
            execution=ReconstructionProviderExecution.IN_PROCESS,
            priority=priority,
            metadata={**common, "authority": "canonical_geometry"},
        ),
        ReconstructionProviderDescriptor(
            provider_id=PIPELINE_SURFACE_PROVIDER_ID,
            provider_version="phase8-7a-core",
            adapter_version=PIPELINE_F3_ADAPTER_VERSION,
            roles=(ReconstructionProviderRole.SURFACE_MESH,),
            execution=ReconstructionProviderExecution.IN_PROCESS,
            priority=priority,
            metadata={**common, "authority": "canonical_geometry"},
        ),
        ReconstructionProviderDescriptor(
            provider_id=PIPELINE_SEMANTICS_PROVIDER_ID,
            provider_version="phase8-7b-core",
            adapter_version=PIPELINE_F3_ADAPTER_VERSION,
            roles=(ReconstructionProviderRole.SEMANTICS,),
            execution=ReconstructionProviderExecution.IN_PROCESS,
            priority=priority,
            metadata={**common, "authority": "semantic_context"},
        ),
        ReconstructionProviderDescriptor(
            provider_id=PIPELINE_INSPECTION_PROVIDER_ID,
            provider_version="phase8-7c-core",
            adapter_version=PIPELINE_F3_ADAPTER_VERSION,
            roles=(ReconstructionProviderRole.INSPECTION_QC,),
            execution=ReconstructionProviderExecution.IN_PROCESS,
            priority=priority,
            metadata={**common, "authority": "inspection_decision"},
        ),
    )


class CanonicalDenseGeometryProviderExecutor:
    def __init__(
        self,
        *,
        store: PipelinePayloadStore,
        policy: DenseFusionPolicy,
        quality_policy: DenseGeometryQualityPolicy | None = None,
        quality_evidence_builder: Callable[[DenseFusionResult], DenseGeometryEvidence] | None = None,
    ) -> None:
        if not isinstance(policy, DenseFusionPolicy):
            raise TypeError("policy must be DenseFusionPolicy.")
        self.store = store
        self.policy = policy
        self.quality_policy = quality_policy
        self.quality_evidence_builder = quality_evidence_builder

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != PIPELINE_DENSE_PROVIDER_ID:
            raise ReconstructionValidationError("Dense provider identity mismatch.")
        if context.planned_stage.stage is not ReconstructionPipelineStage.DENSE_GEOMETRY:
            raise ReconstructionValidationError("Dense provider can only execute dense_geometry.")
        by_kind = {artifact.kind: artifact for artifact in context.required_inputs}
        camera_artifact = by_kind[ReconstructionArtifactKind.CAMERA_TRAJECTORY]
        depth_artifact = by_kind[ReconstructionArtifactKind.DEPTH_EVIDENCE]
        # The ingestion artifact is an ancestor rather than a direct F1 dense-stage input;
        # recover it through the request-scoped store by finding the registered session payload.
        ingestion = self._find_ingestion(context.request.session_id)
        trajectory = self.store.get(camera_artifact.payload_ref, PipelineCameraTrajectoryEvidence)
        depth = self.store.get(depth_artifact.payload_ref, PipelineDepthEvidence)
        self._validate_join(ingestion, trajectory, depth)
        frames = tuple(
            DenseFusionFrame(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                rgb=rgb,
                depth=depth_frame,
                T_reconstruction_from_camera=pose,
                depth_unit=depth.depth_unit,
                depth_alignment=alignment,
                confidence=confidence,
                pose_layer=pose_layer,
            )
            for frame_id, timestamp_ns, rgb, pose, pose_layer, depth_frame, alignment, confidence in zip(
                ingestion.frame_ids,
                ingestion.timestamps_ns,
                ingestion.rgb_frames,
                trajectory.T_reconstruction_from_camera,
                trajectory.pose_layers,
                depth.depth_frames,
                depth.depth_alignments,
                depth.confidence_frames,
            )
        )
        dense = fuse_dense_frames(
            frames,
            session_id=context.request.session_id,
            camera_profile=ingestion.camera_profile,
            scale_profile=ingestion.scale_profile,
            policy=self.policy,
            reconstruction_frame_id=ingestion.reconstruction_frame_id,
        )
        evidence = self.quality_evidence_builder(dense) if self.quality_evidence_builder else None
        quality = assess_dense_geometry_quality(
            dense,
            camera_profile=ingestion.camera_profile,
            scale_profile=ingestion.scale_profile,
            policy=self.quality_policy,
            evidence=evidence,
        )
        payload = CanonicalDenseGeometryPayload(
            dense=dense,
            quality=quality,
            camera_profile=ingestion.camera_profile,
            scale_profile=ingestion.scale_profile,
        )
        ref = self.store.put(payload)
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:dense_geometry:{PIPELINE_DENSE_PROVIDER_ID}",
            session_id=context.request.session_id,
            stage=ReconstructionPipelineStage.DENSE_GEOMETRY,
            kind=ReconstructionArtifactKind.DENSE_GEOMETRY,
            authority=ReconstructionArtifactAuthority.CANONICAL_GEOMETRY,
            provider_id=PIPELINE_DENSE_PROVIDER_ID,
            payload_ref=ref,
            metadata={
                "point_count": len(dense.point_cloud.points_reconstruction),
                "metric_geometry_status": dense.metric_geometry_status.value,
                "scale_profile_id": dense.scale_profile_id,
                "quality_high_fraction": quality.summary.high_quality_ratio,
                "quality_medium_fraction": quality.summary.medium_quality_ratio,
                "canonical_geometry_authoritative": True,
                "certified_metrology": False,
                "depth_source_provider_id": depth.source_provider_id,
                "pose_source_provider_id": trajectory.source_provider_id,
            },
        )

    def _find_ingestion(self, session_id: str) -> PipelineIngestionEvidence:
        return self.store.find_unique(
            PipelineIngestionEvidence,
            predicate=lambda value: value.session_id == session_id,
        )

    @staticmethod
    def _validate_join(
        ingestion: PipelineIngestionEvidence,
        trajectory: PipelineCameraTrajectoryEvidence,
        depth: PipelineDepthEvidence,
    ) -> None:
        if not (ingestion.session_id == trajectory.session_id == depth.session_id):
            raise ReconstructionValidationError("F3 evidence session IDs differ.")
        if ingestion.frame_ids != trajectory.frame_ids or ingestion.frame_ids != depth.frame_ids:
            raise ReconstructionValidationError("F3 evidence frame ordering differs across RGB/pose/depth.")
        expected = (
            ingestion.camera_profile.camera.image_size.height,
            ingestion.camera_profile.camera.image_size.width,
        )
        if any(value.shape != expected for value in depth.depth_frames):
            raise ReconstructionValidationError("Depth evidence does not match the ingestion camera pixel space.")


class SurfaceMeshProviderExecutor:
    def __init__(self, *, store: PipelinePayloadStore, policy: SurfaceMeshingPolicy | None = None) -> None:
        self.store = store
        self.policy = policy

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != PIPELINE_SURFACE_PROVIDER_ID:
            raise ReconstructionValidationError("Surface provider identity mismatch.")
        if context.planned_stage.stage is not ReconstructionPipelineStage.SURFACE_MESH:
            raise ReconstructionValidationError("Surface provider can only execute surface_mesh.")
        source = self.store.get(context.required_inputs[0].payload_ref, CanonicalDenseGeometryPayload)
        result = generate_quality_aware_surface_mesh(source.dense, source.quality, policy=self.policy)
        ref = self.store.put(result)
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:surface_mesh:{PIPELINE_SURFACE_PROVIDER_ID}",
            session_id=context.request.session_id,
            stage=ReconstructionPipelineStage.SURFACE_MESH,
            kind=ReconstructionArtifactKind.SURFACE_MESH,
            authority=ReconstructionArtifactAuthority.CANONICAL_GEOMETRY,
            provider_id=PIPELINE_SURFACE_PROVIDER_ID,
            payload_ref=ref,
            metadata={
                "vertex_count": result.mesh.vertex_count,
                "face_count": result.mesh.face_count,
                "metric_geometry_status": result.metric_geometry_status.value,
                "canonical_geometry_authoritative": True,
                "certified_metrology": False,
            },
        )


SemanticObservationResolver = Callable[[ReconstructionPipelineRequest, SurfaceMeshResult], Sequence[SemanticFrameObservation]]
ReferenceAssemblyResolver = Callable[[ReconstructionPipelineRequest, SurfaceMeshResult, SemanticGeometryResult], ReferenceAssembly]


class SemanticGeometryProviderExecutor:
    def __init__(
        self,
        *,
        store: PipelinePayloadStore,
        observation_resolver: SemanticObservationResolver,
        policy: SemanticAssociationPolicy | None = None,
        role_by_class_name: Mapping[str, SemanticGeometryRole | str] | None = None,
    ) -> None:
        self.store = store
        self.observation_resolver = observation_resolver
        self.policy = policy
        self.role_by_class_name = dict(role_by_class_name or {})

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != PIPELINE_SEMANTICS_PROVIDER_ID:
            raise ReconstructionValidationError("Semantic provider identity mismatch.")
        surface = self.store.get(context.required_inputs[0].payload_ref, SurfaceMeshResult)
        observations = tuple(self.observation_resolver(context.request, surface))
        result = build_semantic_3d_geometry(
            surface,
            observations,
            policy=self.policy,
            role_by_class_name=self.role_by_class_name,
        )
        ref = self.store.put(result)
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:semantic_geometry:{PIPELINE_SEMANTICS_PROVIDER_ID}",
            session_id=context.request.session_id,
            stage=ReconstructionPipelineStage.SEMANTIC_GEOMETRY,
            kind=ReconstructionArtifactKind.SEMANTIC_GEOMETRY,
            authority=ReconstructionArtifactAuthority.SEMANTIC_CONTEXT,
            provider_id=PIPELINE_SEMANTICS_PROVIDER_ID,
            payload_ref=ref,
            metadata={
                "object_count": len(result.objects),
                "observations_total": result.statistics.observations_total,
                "observations_accepted": result.statistics.observations_accepted,
                "metric_geometry_status": result.metric_geometry_status.value,
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
        )


class InspectionQCProviderExecutor:
    def __init__(
        self,
        *,
        store: PipelinePayloadStore,
        reference_resolver: ReferenceAssemblyResolver,
        policy: RegistrationInspectionPolicy | None = None,
    ) -> None:
        self.store = store
        self.reference_resolver = reference_resolver
        self.policy = policy

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != PIPELINE_INSPECTION_PROVIDER_ID:
            raise ReconstructionValidationError("Inspection provider identity mismatch.")
        by_kind = {artifact.kind: artifact for artifact in context.required_inputs}
        surface = self.store.get(by_kind[ReconstructionArtifactKind.SURFACE_MESH].payload_ref, SurfaceMeshResult)
        semantic = self.store.get(by_kind[ReconstructionArtifactKind.SEMANTIC_GEOMETRY].payload_ref, SemanticGeometryResult)
        reference = self.reference_resolver(context.request, surface, semantic)
        report = inspect_semantic_geometry(reference, surface, semantic, policy=self.policy)
        ref = self.store.put(report)
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:inspection_qc:{PIPELINE_INSPECTION_PROVIDER_ID}",
            session_id=context.request.session_id,
            stage=ReconstructionPipelineStage.INSPECTION_QC,
            kind=ReconstructionArtifactKind.INSPECTION_REPORT,
            authority=ReconstructionArtifactAuthority.INSPECTION_DECISION,
            provider_id=PIPELINE_INSPECTION_PROVIDER_ID,
            payload_ref=ref,
            metadata={
                "decision": report.decision.value,
                "registration_status": report.registration.status.value,
                "comparison_metric_available": report.comparison_metric_available,
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
        )


def normalize_da3_confidence_for_fusion(value: object) -> np.ndarray:
    """Map non-negative DA3 confidence evidence to a bounded fusion weight.

    DA3 confidence is not reinterpreted as a calibrated probability.  Values in
    [0,1] are preserved.  Unbounded positive scores are divided by their finite
    99th percentile and clipped to [0,1], preserving ordering for weighting.
    """

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ReconstructionValidationError("DA3 confidence must be finite non-negative HxW.")
    maximum = float(np.max(array)) if array.size else 0.0
    if maximum <= 1.0:
        return _readonly(array, np.float32)
    scale = float(np.percentile(array, 99.0))
    if not isfinite(scale) or scale <= 0.0:
        scale = maximum
    normalized = np.clip(array / max(scale, 1e-12), 0.0, 1.0)
    return _readonly(normalized, np.float32)


def da3_camera_profile_from_result(
    manifest: DA3ResultManifest,
    *,
    reconstruction_frame_id: str,
    focal_length_relative_uncertainty: float = 0.25,
    maximum_intrinsics_relative_spread: float = 0.10,
) -> CameraGeometryProfile:
    arrays = validate_da3_npz_result(manifest)
    intrinsics = np.asarray(arrays["intrinsics"], dtype=np.float64)
    representative = np.median(intrinsics, axis=0)
    denominators = np.maximum(np.abs(representative), 1.0)
    spread = float(np.max(np.abs(intrinsics - representative) / denominators))
    maximum = float(maximum_intrinsics_relative_spread)
    if not isfinite(maximum) or maximum < 0.0 or maximum >= 1.0:
        raise ReconstructionValidationError("maximum_intrinsics_relative_spread must be in [0,1).")
    if spread > maximum:
        raise ReconstructionValidationError(
            f"DA3 per-frame intrinsics spread {spread:.6f} exceeds single-camera F3 limit {maximum:.6f}."
        )
    matrix = representative.copy()
    matrix[2, :] = (0.0, 0.0, 1.0)
    size = ImageSize(manifest.width, manifest.height)
    pixel_space_id = f"da3-{manifest.model_revision[:12]}-{manifest.width}x{manifest.height}"
    camera = ReconstructionCamera(
        camera_id="depth-anything-3-camera",
        image_size=size,
        intrinsics=CameraIntrinsics(
            matrix,
            pixel_space_id,
            IntrinsicsSource.ESTIMATED,
            CalibrationStatus.NOT_CALIBRATED,
        ),
        camera_frame=CoordinateFrame(
            "depth-anything-3-camera-frame",
            CoordinateFrameKind.CAMERA_GENERIC,
            AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
            parent_frame_id=require_nonempty(reconstruction_frame_id, field_name="reconstruction_frame_id"),
        ),
        distortion_model=CameraDistortionModel.NONE,
        distortion_coefficients=(),
    )
    return CameraGeometryProfile(
        profile_id=f"da3-estimated-{manifest.model_revision[:12]}",
        camera=camera,
        mode=CameraGeometryMode.ESTIMATED,
        method=CameraGeometryMethod.PROVIDED_INTRINSICS,
        focal_length_relative_uncertainty=float(focal_length_relative_uncertainty),
        distortion_assumed_zero=True,
        notes=(
            "Phase 8.7F3 representative DA3-predicted intrinsics; provisional relative geometry only. "
            f"maximum_relative_intrinsics_spread={spread:.6f}"
        ),
    )


def da3_supporting_evidence_from_result(
    manifest: DA3ResultManifest,
    *,
    session_id: str,
    frame_ids: Sequence[str],
    relative_scale_uncertainty: float = 0.25,
) -> tuple[PipelineCameraTrajectoryEvidence, PipelineDepthEvidence]:
    arrays = validate_da3_npz_result(manifest)
    ids = _frame_ids(tuple(frame_ids))
    if len(ids) != manifest.frame_count:
        raise ReconstructionValidationError("DA3 F3 frame IDs do not match the qualified result frame count.")
    uncertainty = float(relative_scale_uncertainty)
    if not isfinite(uncertainty) or uncertainty < 0.0 or uncertainty >= 1.0:
        raise ReconstructionValidationError("relative_scale_uncertainty must be finite in [0,1).")
    trajectory = PipelineCameraTrajectoryEvidence(
        session_id=session_id,
        frame_ids=ids,
        T_reconstruction_from_camera=tuple(arrays["T_reconstruction_from_camera"]),
        pose_layers=tuple("da3_joint_multiview" for _ in ids),
        source_provider_id=DA3_PROVIDER_ID,
        metadata={
            "source_revision": manifest.source_revision,
            "model_id": manifest.model_id,
            "model_revision": manifest.model_revision,
            "camera_axes": manifest.camera_axes,
            "source_extrinsics_convention": manifest.extrinsics_convention,
            "geometry_scale": manifest.geometry_scale,
        },
    )
    alignment = explicit_depth_scale_alignment(
        scale_to_reconstruction_units=1.0,
        relative_scale_uncertainty=uncertainty,
        provenance=(
            "Depth Anything 3 joint depth/pose arbitrary gauge; unit factor preserves the provider's "
            "internally consistent relative scale and does not establish metric scale."
        ),
    )
    depth = PipelineDepthEvidence(
        session_id=session_id,
        frame_ids=ids,
        depth_frames=tuple(arrays["depth"]),
        depth_unit=DenseDepthUnit.SOURCE_MODEL_UNITS,
        depth_alignments=tuple(alignment for _ in ids),
        confidence_frames=tuple(normalize_da3_confidence_for_fusion(value) for value in arrays["confidence"]),
        source_provider_id=DA3_PROVIDER_ID,
        geometry_scale=manifest.geometry_scale,
        metadata={
            "source_revision": manifest.source_revision,
            "model_id": manifest.model_id,
            "model_revision": manifest.model_revision,
            "confidence_normalization": "preserve_[0,1]_else_99th_percentile_scale_clip",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )
    return trajectory, depth


@dataclass(frozen=True, slots=True)
class F3IntegratedPipelineSummary:
    request_id: str
    pipeline_status: str
    canonical_geometry_artifact_id: str
    dense_point_count: int
    mesh_vertex_count: int
    mesh_face_count: int
    semantic_object_count: int
    inspection_decision: str
    metric_geometry_status: str
    comparison_metric_available: bool
    provider_ids: tuple[str, ...]
    canonical_geometry_authoritative: bool = True
    certified_metrology: bool = False
    reconstruction_ui_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": "8.7F3",
            "request_id": self.request_id,
            "pipeline_status": self.pipeline_status,
            "canonical_geometry_artifact_id": self.canonical_geometry_artifact_id,
            "dense_point_count": self.dense_point_count,
            "mesh_vertex_count": self.mesh_vertex_count,
            "mesh_face_count": self.mesh_face_count,
            "semantic_object_count": self.semantic_object_count,
            "inspection_decision": self.inspection_decision,
            "metric_geometry_status": self.metric_geometry_status,
            "comparison_metric_available": self.comparison_metric_available,
            "provider_ids": list(self.provider_ids),
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "certified_metrology": False,
            "reconstruction_ui_enabled": False,
            "gaussian_nerf_appearance_binding_deferred_to": "8.7F4",
            "persistent_reconstruction_scene_deferred_to": "8.7F5",
        }


def summarize_f3_pipeline_result(
    result: ReconstructionPipelineRunResult,
    *,
    store: PipelinePayloadStore,
) -> F3IntegratedPipelineSummary:
    if not isinstance(result, ReconstructionPipelineRunResult):
        raise TypeError("result must be ReconstructionPipelineRunResult.")
    by_kind = {
        item.artifact.kind: item.artifact
        for item in result.stages
        if item.artifact is not None
    }
    dense_payload = store.get(by_kind[ReconstructionArtifactKind.DENSE_GEOMETRY].payload_ref, CanonicalDenseGeometryPayload)
    surface = store.get(by_kind[ReconstructionArtifactKind.SURFACE_MESH].payload_ref, SurfaceMeshResult)
    semantic = store.get(by_kind[ReconstructionArtifactKind.SEMANTIC_GEOMETRY].payload_ref, SemanticGeometryResult)
    inspection = store.get(by_kind[ReconstructionArtifactKind.INSPECTION_REPORT].payload_ref, InspectionReport)
    if result.canonical_geometry_artifact_id != by_kind[ReconstructionArtifactKind.SURFACE_MESH].artifact_id:
        raise ReconstructionValidationError("F3 canonical pipeline authority must terminate at the surface mesh artifact.")
    if result.certified_metrology or surface.certified_metrology or semantic.certified_metrology or inspection.certified_metrology:
        raise ReconstructionValidationError("F3 pipeline must not claim certified metrology.")
    return F3IntegratedPipelineSummary(
        request_id=result.request_id,
        pipeline_status=result.status.value,
        canonical_geometry_artifact_id=result.canonical_geometry_artifact_id,
        dense_point_count=len(dense_payload.dense.point_cloud.points_reconstruction),
        mesh_vertex_count=surface.mesh.vertex_count,
        mesh_face_count=surface.mesh.face_count,
        semantic_object_count=len(semantic.objects),
        inspection_decision=inspection.decision.value,
        metric_geometry_status=surface.metric_geometry_status.value,
        comparison_metric_available=inspection.comparison_metric_available,
        provider_ids=tuple(item.provider_id for item in result.provider_provenance),
    )
