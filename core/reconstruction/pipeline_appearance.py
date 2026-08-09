"""Phase 8.7F4 Gaussian + NeRF appearance integration.

This module binds the already-qualified Phase 8.7D/E neural appearance
representations into the Phase 8.7F orchestration graph without changing
canonical geometry authority.  Heavy Gaussian/Nerfstudio runtimes remain
external-process concerns supplied through resolver callables.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.gaussian_splatting import GaussianSceneArtifact
from core.reconstruction.nerf import NerfSceneArtifact
from core.reconstruction.neural_scene import (
    NeuralSceneDataset,
    NeuralSceneFrame,
    neural_scene_dataset_fingerprint,
    write_neural_scene_dataset_manifest,
)
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineArtifact,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionStageExecutionContext,
    ReconstructionPipelineStage,
)
from core.reconstruction.pipeline_geometry import (
    PipelineCameraTrajectoryEvidence,
    PipelineIngestionEvidence,
    PipelinePayloadStore,
)
from core.reconstruction.scale_reference import MetricGeometryStatus, assess_reconstruction_geometry
from core.reconstruction.surface_mesh import SurfaceMeshResult

PIPELINE_APPEARANCE_SCHEMA_VERSION = 1
PIPELINE_GAUSSIAN_PROVIDER_ID = "qualified-gaussian-appearance"
PIPELINE_NERF_PROVIDER_ID = "qualified-nerf-appearance"
PIPELINE_F4_ADAPTER_VERSION = "phase8-7f4-1"


@dataclass(frozen=True, slots=True)
class AppearanceDatasetBundle:
    dataset: NeuralSceneDataset
    manifest_path: str
    dataset_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, NeuralSceneDataset):
            raise TypeError("dataset must be NeuralSceneDataset.")
        path = Path(self.manifest_path)
        if not path.is_absolute():
            raise ReconstructionValidationError("appearance manifest_path must be absolute.")
        expected = neural_scene_dataset_fingerprint(self.dataset)
        if str(self.dataset_fingerprint_sha256).lower() != expected:
            raise ReconstructionValidationError("appearance dataset fingerprint mismatch.")
        object.__setattr__(self, "manifest_path", str(path))
        object.__setattr__(self, "dataset_fingerprint_sha256", expected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "frame_count": self.dataset.frame_count,
            "reconstruction_frame_id": self.dataset.reconstruction_frame_id,
            "metric_geometry_status": self.dataset.metric_geometry_status.value,
            "unit_to_metre": self.dataset.unit_to_metre,
            "certified_metrology": False,
        }


def _write_ppm(path: Path, rgb: np.ndarray) -> None:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ReconstructionValidationError("appearance RGB frame must be HxWx3 uint8.")
    height, width, _ = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + np.ascontiguousarray(array).tobytes())


def materialize_appearance_dataset(
    ingestion: PipelineIngestionEvidence,
    trajectory: PipelineCameraTrajectoryEvidence,
    *,
    output_dir: str | Path,
) -> AppearanceDatasetBundle:
    """Materialize RGB+canonical poses as one backend-neutral neural dataset."""
    if not isinstance(ingestion, PipelineIngestionEvidence):
        raise TypeError("ingestion must be PipelineIngestionEvidence.")
    if not isinstance(trajectory, PipelineCameraTrajectoryEvidence):
        raise TypeError("trajectory must be PipelineCameraTrajectoryEvidence.")
    if ingestion.session_id != trajectory.session_id or ingestion.frame_ids != trajectory.frame_ids:
        raise ReconstructionValidationError("appearance RGB and trajectory evidence must describe the same ordered frames.")

    root = Path(output_dir)
    if not root.is_absolute():
        raise ReconstructionValidationError("appearance output_dir must be absolute.")
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    intr = ingestion.camera_profile.camera.intrinsics
    size = ingestion.camera_profile.camera.image_size
    frames: list[NeuralSceneFrame] = []
    for index, (frame_id, timestamp_ns, rgb, pose, layer) in enumerate(
        zip(
            ingestion.frame_ids,
            ingestion.timestamps_ns,
            ingestion.rgb_frames,
            trajectory.T_reconstruction_from_camera,
            trajectory.pose_layers,
        )
    ):
        image_path = (image_dir / f"{index:04d}_{frame_id}.ppm").resolve()
        _write_ppm(image_path, rgb)
        frames.append(
            NeuralSceneFrame(
                frame_id=frame_id,
                image_path=str(image_path),
                width=size.width,
                height=size.height,
                fx=float(intr.matrix[0, 0]),
                fy=float(intr.matrix[1, 1]),
                cx=float(intr.matrix[0, 2]),
                cy=float(intr.matrix[1, 2]),
                T_reconstruction_from_camera=pose,
                pose_layer=layer,
                camera_profile_id=ingestion.camera_profile.profile_id,
                timestamp_ns=timestamp_ns,
            )
        )

    gauge = ingestion.scale_profile.gauge
    metric_status = assess_reconstruction_geometry(ingestion.camera_profile, ingestion.scale_profile).metric_status
    dataset = NeuralSceneDataset(
        session_id=ingestion.session_id,
        scene_profile=ingestion.scene_profile,
        reconstruction_frame_id=ingestion.reconstruction_frame_id,
        camera_profile_id=ingestion.camera_profile.profile_id,
        scale_profile_id=ingestion.scale_profile.profile_id,
        metric_geometry_status=metric_status,
        scale_status=gauge.scale_status,
        world_alignment_status=gauge.alignment_status,
        unit_to_metre=gauge.unit_to_metre,
        relative_scale_uncertainty=gauge.relative_scale_uncertainty,
        frames=tuple(frames),
        certified_metrology=False,
    )
    manifest = write_neural_scene_dataset_manifest(dataset, (root / "dataset.json").resolve())
    return AppearanceDatasetBundle(
        dataset=dataset,
        manifest_path=str(manifest),
        dataset_fingerprint_sha256=neural_scene_dataset_fingerprint(dataset),
    )


def appearance_provider_descriptors(*, priority: int = 600) -> tuple[ReconstructionProviderDescriptor, ...]:
    return (
        ReconstructionProviderDescriptor(
            provider_id=PIPELINE_GAUSSIAN_PROVIDER_ID,
            provider_version=PIPELINE_F4_ADAPTER_VERSION,
            roles=(ReconstructionProviderRole.GAUSSIAN_APPEARANCE,),
            execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
            priority=priority,
            adapter_version=PIPELINE_F4_ADAPTER_VERSION,
            metadata={"authority": "appearance_only", "phase": "8.7F4"},
        ),
        ReconstructionProviderDescriptor(
            provider_id=PIPELINE_NERF_PROVIDER_ID,
            provider_version=PIPELINE_F4_ADAPTER_VERSION,
            roles=(ReconstructionProviderRole.NERF_APPEARANCE,),
            execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
            priority=priority,
            adapter_version=PIPELINE_F4_ADAPTER_VERSION,
            metadata={"authority": "appearance_only", "phase": "8.7F4"},
        ),
    )


GaussianArtifactResolver = Callable[[ReconstructionStageExecutionContext, AppearanceDatasetBundle, SurfaceMeshResult], GaussianSceneArtifact]
NerfArtifactResolver = Callable[[ReconstructionStageExecutionContext, AppearanceDatasetBundle], NerfSceneArtifact]


def _validate_common_artifact(
    *,
    dataset: NeuralSceneDataset,
    fingerprint: str,
    reconstruction_frame_id: str,
    metric_status: MetricGeometryStatus,
    unit_to_metre: float | None,
    artifact: GaussianSceneArtifact | NerfSceneArtifact,
) -> None:
    if artifact.dataset_fingerprint_sha256 != fingerprint:
        raise ReconstructionValidationError("appearance artifact dataset fingerprint does not match pipeline evidence.")
    if artifact.reconstruction_frame_id != reconstruction_frame_id:
        raise ReconstructionValidationError("appearance artifact reconstruction frame does not match canonical geometry.")
    if artifact.input_frame_count != dataset.frame_count:
        raise ReconstructionValidationError("appearance artifact must consume the complete F4 neural dataset.")
    if artifact.metric_geometry_status is not metric_status:
        raise ReconstructionValidationError("appearance artifact metric status does not match canonical scene scale.")
    if artifact.unit_to_metre != unit_to_metre:
        raise ReconstructionValidationError("appearance artifact unit_to_metre does not match canonical scene scale.")
    if artifact.canonical_geometry_authoritative or artifact.certified_metrology:
        raise ReconstructionValidationError("neural appearance artifacts cannot claim canonical/metrology authority.")
    if not artifact.render_capable:
        raise ReconstructionValidationError("F4 requires a render-capable appearance artifact.")


class GaussianAppearanceProviderExecutor:
    def __init__(
        self,
        *,
        store: PipelinePayloadStore,
        output_root: str | Path,
        artifact_resolver: GaussianArtifactResolver,
    ) -> None:
        self.store = store
        self.output_root = Path(output_root).resolve()
        self.artifact_resolver = artifact_resolver

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != PIPELINE_GAUSSIAN_PROVIDER_ID:
            raise ReconstructionValidationError("Gaussian F4 provider identity mismatch.")
        if context.planned_stage.stage is not ReconstructionPipelineStage.GAUSSIAN_APPEARANCE:
            raise ReconstructionValidationError("Gaussian F4 provider can only execute gaussian_appearance.")
        by_kind = {item.kind: item for item in context.required_inputs}
        ingestion = self.store.get(by_kind[ReconstructionArtifactKind.INGESTION_RESULT].payload_ref, PipelineIngestionEvidence)
        trajectory = self.store.get(by_kind[ReconstructionArtifactKind.CAMERA_TRAJECTORY].payload_ref, PipelineCameraTrajectoryEvidence)
        surface = self.store.get(by_kind[ReconstructionArtifactKind.SURFACE_MESH].payload_ref, SurfaceMeshResult)
        bundle = materialize_appearance_dataset(
            ingestion,
            trajectory,
            output_dir=self.output_root / context.request.request_id / "shared-neural-scene",
        )
        artifact = self.artifact_resolver(context, bundle, surface)
        _validate_common_artifact(
            dataset=bundle.dataset,
            fingerprint=bundle.dataset_fingerprint_sha256,
            reconstruction_frame_id=surface.reconstruction_frame_id,
            metric_status=surface.metric_geometry_status,
            unit_to_metre=surface.unit_to_metre,
            artifact=artifact,
        )
        ref = self.store.put(artifact)
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:gaussian_appearance:{PIPELINE_GAUSSIAN_PROVIDER_ID}",
            session_id=context.request.session_id,
            stage=ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,
            kind=ReconstructionArtifactKind.GAUSSIAN_SCENE,
            authority=ReconstructionArtifactAuthority.APPEARANCE_ONLY,
            provider_id=PIPELINE_GAUSSIAN_PROVIDER_ID,
            payload_ref=ref,
            metadata={
                "backend_id": artifact.backend_id,
                "backend_version": artifact.backend_version,
                "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
                "render_capable": artifact.render_capable,
                "scene_role": artifact.scene_role.value,
                "canonical_surface_artifact_id": by_kind[ReconstructionArtifactKind.SURFACE_MESH].artifact_id,
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
        )


class NerfAppearanceProviderExecutor:
    def __init__(
        self,
        *,
        store: PipelinePayloadStore,
        output_root: str | Path,
        artifact_resolver: NerfArtifactResolver,
    ) -> None:
        self.store = store
        self.output_root = Path(output_root).resolve()
        self.artifact_resolver = artifact_resolver

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != PIPELINE_NERF_PROVIDER_ID:
            raise ReconstructionValidationError("NeRF F4 provider identity mismatch.")
        if context.planned_stage.stage is not ReconstructionPipelineStage.NERF_APPEARANCE:
            raise ReconstructionValidationError("NeRF F4 provider can only execute nerf_appearance.")
        by_kind = {item.kind: item for item in context.required_inputs}
        ingestion = self.store.get(by_kind[ReconstructionArtifactKind.INGESTION_RESULT].payload_ref, PipelineIngestionEvidence)
        trajectory = self.store.get(by_kind[ReconstructionArtifactKind.CAMERA_TRAJECTORY].payload_ref, PipelineCameraTrajectoryEvidence)
        bundle = materialize_appearance_dataset(
            ingestion,
            trajectory,
            output_dir=self.output_root / context.request.request_id / "shared-neural-scene",
        )
        artifact = self.artifact_resolver(context, bundle)
        metric_status = bundle.dataset.metric_geometry_status
        _validate_common_artifact(
            dataset=bundle.dataset,
            fingerprint=bundle.dataset_fingerprint_sha256,
            reconstruction_frame_id=bundle.dataset.reconstruction_frame_id,
            metric_status=metric_status,
            unit_to_metre=bundle.dataset.unit_to_metre,
            artifact=artifact,
        )
        ref = self.store.put(artifact)
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:nerf_appearance:{PIPELINE_NERF_PROVIDER_ID}",
            session_id=context.request.session_id,
            stage=ReconstructionPipelineStage.NERF_APPEARANCE,
            kind=ReconstructionArtifactKind.NERF_SCENE,
            authority=ReconstructionArtifactAuthority.APPEARANCE_ONLY,
            provider_id=PIPELINE_NERF_PROVIDER_ID,
            payload_ref=ref,
            metadata={
                "backend_id": artifact.backend_id,
                "backend_version": artifact.backend_version,
                "backend_revision": artifact.backend_revision,
                "method_name": artifact.method_name,
                "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
                "render_capable": artifact.render_capable,
                "scene_role": artifact.scene_role.value,
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
        )


@dataclass(frozen=True, slots=True)
class F4AppearanceIntegrationSummary:
    pipeline_status: str
    canonical_geometry_artifact_id: str
    gaussian_artifact_id: str
    nerf_artifact_id: str
    shared_dataset_fingerprint_sha256: str
    canonical_geometry_authoritative: bool
    gaussian_appearance_only: bool
    nerf_appearance_only: bool
    certified_metrology: bool = False
    reconstruction_ui_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline_status": self.pipeline_status,
            "canonical_geometry_artifact_id": self.canonical_geometry_artifact_id,
            "gaussian_artifact_id": self.gaussian_artifact_id,
            "nerf_artifact_id": self.nerf_artifact_id,
            "shared_dataset_fingerprint_sha256": self.shared_dataset_fingerprint_sha256,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "gaussian_appearance_only": self.gaussian_appearance_only,
            "nerf_appearance_only": self.nerf_appearance_only,
            "certified_metrology": False,
            "reconstruction_ui_enabled": False,
            "persistent_reconstruction_scene_deferred_to": "8.7F5",
        }
