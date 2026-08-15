"""Read-only dense reconstruction provider presentation for Phase 9.3G.

9.3G/9.3H expose provider identity, scene binding, execution mode and authority in
Reconstruction. The existing persistent scene remains the source of provider/artifact
provenance. Phase 9.3H may bind isolated DA3 pose/depth supporting evidence into a
new canonical dense/surface scene; DA3 itself never gains canonical or metric authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.reconstruction_canonical_scene_presentation import CanonicalScenePresentation
from core.reconstruction.da3 import DA3_PROVIDER_ID
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionPipelineStage,
)
from core.reconstruction.persistent_scene import PersistentReconstructionScene


@dataclass(frozen=True, slots=True)
class DenseProviderEntry:
    provider_id: str
    label: str
    role_text: str
    execution_text: str
    scene_use_text: str
    authority_text: str
    state_text: str
    scene_bound: bool
    external: bool

    def __post_init__(self) -> None:
        for field_name in (
            "provider_id",
            "label",
            "role_text",
            "execution_text",
            "scene_use_text",
            "authority_text",
            "state_text",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "scene_bound", bool(self.scene_bound))
        object.__setattr__(self, "external", bool(self.external))


@dataclass(frozen=True, slots=True)
class DenseProviderPresentation:
    available: bool
    scene_id: str
    pipeline_text: str
    scale_text: str
    integration_text: str
    authority_text: str
    entries: tuple[DenseProviderEntry, ...]
    canonical_geometry_authoritative: bool
    certified_metrology: bool = False
    backend_started: bool = False

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if len({item.provider_id for item in entries}) != len(entries):
            raise ValueError("Dense provider presentation entries must have unique provider IDs.")
        if self.certified_metrology:
            raise ValueError("9.3G provider presentation cannot certify metrology.")
        if self.backend_started:
            raise ValueError("9.3G provider presentation must never start a backend.")
        if self.available and not self.canonical_geometry_authoritative:
            raise ValueError("A persistent provider presentation must preserve canonical geometry authority.")
        if not self.available and entries:
            raise ValueError("Unavailable provider presentation cannot expose scene provider entries.")
        object.__setattr__(self, "entries", entries)


_ROLE_BY_STAGE = {
    ReconstructionPipelineStage.DENSE_GEOMETRY: "Canonical dense geometry",
    ReconstructionPipelineStage.SURFACE_MESH: "Canonical surface mesh",
    ReconstructionPipelineStage.DEPTH_EVIDENCE: "Supporting depth evidence",
}

_LABEL_BY_STAGE = {
    ReconstructionPipelineStage.DENSE_GEOMETRY: "Dense fusion",
    ReconstructionPipelineStage.SURFACE_MESH: "Surface mesh",
    ReconstructionPipelineStage.DEPTH_EVIDENCE: "Depth evidence",
}


def _execution_text(scene: PersistentReconstructionScene, provider_id: str) -> tuple[str, bool]:
    provenance = next(
        (item for item in scene.provider_provenance if item.provider_id == provider_id),
        None,
    )
    if provenance is None:
        if provider_id == DA3_PROVIDER_ID:
            return "External process • isolated runtime", True
        return "Execution provenance unavailable", False
    value = provenance.execution.value.replace("_", " ").strip().title()
    return value, provenance.execution.value == "external_process"


def _authority_text(authority: ReconstructionArtifactAuthority) -> str:
    if authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY:
        return "Canonical geometry authority"
    if authority is ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY:
        return "Supporting geometry only"
    return authority.value.replace("_", " ").strip().title()


def build_dense_provider_presentation(
    scene: PersistentReconstructionScene | None,
    canonical_scene: CanonicalScenePresentation | None,
) -> DenseProviderPresentation:
    """Build a fail-closed provider inspector without probing/starting backends."""

    if scene is None:
        return DenseProviderPresentation(
            available=False,
            scene_id="",
            pipeline_text="No persistent reconstruction scene",
            scale_text="—",
            integration_text="Phase 9.3H DA3 canonical integration requires an image/video source",
            authority_text="No provider authority presented",
            entries=(),
            canonical_geometry_authoritative=False,
        )
    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene or None.")
    if canonical_scene is None or not isinstance(canonical_scene, CanonicalScenePresentation):
        raise ValueError("9.3G provider presentation requires the active canonical-scene presentation.")
    if not canonical_scene.available or canonical_scene.scene_id != scene.scene_id:
        raise ValueError("Canonical-scene presentation is not bound to the active persistent scene.")
    if not scene.canonical_geometry_authoritative or not canonical_scene.canonical_geometry_authoritative:
        raise ValueError("9.3G cannot present a scene that lost canonical geometry authority.")

    by_stage = {item.stage: item for item in scene.artifacts}
    dense = by_stage.get(ReconstructionPipelineStage.DENSE_GEOMETRY)
    surface = by_stage.get(ReconstructionPipelineStage.SURFACE_MESH)
    depth = by_stage.get(ReconstructionPipelineStage.DEPTH_EVIDENCE)
    if dense is None or surface is None or depth is None:
        raise ValueError("Persistent scene is missing dense/surface/depth provider evidence required by 9.3G.")
    if dense.authority is not ReconstructionArtifactAuthority.CANONICAL_GEOMETRY:
        raise ValueError("Dense geometry artifact lost canonical authority.")
    if surface.authority is not ReconstructionArtifactAuthority.CANONICAL_GEOMETRY:
        raise ValueError("Surface mesh artifact lost canonical authority.")
    if depth.authority is not ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY:
        raise ValueError("Depth evidence must remain supporting geometry.")

    entries: list[DenseProviderEntry] = []
    seen: set[str] = set()
    for stage, artifact in (
        (ReconstructionPipelineStage.DENSE_GEOMETRY, dense),
        (ReconstructionPipelineStage.SURFACE_MESH, surface),
        (ReconstructionPipelineStage.DEPTH_EVIDENCE, depth),
    ):
        if artifact.provider_id in seen:
            continue
        execution, external = _execution_text(scene, artifact.provider_id)
        is_da3 = artifact.provider_id == DA3_PROVIDER_ID
        entries.append(
            DenseProviderEntry(
                provider_id=artifact.provider_id,
                label=("Depth Anything 3" if is_da3 else _LABEL_BY_STAGE[stage]),
                role_text=(
                    "Supporting relative depth / pose evidence"
                    if is_da3
                    else _ROLE_BY_STAGE[stage]
                ),
                execution_text=execution,
                scene_use_text="Bound to current persistent scene",
                authority_text=(
                    "Supporting geometry only • never creates metric scale"
                    if is_da3
                    else _authority_text(artifact.authority)
                ),
                state_text=(
                    "Persisted DA3 supporting evidence • canonicalized by separate dense/surface providers"
                    if is_da3
                    else (
                        "Persisted canonical provider evidence"
                        if artifact.authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY
                        else "Persisted supporting evidence"
                    )
                ),
                scene_bound=True,
                external=external,
            )
        )
        seen.add(artifact.provider_id)

    da3_bound = DA3_PROVIDER_ID in seen
    if not da3_bound:
        entries.append(
            DenseProviderEntry(
                provider_id=DA3_PROVIDER_ID,
                label="Depth Anything 3",
                role_text="Supporting relative depth / pose evidence",
                execution_text="External process • isolated runtime",
                scene_use_text="Not used by current persistent scene",
                authority_text="Supporting geometry only • never creates metric scale",
                state_text="Known isolated provider • runtime not probed • explicit 9.3H integration available when a supported source is selected",
                scene_bound=False,
                external=True,
            )
        )

    integration_text = (
        "DA3 support bound → canonical-dense-fusion → quality-aware-surface-mesh"
        if da3_bound
        else "DA3 integration available • explicit source-driven action • no metric promotion"
    )
    return DenseProviderPresentation(
        available=True,
        scene_id=scene.scene_id,
        pipeline_text=f"{dense.provider_id} → {surface.provider_id}",
        scale_text=canonical_scene.scale_text,
        integration_text=integration_text,
        authority_text="Dense/surface canonical • depth providers remain supporting evidence",
        entries=tuple(entries),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        backend_started=False,
    )


__all__ = [
    "DenseProviderEntry",
    "DenseProviderPresentation",
    "build_dense_provider_presentation",
]
