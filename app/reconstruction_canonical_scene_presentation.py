"""Passive canonical-scene presentation for Phase 9.3A.

The persistent reconstruction scene remains the source of canonical geometry
identity/authority and the Phase 9.1H authority sidecar remains the source of
camera-calibration and metric-scale qualification.  This module only composes
those already-qualified contracts into dependency-light UI state.

It intentionally does not parse mesh payloads, run reconstruction providers,
start isolated neural backends, reinterpret semantic geometry, compute
measurements, or establish metric/metrology authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.reconstruction.metric_qualification import MetricQualificationState
from core.reconstruction.orchestration import ReconstructionArtifactKind
from core.reconstruction.persistent_authority import PersistentSceneAuthoritySnapshot
from core.reconstruction.persistent_scene import (
    PersistentReconstructionScene,
    PersistentReconstructionSceneState,
    PersistentSceneArtifactRecord,
    PersistentSceneFileReference,
)


CANONICAL_SURFACE_RESOURCE_ROLE = "surface_obj_path"


class CanonicalSceneScaleState(str, Enum):
    """Truthful scale state exposed by the presentation-only scene summary."""

    NO_SCENE = "no_scene"
    NOT_QUALIFIED = "not_qualified"
    PROVISIONAL = "provisional"
    QUALIFIED = "qualified"


class CanonicalSceneResourceState(str, Enum):
    """Materialization state of the canonical surface resource for later 9.3B."""

    NO_SCENE = "no_scene"
    NOT_REFERENCED = "not_referenced"
    MISSING = "missing"
    MATERIALIZED = "materialized"


@dataclass(frozen=True, slots=True)
class CanonicalScenePresentation:
    """Read-only UI projection of one persistent canonical reconstruction scene."""

    available: bool
    scene_id: str
    scene_state: str
    artifact_id: str
    artifact_kind: str
    provider_id: str
    canonical_geometry_authoritative: bool
    resource_state: CanonicalSceneResourceState
    resource_role: str
    resource_path: str
    calibration_available: bool
    scale_state: CanonicalSceneScaleState
    scale_relative_uncertainty: float | None
    scale_unit_to_metre: float | None
    metric_evidence_id: str
    metric_assessment_state: str
    certified_metrology: bool

    def __post_init__(self) -> None:
        if self.certified_metrology:
            raise ValueError("Canonical scene presentation cannot claim certified metrology.")
        if not self.available:
            if self.canonical_geometry_authoritative:
                raise ValueError("Unavailable canonical scene cannot claim geometry authority.")
            if self.scale_state is not CanonicalSceneScaleState.NO_SCENE:
                raise ValueError("Unavailable canonical scene must use NO_SCENE scale state.")
            if self.resource_state is not CanonicalSceneResourceState.NO_SCENE:
                raise ValueError("Unavailable canonical scene must use NO_SCENE resource state.")
        elif not self.canonical_geometry_authoritative:
            raise ValueError("Persistent canonical scene must preserve canonical geometry authority.")

    @property
    def viewer_resource_ready(self) -> bool:
        """Whether 9.3B has a currently materialized canonical surface to consume."""

        return self.resource_state is CanonicalSceneResourceState.MATERIALIZED

    @property
    def scene_text(self) -> str:
        if not self.available:
            return "No persistent scene"
        if self.scene_state == PersistentReconstructionSceneState.DEGRADED.value:
            return "Degraded persistent scene"
        return "Ready persistent scene"

    @property
    def geometry_text(self) -> str:
        if not self.available:
            return "—"
        kind = self.artifact_kind.replace("_", " ").strip().title() or "Canonical geometry"
        return kind

    @property
    def resource_text(self) -> str:
        if self.resource_state is CanonicalSceneResourceState.NO_SCENE:
            return "—"
        if self.resource_state is CanonicalSceneResourceState.NOT_REFERENCED:
            return "Not materialized for viewer"
        if self.resource_state is CanonicalSceneResourceState.MISSING:
            return "Referenced surface missing"
        return "Materialized • 9.3B ready"

    @property
    def scale_text(self) -> str:
        if self.scale_state is CanonicalSceneScaleState.NO_SCENE:
            return "—"
        if self.scale_state is CanonicalSceneScaleState.QUALIFIED:
            prefix = "Qualified metric"
        elif self.scale_state is CanonicalSceneScaleState.PROVISIONAL:
            prefix = "Provisional metric"
        else:
            prefix = "Not qualified"

        if self.scale_relative_uncertainty is None:
            return prefix
        return f"{prefix} • ±{self.scale_relative_uncertainty * 100.0:.2f}% scale"

    @property
    def viewer_scale_banner_text(self) -> str:
        """Safety banner text for the presentation-space Z-up viewport."""

        if self.scale_state is CanonicalSceneScaleState.QUALIFIED:
            return "VIEWER Z-UP • METRIC SCALE QUALIFIED"
        if self.scale_state is CanonicalSceneScaleState.PROVISIONAL:
            return "VIEWER Z-UP • PROVISIONAL METRIC SCALE"
        return "VIEWER Z-UP • ARBITRARY GAUGE"

    @property
    def authority_text(self) -> str:
        if not self.available:
            return "No canonical scene authority"
        return "Canonical geometry • metrology not certified"

    @property
    def state_text(self) -> str:
        if not self.available:
            return "Waiting for persistent reconstruction scene"
        if self.resource_state is CanonicalSceneResourceState.MISSING:
            return "Canonical authority retained • viewer resource unavailable"
        if self.viewer_resource_ready:
            return "Presentation foundation ready • rendering deferred to 9.3B"
        return "Canonical authority ready • surface rendering deferred to 9.3B"


def _canonical_artifact(scene: PersistentReconstructionScene) -> PersistentSceneArtifactRecord:
    artifact = next(
        (item for item in scene.artifacts if item.artifact_id == scene.canonical_geometry_artifact_id),
        None,
    )
    if artifact is None:
        # PersistentReconstructionScene validates this invariant, so reaching
        # this branch means a caller supplied a corrupted/mutated contract.
        raise ValueError("Persistent scene canonical artifact is unavailable.")
    if not artifact.canonical_geometry_authoritative:
        raise ValueError("Persistent scene canonical artifact lost canonical geometry authority.")
    return artifact


def _canonical_surface_reference(
    scene: PersistentReconstructionScene,
    artifact: PersistentSceneArtifactRecord,
) -> PersistentSceneFileReference | None:
    # 9.3A only recognizes the already-qualified Phase 8.9F5 surface OBJ
    # snapshot as the handoff resource for 9.3B.  It does not infer viewer
    # resources from arbitrary payload refs or appearance artifacts.
    if artifact.kind is not ReconstructionArtifactKind.SURFACE_MESH:
        return None
    return next(
        (
            item
            for item in scene.file_references
            if item.artifact_id == artifact.artifact_id
            and item.role == CANONICAL_SURFACE_RESOURCE_ROLE
        ),
        None,
    )


def _resource_state(reference: PersistentSceneFileReference | None) -> CanonicalSceneResourceState:
    if reference is None:
        return CanonicalSceneResourceState.NOT_REFERENCED
    path = Path(reference.path)
    if not path.is_file():
        return CanonicalSceneResourceState.MISSING
    try:
        size_matches = path.stat().st_size == reference.size_bytes
    except OSError:
        return CanonicalSceneResourceState.MISSING
    # This is deliberately not a replacement for persistence integrity
    # verification.  The persistence controller owns SHA-256 validation.
    return (
        CanonicalSceneResourceState.MATERIALIZED
        if size_matches
        else CanonicalSceneResourceState.MISSING
    )


def _authority_matches_scene(
    scene: PersistentReconstructionScene,
    snapshot: PersistentSceneAuthoritySnapshot,
) -> bool:
    return (
        snapshot.scene_id == scene.scene_id
        and snapshot.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
        and snapshot.session_id == scene.session_id
    )


def build_canonical_scene_presentation(
    scene: PersistentReconstructionScene | None,
    authority_snapshot: PersistentSceneAuthoritySnapshot | None = None,
) -> CanonicalScenePresentation:
    """Compose canonical-scene UI state without loading geometry or backends."""

    if scene is None:
        if authority_snapshot is not None:
            raise ValueError("Authority snapshot cannot be presented without its persistent scene.")
        return CanonicalScenePresentation(
            available=False,
            scene_id="",
            scene_state="",
            artifact_id="",
            artifact_kind="",
            provider_id="",
            canonical_geometry_authoritative=False,
            resource_state=CanonicalSceneResourceState.NO_SCENE,
            resource_role="",
            resource_path="",
            calibration_available=False,
            scale_state=CanonicalSceneScaleState.NO_SCENE,
            scale_relative_uncertainty=None,
            scale_unit_to_metre=None,
            metric_evidence_id="",
            metric_assessment_state="",
            certified_metrology=False,
        )
    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene or None.")
    if authority_snapshot is not None and not isinstance(
        authority_snapshot, PersistentSceneAuthoritySnapshot
    ):
        raise TypeError("authority_snapshot must be PersistentSceneAuthoritySnapshot or None.")
    if authority_snapshot is not None and not _authority_matches_scene(scene, authority_snapshot):
        raise ValueError("Persistent authority snapshot is not bound to the active persistent scene.")

    artifact = _canonical_artifact(scene)
    reference = _canonical_surface_reference(scene, artifact)

    scale_state = CanonicalSceneScaleState.NOT_QUALIFIED
    relative_uncertainty: float | None = None
    unit_to_metre: float | None = None
    metric_evidence_id = ""
    metric_assessment_state = ""
    calibration_available = False

    if authority_snapshot is not None:
        calibration_available = authority_snapshot.camera_calibration_profile is not None
        assessment = authority_snapshot.metric_qualification_assessment
        if assessment is not None:
            metric_evidence_id = assessment.evidence_id
            metric_assessment_state = assessment.state.value
            if assessment.authorizes_calibrated_metric_scale:
                scale_state = CanonicalSceneScaleState.QUALIFIED
            elif assessment.authorizes_metric_coordinates:
                scale_state = CanonicalSceneScaleState.PROVISIONAL
            else:
                scale_state = CanonicalSceneScaleState.NOT_QUALIFIED
            relative_uncertainty = assessment.relative_scale_uncertainty
            if assessment.scale_profile is not None:
                unit_to_metre = assessment.scale_profile.gauge.unit_to_metre
        elif authority_snapshot.metric_qualification_evidence is not None:
            # Structurally unreachable for a valid sidecar, but fail closed if
            # a mutated object reaches this presentation layer.
            metric_evidence_id = authority_snapshot.metric_qualification_evidence.evidence_id
            metric_assessment_state = MetricQualificationState.REJECTED.value

    return CanonicalScenePresentation(
        available=True,
        scene_id=scene.scene_id,
        scene_state=scene.state.value,
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.kind.value,
        provider_id=artifact.provider_id,
        canonical_geometry_authoritative=scene.canonical_geometry_authoritative,
        resource_state=_resource_state(reference),
        resource_role="" if reference is None else reference.role,
        resource_path="" if reference is None else reference.path,
        calibration_available=calibration_available,
        scale_state=scale_state,
        scale_relative_uncertainty=relative_uncertainty,
        scale_unit_to_metre=unit_to_metre,
        metric_evidence_id=metric_evidence_id,
        metric_assessment_state=metric_assessment_state,
        certified_metrology=False,
    )
