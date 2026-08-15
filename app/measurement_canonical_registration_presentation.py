"""Phase 9.3J shared measurement -> canonical registration presentation.

The Phase 7.5 inspection bundle remains the source measurement authority.  This
module binds that existing evidence to one persisted canonical semantic object
and one explicitly fingerprinted canonical camera pose.  It never guesses from
class labels or 3D proximity, never creates metric scale, and emits dimension
geometry only when the scene scale and pose/registration uncertainty are all
explicitly qualified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.reconstruction_canonical_cameras import CanonicalCameraTrajectoryPresentation
from app.reconstruction_canonical_semantics import CanonicalSemanticScenePresentation
from core.inspection.contracts import InspectionBoxMethod, InspectionBundleSummary, InspectionObjectSummary
from core.reconstruction.measurement_registration import (
    CanonicalMeasurementRegistrationEvidence,
    CanonicalMeasurementRegistrationResult,
    CanonicalMeasurementRegistrationState,
    CanonicalRegisteredDimension,
    build_registered_dimensions,
)
from core.reconstruction.persistent_authority import PersistentSceneAuthoritySnapshot
from core.reconstruction.persistent_scene import PersistentReconstructionScene


@dataclass(frozen=True, slots=True)
class MeasurementCanonicalRegistrationPresentation:
    """Bounded UI state shared by Measure & Inspect and Reconstruction."""

    scene_id: str
    observation_id: str
    canonical_object_id: str
    canonical_camera_id: str
    state: CanonicalMeasurementRegistrationState
    dimensions: tuple[CanonicalRegisteredDimension, ...]
    reasons: tuple[str, ...]
    source_frame_sha256: str = ""
    source_measurement_authority: str = "Phase 7.5 source measurement + uncertainty"
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", str(self.scene_id).strip())
        object.__setattr__(self, "observation_id", str(self.observation_id).strip())
        object.__setattr__(self, "canonical_object_id", str(self.canonical_object_id).strip())
        object.__setattr__(self, "canonical_camera_id", str(self.canonical_camera_id).strip())
        object.__setattr__(self, "source_frame_sha256", str(self.source_frame_sha256).strip().lower())
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(
            self,
            "reasons",
            tuple(str(value).strip() for value in self.reasons if str(value).strip()),
        )
        if not isinstance(self.state, CanonicalMeasurementRegistrationState):
            raise TypeError("state must be CanonicalMeasurementRegistrationState.")
        if self.certified_metrology:
            raise ValueError("Phase 9.3J canonical registration cannot certify metrology.")
        object.__setattr__(self, "certified_metrology", False)

    @property
    def status_text(self) -> str:
        return self.state.value.replace("_", " ").title()

    @property
    def binding_text(self) -> str:
        if self.canonical_object_id and self.canonical_camera_id:
            return f"{self.canonical_object_id} • {self.canonical_camera_id}"
        if self.canonical_object_id:
            return f"{self.canonical_object_id} • camera not bound"
        return "No exact canonical object/camera binding"

    @property
    def uncertainty_text(self) -> str:
        if not self.dimensions:
            return "No canonical dimension placement uncertainty"
        return "\n".join(
            f"{item.label}: ± {item.endpoint_placement_uncertainty_si * 1000.0:.1f} mm placement"
            for item in self.dimensions
        )

    @property
    def dimensions_text(self) -> str:
        if not self.dimensions:
            return "No canonical dimension geometry"
        return "\n".join(item.value_text for item in self.dimensions)

    @property
    def reason_text(self) -> str:
        if self.reasons:
            return "; ".join(self.reasons)
        if self.state is CanonicalMeasurementRegistrationState.QUALIFIED:
            return "Exact source frame, camera pose, metric scale, and registration uncertainty qualified"
        if self.state is CanonicalMeasurementRegistrationState.PROVISIONAL:
            return "Registration evidence is incomplete for authoritative 3D dimension placement"
        return "Canonical registration evidence unavailable"

    @property
    def authority_text(self) -> str:
        return (
            "Original measurement remains source truth • canonical placement adds registration/scale "
            "uncertainty • not certified metrology"
        )


def build_measurement_canonical_registration_presentation(
    *,
    scene: PersistentReconstructionScene | None,
    canonical_cameras: CanonicalCameraTrajectoryPresentation | None,
    canonical_semantics: CanonicalSemanticScenePresentation | None,
    authority: PersistentSceneAuthoritySnapshot | None,
    inspection_summary: InspectionBundleSummary | None,
    inspection_object: InspectionObjectSummary | None,
) -> MeasurementCanonicalRegistrationPresentation:
    """Fail-closed registration builder for one current inspection object."""

    observation_id = "" if inspection_object is None else inspection_object.observation_id
    scene_id = "" if scene is None else scene.scene_id
    source_digest = "" if inspection_summary is None else inspection_summary.source_frame_sha256

    def unavailable(*reasons: str, object_id: str = "", camera_id: str = ""):
        return MeasurementCanonicalRegistrationPresentation(
            scene_id=scene_id,
            observation_id=observation_id,
            canonical_object_id=object_id,
            canonical_camera_id=camera_id,
            state=CanonicalMeasurementRegistrationState.UNAVAILABLE,
            dimensions=(),
            reasons=tuple(reasons),
            source_frame_sha256=source_digest,
        )

    if scene is None:
        return unavailable("No persistent canonical scene is loaded")
    if inspection_summary is None or inspection_object is None:
        return unavailable("No Measure & Inspect object is selected")
    if not source_digest:
        return unavailable("Inspection bundle predates exact source-frame fingerprinting")
    if canonical_semantics is None:
        return unavailable("Canonical semantic object context is unavailable for this scene")

    matches = [
        item for item in canonical_semantics.objects if observation_id in item.observation_ids
    ]
    if not matches:
        return unavailable("No exact observation-ID match exists in canonical semantics")
    if len(matches) != 1:
        return unavailable("Inspection observation maps to multiple canonical semantic objects")
    semantic = matches[0]

    if canonical_cameras is None:
        return unavailable(
            "Canonical camera trajectory is unavailable",
            object_id=semantic.object_id,
        )
    camera_index = canonical_cameras.camera_index_for_source_frame(source_digest)
    if camera_index is None:
        return unavailable(
            "No uniquely fingerprinted canonical camera matches the inspection source frame",
            object_id=semantic.object_id,
        )
    camera_id = canonical_cameras.camera_ids[camera_index]

    box = inspection_object.box_payload(InspectionBoxMethod.AUTO)
    if not box:
        return unavailable(
            "Selected inspection object has no persisted 3D measurement box",
            object_id=semantic.object_id,
            camera_id=camera_id,
        )
    coordinate_frame = str(box.get("coordinate_frame_id", "")).strip()
    if not coordinate_frame:
        return unavailable(
            "Measurement box predates explicit coordinate-frame persistence",
            object_id=semantic.object_id,
            camera_id=camera_id,
        )
    box_fit_method = str(box.get("fit_method", "")).strip()
    if not box_fit_method:
        return unavailable(
            "Measurement box predates explicit fit-method persistence",
            object_id=semantic.object_id,
            camera_id=camera_id,
        )
    for measurement in inspection_object.measurements:
        kind = str(measurement.get("kind", "")).strip().lower()
        status = str(measurement.get("status", "")).strip().lower()
        if kind in {"width", "height", "depth"} and status == "available":
            if str(measurement.get("method", "")).strip() != box_fit_method:
                return unavailable(
                    "Source dimension method does not match the persisted 3D measurement box",
                    object_id=semantic.object_id,
                    camera_id=camera_id,
                )

    if authority is None or authority.scene_id != scene.scene_id:
        return unavailable(
            "No scene-bound metric authority snapshot is available",
            object_id=semantic.object_id,
            camera_id=camera_id,
        )
    assessment = authority.metric_qualification_assessment
    if not authority.metric_coordinates_allowed or assessment is None or assessment.scale_profile is None:
        return unavailable(
            "Canonical scene metric scale is not qualified",
            object_id=semantic.object_id,
            camera_id=camera_id,
        )
    scale_profile = assessment.scale_profile
    unit_to_metre = scale_profile.gauge.unit_to_metre
    relative_scale_uncertainty = assessment.relative_scale_uncertainty
    if unit_to_metre is None or relative_scale_uncertainty is None:
        return unavailable(
            "Qualified metric authority lacks explicit scale/uncertainty",
            object_id=semantic.object_id,
            camera_id=camera_id,
        )
    if assessment.reconstruction_frame_id and canonical_cameras.reconstruction_frame_id:
        if assessment.reconstruction_frame_id != canonical_cameras.reconstruction_frame_id:
            return unavailable(
                "Metric authority reconstruction frame does not match canonical camera frame",
                object_id=semantic.object_id,
                camera_id=camera_id,
            )

    translation_u, rotation_u = canonical_cameras.pose_uncertainty_for_camera(camera_index)
    state = CanonicalMeasurementRegistrationState.QUALIFIED
    reasons: tuple[str, ...] = ()
    if translation_u is None or rotation_u is None:
        state = CanonicalMeasurementRegistrationState.PROVISIONAL
        reasons = ("Canonical camera pose uncertainty is not quantified",)

    evidence = CanonicalMeasurementRegistrationEvidence(
        evidence_id=f"cmr-{scene.scene_id}-{observation_id}",
        scene_id=scene.scene_id,
        canonical_object_id=semantic.object_id,
        source_observation_id=observation_id,
        source_frame_sha256=source_digest,
        canonical_camera_id=camera_id,
        measurement_coordinate_frame_id=coordinate_frame,
        reconstruction_frame_id=canonical_cameras.reconstruction_frame_id,
        state=state,
        T_reconstruction_from_measurement_camera=canonical_cameras.matrices_reconstruction[camera_index],
        unit_to_metre=float(unit_to_metre),
        relative_scale_uncertainty=float(relative_scale_uncertainty),
        translation_uncertainty_m=translation_u,
        rotation_uncertainty_rad=rotation_u,
        provenance=(
            "inspection source-frame SHA-256 -> persisted canonical camera pose -> "
            "scene-bound Phase 9 metric authority"
        ),
        reasons=reasons,
        certified_metrology=False,
    )
    result: CanonicalMeasurementRegistrationResult
    if state is CanonicalMeasurementRegistrationState.QUALIFIED:
        result = build_registered_dimensions(
            evidence,
            box_payload=box,
            measurement_entries=inspection_object.measurements,
        )
    else:
        result = CanonicalMeasurementRegistrationResult(evidence=evidence, dimensions=())

    return MeasurementCanonicalRegistrationPresentation(
        scene_id=scene.scene_id,
        observation_id=observation_id,
        canonical_object_id=semantic.object_id,
        canonical_camera_id=camera_id,
        state=state,
        dimensions=result.dimensions,
        reasons=reasons,
        source_frame_sha256=source_digest,
        certified_metrology=False,
    )


__all__ = [
    "MeasurementCanonicalRegistrationPresentation",
    "build_measurement_canonical_registration_presentation",
]
