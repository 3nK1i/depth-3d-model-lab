"""Per-measurement eligibility rules; object warnings never reject all kinds."""
from __future__ import annotations
from core.measurement.contracts import MeasurementKind, MeasurementStatus, RejectionReason
from core.measurement.geometry import is_orientation_kind
from core.measurement.policy import MeasurementSelectionPolicy
from core.models.metadata import DepthScale
from core.object_geometry.fitting_contracts import ObjectGeometryFitResult, ObjectGeometryFitStatus, OrientationStability


def eligibility_for_kind(*, kind: MeasurementKind, depth_scale: DepthScale, scale_recovery_succeeded: bool, border_contacts: tuple[str,...], fit: ObjectGeometryFitResult, policy: MeasurementSelectionPolicy) -> tuple[MeasurementStatus, tuple[RejectionReason,...], tuple[str,...]]:
    if fit.status is ObjectGeometryFitStatus.REJECTED:
        return MeasurementStatus.REJECTED,(RejectionReason.FITTING_FAILED,),()
    if depth_scale in {DepthScale.SYNTHETIC,DepthScale.UNKNOWN}:
        return MeasurementStatus.UNAVAILABLE,(RejectionReason.INVALID_DEPTH_SCALE,),()
    if depth_scale is DepthScale.RELATIVE and not scale_recovery_succeeded and not is_orientation_kind(kind):
        return MeasurementStatus.UNAVAILABLE,(RejectionReason.RELATIVE_DEPTH_WITHOUT_SCALE,),()
    contacts={c.lower() for c in border_contacts}
    if kind is MeasurementKind.WIDTH and policy.reject_width_on_horizontal_border and contacts & {'left','right'}:
        return MeasurementStatus.UNAVAILABLE,(RejectionReason.BORDER_CONTACT,),('horizontal extent is incomplete',)
    if kind is MeasurementKind.HEIGHT and policy.reject_height_on_vertical_border and contacts & {'top','bottom'}:
        return MeasurementStatus.UNAVAILABLE,(RejectionReason.BORDER_CONTACT,),('vertical extent is incomplete',)
    if kind is MeasurementKind.VOLUME and policy.reject_volume_on_any_border and contacts:
        return MeasurementStatus.UNAVAILABLE,(RejectionReason.BORDER_CONTACT,),('volume requires complete visible extents',)
    if is_orientation_kind(kind):
        stability=fit.object_frame.orientation_stability if fit.object_frame is not None else OrientationStability.UNAVAILABLE
        if stability in {OrientationStability.AMBIGUOUS,OrientationStability.UNAVAILABLE}:
            return MeasurementStatus.UNAVAILABLE,(RejectionReason.AMBIGUOUS_ORIENTATION,),('object orientation is ambiguous',)
        if stability is OrientationStability.WEAK and not policy.allow_weak_orientation:
            return MeasurementStatus.UNAVAILABLE,(RejectionReason.FITTING_FAILED,),('object orientation is weak',)
    return MeasurementStatus.AVAILABLE,(),()
