"""Offline sparse-SfM health presentation for Phase 8.8D3.

D3 presents the quality assessment already produced by the qualified SfM
backend.  It does not recompute reconstruction acceptance thresholds in the UI,
and it never treats sparse-SfM health as metric accuracy or metrology evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence


_ALLOWED_STATUSES = {"accepted", "degraded", "rejected"}


@dataclass(frozen=True, slots=True)
class OfflineSfmHealthPresentation:
    """Validated, unit-honest presentation of one sparse-SfM quality gate."""

    status: str
    usable: bool | None
    authoritative_assessment: bool
    input_keyframes: int | None
    registered_keyframes: int | None
    landmarks: int | None
    observations: int | None
    registration_ratio: float | None
    landmarks_per_registered_keyframe: float | None
    mean_track_length: float | None
    mean_reprojection_error_px: float | None
    reasons: tuple[str, ...] = ()
    error: str = ""

    @property
    def status_label(self) -> str:
        return {
            "accepted": "ACCEPTED",
            "degraded": "DEGRADED",
            "rejected": "REJECTED",
            "unavailable": "UNAVAILABLE",
        }.get(self.status, "UNAVAILABLE")

    @property
    def status_kind(self) -> str:
        return {
            "accepted": "ready",
            "degraded": "warning",
            "rejected": "error",
            "unavailable": "warning",
        }.get(self.status, "warning")

    @property
    def registration_text(self) -> str:
        registered = self.registered_keyframes
        total = self.input_keyframes
        ratio = self.registration_ratio
        if registered is None:
            return "Not reported"
        if total is not None and total > 0 and ratio is not None:
            return f"{registered}/{total} • {ratio:.1%}"
        if total is not None and total > 0:
            return f"{registered}/{total}"
        if ratio is not None:
            return f"{registered} • {ratio:.1%}"
        return str(registered)

    @property
    def reprojection_text(self) -> str:
        if self.mean_reprojection_error_px is None:
            return "Not reported"
        return f"{self.mean_reprojection_error_px:.3f} px mean"

    @property
    def track_support_text(self) -> str:
        if self.mean_track_length is None:
            return "Not reported"
        return f"{self.mean_track_length:.2f} views / point"

    @property
    def density_text(self) -> str:
        if self.landmarks_per_registered_keyframe is None:
            return "Not reported"
        return f"{self.landmarks_per_registered_keyframe:.1f} points / camera"

    @property
    def support_text(self) -> str:
        parts = [
            item
            for item in (self.track_support_text, self.density_text)
            if item != "Not reported"
        ]
        return " • ".join(parts) if parts else "Not reported"

    @property
    def gate_text(self) -> str:
        if self.status == "unavailable":
            return "Backend assessment unavailable"
        if self.reasons:
            return self.reasons[0] if len(self.reasons) == 1 else f"{self.reasons[0]} (+{len(self.reasons) - 1})"
        return "No sparse-SfM gate warnings"

    @property
    def gate_tooltip(self) -> str:
        if self.error:
            return self.error
        if self.reasons:
            return "\n".join(f"• {reason}" for reason in self.reasons)
        return (
            "Operational sparse-SfM health assessment reported by the qualified backend. "
            "This is not a metric-accuracy score and does not certify metrology."
        )


def build_offline_sfm_health_presentation(
    sfm_summary: Mapping[str, object],
) -> OfflineSfmHealthPresentation:
    """Validate and present the backend-reported sparse-SfM quality assessment.

    D3 intentionally consumes ``quality_assessment`` rather than duplicating the
    backend's acceptance policy inside the Qt layer.  Summary statistics are
    cross-checked when available so stale/mismatched telemetry is shown as
    unavailable instead of silently trusted.
    """

    if not isinstance(sfm_summary, Mapping):
        raise TypeError("sfm_summary must be a mapping.")

    input_keyframes = _optional_nonnegative_int(sfm_summary.get("input_keyframes"), "input_keyframes")
    registered = _optional_nonnegative_int(sfm_summary.get("registered_keyframes"), "registered_keyframes")
    landmarks = _optional_nonnegative_int(sfm_summary.get("landmarks"), "landmarks")
    observations = _optional_nonnegative_int(sfm_summary.get("observations"), "observations")
    summary_error = _optional_nonnegative_float(
        sfm_summary.get("mean_reprojection_error_px"),
        "mean_reprojection_error_px",
    )

    raw = sfm_summary.get("quality_assessment")
    if not isinstance(raw, Mapping) or not raw:
        return OfflineSfmHealthPresentation(
            status="unavailable",
            usable=None,
            authoritative_assessment=False,
            input_keyframes=input_keyframes,
            registered_keyframes=registered,
            landmarks=landmarks,
            observations=observations,
            registration_ratio=_derived_ratio(registered, input_keyframes),
            landmarks_per_registered_keyframe=_derived_density(landmarks, registered),
            mean_track_length=_derived_track_length(observations, landmarks),
            mean_reprojection_error_px=summary_error,
            reasons=(),
            error="The offline result does not contain a backend sparse-SfM quality assessment.",
        )

    status = str(raw.get("status", "")).strip().lower()
    if status not in _ALLOWED_STATUSES:
        raise ValueError(f"Unsupported sparse-SfM quality status: {status!r}.")

    usable_value = raw.get("usable")
    if not isinstance(usable_value, bool):
        raise ValueError("quality_assessment.usable must be boolean.")
    expected_usable = status != "rejected"
    if usable_value != expected_usable:
        raise ValueError("quality_assessment.usable does not match quality status.")

    ratio = _required_probability(raw.get("registration_ratio"), "quality_assessment.registration_ratio")
    density = _required_nonnegative_float(
        raw.get("landmarks_per_registered_keyframe"),
        "quality_assessment.landmarks_per_registered_keyframe",
    )
    track = _required_nonnegative_float(raw.get("mean_track_length"), "quality_assessment.mean_track_length")
    error = _optional_nonnegative_float(
        raw.get("mean_reprojection_error_px"),
        "quality_assessment.mean_reprojection_error_px",
    )
    reasons = _reasons(raw.get("reasons", ()))

    derived_ratio = _derived_ratio(registered, input_keyframes)
    if derived_ratio is not None and abs(ratio - derived_ratio) > 1e-6:
        raise ValueError("Backend quality registration ratio does not match sparse-SfM summary counts.")
    derived_density = _derived_density(landmarks, registered)
    if derived_density is not None and not _near(density, derived_density):
        raise ValueError("Backend quality landmark density does not match sparse-SfM summary counts.")
    derived_track = _derived_track_length(observations, landmarks)
    if derived_track is not None and not _near(track, derived_track):
        raise ValueError("Backend quality mean track length does not match sparse-SfM summary counts.")
    if error is not None and summary_error is not None and not _near(error, summary_error):
        raise ValueError("Backend quality reprojection error does not match sparse-SfM summary telemetry.")

    return OfflineSfmHealthPresentation(
        status=status,
        usable=usable_value,
        authoritative_assessment=True,
        input_keyframes=input_keyframes,
        registered_keyframes=registered,
        landmarks=landmarks,
        observations=observations,
        registration_ratio=ratio,
        landmarks_per_registered_keyframe=density,
        mean_track_length=track,
        mean_reprojection_error_px=error,
        reasons=reasons,
    )


def unavailable_offline_sfm_health(error: str = "") -> OfflineSfmHealthPresentation:
    return OfflineSfmHealthPresentation(
        status="unavailable",
        usable=None,
        authoritative_assessment=False,
        input_keyframes=None,
        registered_keyframes=None,
        landmarks=None,
        observations=None,
        registration_ratio=None,
        landmarks_per_registered_keyframe=None,
        mean_track_length=None,
        mean_reprojection_error_px=None,
        reasons=(),
        error=str(error).strip(),
    )


def _optional_nonnegative_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer or null.")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and isfinite(value) and value.is_integer():
        result = int(value)
    else:
        raise ValueError(f"{field_name} must be a non-negative integer or null.")
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return result


def _optional_nonnegative_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _required_nonnegative_float(value, field_name)


def _required_nonnegative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return result


def _required_probability(value: object, field_name: str) -> float:
    result = _required_nonnegative_float(value, field_name)
    if result > 1.0:
        raise ValueError(f"{field_name} must lie within [0, 1].")
    return result


def _reasons(value: object) -> tuple[str, ...]:
    # Reconstruction metadata normalizes JSON-compatible containers into
    # immutable tuples.  The generic reverse conversion cannot distinguish an
    # empty JSON array from an empty JSON object because both normalize to ().
    # For this field only, an empty mapping is therefore the known round-trip
    # representation of backend ``reasons: []``.  Non-empty mappings remain
    # invalid so D3 never weakens the quality-assessment schema.
    if isinstance(value, Mapping):
        if not value:
            return ()
        raise ValueError("quality_assessment.reasons must be a sequence of strings.")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("quality_assessment.reasons must be a sequence of strings.")
    reasons = tuple(str(item).strip() for item in value if str(item).strip())
    return reasons


def _derived_ratio(registered: int | None, total: int | None) -> float | None:
    if registered is None or total is None or total <= 0:
        return None
    if registered > total:
        raise ValueError("registered_keyframes cannot exceed input_keyframes.")
    return registered / total


def _derived_density(landmarks: int | None, registered: int | None) -> float | None:
    if landmarks is None or registered is None or registered <= 0:
        return None
    return landmarks / registered


def _derived_track_length(observations: int | None, landmarks: int | None) -> float | None:
    if observations is None or landmarks is None or landmarks <= 0:
        return None
    return observations / landmarks


def _near(left: float, right: float) -> bool:
    tolerance = max(1e-6, 1e-6 * max(abs(left), abs(right), 1.0))
    return abs(left - right) <= tolerance


__all__ = [
    "OfflineSfmHealthPresentation",
    "build_offline_sfm_health_presentation",
    "unavailable_offline_sfm_health",
]
