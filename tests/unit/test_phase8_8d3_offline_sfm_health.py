from __future__ import annotations

import pytest

from app.offline_reconstruction_quality import build_offline_sfm_health_presentation


def _summary(*, status: str = "accepted", usable: bool = True, reasons=()):
    input_keyframes = 80
    registered = 80
    landmarks = 50_000
    observations = 250_000
    return {
        "input_keyframes": input_keyframes,
        "registered_keyframes": registered,
        "landmarks": landmarks,
        "observations": observations,
        "mean_reprojection_error_px": 0.52,
        "quality_assessment": {
            "status": status,
            "usable": usable,
            "registration_ratio": registered / input_keyframes,
            "landmarks_per_registered_keyframe": landmarks / registered,
            "mean_track_length": observations / landmarks,
            "mean_reprojection_error_px": 0.52,
            "reasons": list(reasons),
        },
    }


def test_d3_accepted_health_is_backend_reported_and_unit_honest() -> None:
    presentation = build_offline_sfm_health_presentation(_summary())
    assert presentation.status == "accepted"
    assert presentation.status_label == "ACCEPTED"
    assert presentation.status_kind == "ready"
    assert presentation.authoritative_assessment is True
    assert presentation.registration_text == "80/80 • 100.0%"
    assert presentation.reprojection_text == "0.520 px mean"
    assert presentation.track_support_text == "5.00 views / point"
    assert presentation.density_text == "625.0 points / camera"
    assert presentation.gate_text == "No sparse-SfM gate warnings"
    assert "not a metric-accuracy score" in presentation.gate_tooltip


def test_d3_degraded_health_preserves_backend_reasons_without_inventing_score() -> None:
    presentation = build_offline_sfm_health_presentation(
        _summary(
            status="degraded",
            usable=True,
            reasons=("Mean sparse track length 2.10 is weak.", "Registration coverage is marginal."),
        )
    )
    assert presentation.status_kind == "warning"
    assert presentation.gate_text == "Mean sparse track length 2.10 is weak. (+1)"
    assert "Registration coverage is marginal." in presentation.gate_tooltip


def test_d3_rejected_health_requires_usable_false() -> None:
    presentation = build_offline_sfm_health_presentation(
        _summary(status="rejected", usable=False, reasons=("Sparse reconstruction rejected.",))
    )
    assert presentation.status_label == "REJECTED"
    assert presentation.status_kind == "error"
    assert presentation.usable is False


def test_d3_missing_quality_assessment_is_unavailable_not_synthesized() -> None:
    summary = _summary()
    summary.pop("quality_assessment")
    presentation = build_offline_sfm_health_presentation(summary)
    assert presentation.status == "unavailable"
    assert presentation.authoritative_assessment is False
    assert presentation.registration_text == "80/80 • 100.0%"
    assert presentation.reprojection_text == "0.520 px mean"
    assert presentation.gate_text == "Backend assessment unavailable"


def test_d3_rejects_stale_quality_ratio_that_disagrees_with_summary() -> None:
    summary = _summary()
    summary["quality_assessment"]["registration_ratio"] = 0.5
    with pytest.raises(ValueError, match="registration ratio"):
        build_offline_sfm_health_presentation(summary)


def test_d3_rejects_status_usable_mismatch() -> None:
    with pytest.raises(ValueError, match="usable"):
        build_offline_sfm_health_presentation(_summary(status="rejected", usable=True))

def test_d3_accepts_metadata_roundtrip_empty_reasons_object_only_as_empty_sequence() -> None:
    # Accepted backend results legitimately have reasons=[].  The generic
    # reconstruction metadata round-trip represents that empty array as {},
    # because the immutable empty tuple is ambiguous on reverse conversion.
    summary = _summary()
    summary["quality_assessment"]["reasons"] = {}
    presentation = build_offline_sfm_health_presentation(summary)
    assert presentation.status == "accepted"
    assert presentation.authoritative_assessment is True
    assert presentation.reasons == ()
    assert presentation.gate_text == "No sparse-SfM gate warnings"


def test_d3_still_rejects_nonempty_mapping_for_reasons() -> None:
    summary = _summary()
    summary["quality_assessment"]["reasons"] = {"unexpected": "value"}
    with pytest.raises(ValueError, match="reasons must be a sequence"):
        build_offline_sfm_health_presentation(summary)

