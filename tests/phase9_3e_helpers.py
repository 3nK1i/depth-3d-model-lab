from __future__ import annotations


def measurement_payload(*, observation_id: str = "phase9-3e:obs:1") -> dict:
    """Deterministic pre-existing measurement evidence for 9.3E presentation tests."""

    return {
        "observation_id": observation_id,
        "measurements": [
            {
                "kind": "width",
                "status": "available",
                "value_si": 0.042,
                "unit_si": "m",
                "method": "existing_measurement_evidence",
                "expanded_uncertainty_si": 0.0015,
                "lower_bound_si": 0.0405,
                "upper_bound_si": 0.0435,
                "reliability": "high",
                "reliability_score": 0.92,
                "valid_observation_count": 5,
                "warnings": (),
                "rejection_reasons": (),
                "uncertainty_basis": "source",
                "uncertainty_qualification": "Expanded source uncertainty",
            },
            {
                "kind": "depth",
                "status": "rejected",
                "value_si": None,
                "unit_si": "m",
                "method": "existing_measurement_evidence",
                "expanded_uncertainty_si": None,
                "lower_bound_si": None,
                "upper_bound_si": None,
                "reliability": "rejected",
                "reliability_score": None,
                "valid_observation_count": 2,
                "warnings": (),
                "rejection_reasons": ("excessive_uncertainty",),
                "uncertainty_basis": "source",
                "uncertainty_qualification": "",
            },
        ],
    }


def measurement_context() -> dict:
    return {
        "depth_scale": "metric",
        "intrinsics_source": "calibrated",
        "calibration_status": "calibrated",
        "confidence_kind": "uncertainty",
    }


def inspection_report(*, session_id: str, object_id: str, include_object: bool = True) -> dict:
    feature_results = []
    if include_object:
        feature_results.append(
            {
                "candidate_object_id": object_id,
                "decision": "warn",
                "findings": ["width_near_tolerance"],
            }
        )
    return {
        "schema_version": 1,
        "inspection_qc_enabled": True,
        "certified_metrology": False,
        "session_id": session_id,
        "reference_id": "phase9-3e:reference:1",
        "decision": "warn",
        "registration": {
            "status": "success",
            "confidence": 0.94,
            "reason": "existing_registration",
        },
        "statistics": {
            "reference_feature_count": 1,
            "matched_feature_count": 1 if include_object else 0,
            "missing_required_count": 0 if include_object else 1,
            "extra_candidate_count": 0,
            "pass_count": 0,
            "warn_count": 1 if include_object else 0,
            "reject_count": 0,
            "insufficient_evidence_count": 0 if include_object else 1,
        },
        "feature_results": feature_results,
        "comparison_metric_available": False,
        "candidate_metric_geometry_status": "arbitrary",
        "reference_metric_geometry_status": "validated_metric",
    }
