from types import SimpleNamespace

from core.measurement.contracts import (
    MeasurementKind,
    MeasurementStatus,
    ReliabilityLevel,
    UncertaintyBasis,
    UncertaintyComponent,
    UncertaintyComponentKind,
)
from scripts.run_measurement_inspection_job import _measurement_payload


def test_measurement_payload_uses_step6_uncertainty_component_contract() -> None:
    component = UncertaintyComponent(
        kind=UncertaintyComponentKind.DEPTH,
        contribution_si=0.012,
        method="test-depth",
        sample_count=4,
        lower_contribution_si=0.010,
        upper_contribution_si=0.014,
        note="synthetic",
    )
    estimate = SimpleNamespace(
        kind=MeasurementKind.WIDTH,
        status=MeasurementStatus.AVAILABLE,
        value_si=1.0,
        unit_si="m",
        lower_bound_si=0.97,
        upper_bound_si=1.03,
        reliability=SimpleNamespace(level=ReliabilityLevel.HIGH, score=0.9),
        valid_observation_count=4,
        method="camera_aligned_box",
        rejection_reasons=(),
        warnings=(),
        uncertainty=SimpleNamespace(
            expanded_uncertainty_si=0.03,
            basis=UncertaintyBasis.HEURISTIC_PERTURBATION,
            display_qualification="Estimated Uncertainty — Not Calibrated",
            components=(component,),
        ),
    )

    payload = _measurement_payload(estimate)
    serialized = payload["uncertainty_components"][0]
    assert serialized["contribution_si"] == 0.012
    assert serialized["lower_contribution_si"] == 0.010
    assert serialized["upper_contribution_si"] == 0.014
    assert serialized["note"] == "synthetic"
    assert "magnitude_si" not in serialized
