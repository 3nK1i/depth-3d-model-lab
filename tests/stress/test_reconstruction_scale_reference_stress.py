from __future__ import annotations

import numpy as np

from core.reconstruction import (
    known_dimension_scale_reference,
    reconstruction_points_to_metres,
    scale_reference_profile_from_json,
    scale_reference_profile_to_json,
)


def test_scale_profiles_round_trip_and_convert_stably_across_many_reference_ratios() -> None:
    rng = np.random.default_rng(8602)
    raw_points = rng.normal(size=(250, 3))
    for index in range(200):
        physical_m = 0.005 + index * 0.0005
        measured_units = 0.25 + index * 0.02
        profile = known_dimension_scale_reference(
            profile_id=f"stress-{index}",
            reference_id=f"reference-{index}",
            known_length_m=physical_m,
            measured_length_reconstruction_units=measured_units,
            known_length_relative_uncertainty=0.002,
            measurement_relative_uncertainty=0.015,
            provenance="deterministic stress reference",
        )
        restored = scale_reference_profile_from_json(scale_reference_profile_to_json(profile))
        assert restored.as_dict() == profile.as_dict()
        scaled = reconstruction_points_to_metres(raw_points, restored)
        np.testing.assert_allclose(
            scaled,
            raw_points * (physical_m / measured_units),
            rtol=1e-12,
            atol=1e-12,
        )
        assert scaled.flags.writeable is False
