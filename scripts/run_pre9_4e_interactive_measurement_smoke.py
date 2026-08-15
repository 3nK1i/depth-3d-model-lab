#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_interactive_measurement import pick_projected_surface_point
from core.measurement.interactive_distance import (
    InteractiveDistanceState,
    estimate_depth_pixel_distance,
    estimate_reconstruction_distance,
)


def main() -> int:
    depth = np.full((80, 100), 2.0, dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    K = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    image = estimate_depth_pixel_distance(
        depth=depth,
        depth_valid_mask=valid,
        intrinsics_matrix=K,
        pixel_a_xy=(30, 40),
        pixel_b_xy=(70, 40),
        depth_scale="metric",
        intrinsics_source="model_predicted",
        calibration_status="not_calibrated",
    )
    reconstruction = estimate_reconstruction_distance(
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(2.0, 0.0, 0.0),
        unit_to_metre=0.5,
        relative_scale_uncertainty=0.04,
        scale_state="provisional",
        calibration_available=False,
    )
    arbitrary = estimate_reconstruction_distance(
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(3.0, 4.0, 0.0),
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        scale_state="not_qualified",
        calibration_available=False,
    )
    pick = pick_projected_surface_point(
        vertices_viewer=np.asarray([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float64),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        mapped_homogeneous=np.asarray([[10, 10, 0.2, 1], [110, 10, 0.2, 1], [10, 110, 0.2, 1]], dtype=np.float64),
        click_xy=(35, 35),
    )
    ok = (
        image.state is InteractiveDistanceState.ESTIMATED
        and image.distance_m is not None
        and image.expanded_uncertainty_m is not None
        and reconstruction.state is InteractiveDistanceState.ESTIMATED
        and reconstruction.distance_m is not None
        and arbitrary.state is InteractiveDistanceState.UNAVAILABLE
        and arbitrary.distance_m is None
        and pick is not None
        and not image.certified_metrology
        and not reconstruction.certified_metrology
    )
    payload = {
        "step": "pre-9.4E-interactive-measurement",
        "status": "PASS" if ok else "FAIL",
        "measure_and_inspect": {
            "state": image.state.value,
            "estimated_distance_m": image.distance_m,
            "expanded_uncertainty_m": image.expanded_uncertainty_m,
            "certified_metrology": image.certified_metrology,
        },
        "reconstruction_provisional": {
            "state": reconstruction.state.value,
            "estimated_distance_m": reconstruction.distance_m,
            "expanded_uncertainty_m": reconstruction.expanded_uncertainty_m,
            "certified_metrology": reconstruction.certified_metrology,
        },
        "reconstruction_arbitrary": {
            "state": arbitrary.state.value,
            "metric_value_exposed": arbitrary.distance_m is not None,
            "native_distance_units": arbitrary.distance_source_units,
        },
        "canonical_surface_pick": {
            "resolved": pick is not None,
            "face_index": pick.face_index if pick is not None else None,
        },
        "gaussian_measurement_authority": False,
        "nerf_measurement_authority": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
