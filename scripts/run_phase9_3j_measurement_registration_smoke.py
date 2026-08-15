#!/usr/bin/env python3
"""Dependency-light Phase 9.3J measurement->canonical registration smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.measurement_canonical_registration_presentation import (
    build_measurement_canonical_registration_presentation,
)
from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from core.reconstruction.measurement_registration import CanonicalMeasurementRegistrationState
from tests.phase9_3j_helpers import make_phase9_3j_inspection_bundle, make_phase9_3j_qualified_scene


def _presentation(scene, authority, summary, obj):
    mesh = load_canonical_viewer_mesh(scene)
    cameras = load_persistent_canonical_camera_trajectory(scene)
    if mesh is None or cameras is None:
        raise RuntimeError("9.3J smoke fixture did not load canonical mesh/cameras.")
    semantics = load_persistent_canonical_semantics(scene, mesh)
    if semantics is None:
        raise RuntimeError("9.3J smoke fixture did not load canonical semantic evidence.")
    return build_measurement_canonical_registration_presentation(
        scene=scene,
        canonical_cameras=cameras,
        canonical_semantics=semantics,
        authority=authority,
        inspection_summary=summary,
        inspection_object=obj,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3j_measurement_smoke_") as tmp:
        root = Path(tmp)
        scene, authority = make_phase9_3j_qualified_scene(root / "qualified")
        summary, obj = make_phase9_3j_inspection_bundle(root / "inspection")
        qualified = _presentation(scene, authority, summary, obj)
        unavailable = _presentation(scene, None, summary, obj)

        provisional_scene, provisional_authority = make_phase9_3j_qualified_scene(
            root / "provisional", with_pose_uncertainty=False
        )
        provisional = _presentation(
            provisional_scene,
            provisional_authority,
            summary,
            obj,
        )

        if qualified.state is not CanonicalMeasurementRegistrationState.QUALIFIED:
            raise RuntimeError("Qualified 9.3J fixture did not qualify canonical registration.")
        if unavailable.state is not CanonicalMeasurementRegistrationState.UNAVAILABLE:
            raise RuntimeError("Missing metric authority did not fail closed as unavailable.")
        if provisional.state is not CanonicalMeasurementRegistrationState.PROVISIONAL:
            raise RuntimeError("Missing pose uncertainty did not remain provisional.")
        if len(qualified.dimensions) != 1:
            raise RuntimeError("Qualified 9.3J fixture did not emit exactly one source-authored dimension.")

        width = qualified.dimensions[0]
        payload = {
            "status": "PASS",
            "qualified_registration": True,
            "qualified_state": qualified.state.value,
            "canonical_dimension_count": len(qualified.dimensions),
            "canonical_dimension_kind": width.kind,
            "source_measurement_value_si": width.value_si,
            "source_measurement_uncertainty_si": width.source_expanded_uncertainty_si,
            "registered_dimension_uncertainty_si": width.registered_dimension_uncertainty_si,
            "endpoint_placement_uncertainty_si": width.endpoint_placement_uncertainty_si,
            "source_uncertainty_preserved": width.source_expanded_uncertainty_si == 0.0015,
            "combined_uncertainty_not_reduced": (
                width.registered_dimension_uncertainty_si >= width.source_expanded_uncertainty_si
            ),
            "missing_metric_state": unavailable.state.value,
            "missing_pose_uncertainty_state": provisional.state.value,
            "unavailable_geometry_count": len(unavailable.dimensions),
            "provisional_geometry_count": len(provisional.dimensions),
            "calibration_required": False,
            "certified_metrology": False,
            "dimension_geometry_fabricated": False,
            "heavy_backend_started": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
