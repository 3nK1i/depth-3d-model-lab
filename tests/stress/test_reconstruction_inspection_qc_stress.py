from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.reconstruction import (
    FeatureTolerance,
    InspectionDecision,
    RegistrationInspectionPolicy,
    build_reference_assembly,
    inspect_semantic_geometry,
)
from scripts.run_registration_inspection_smoke import (
    _rot_z,
    _semantic_fixture,
    _transform_candidate,
)


def test_hundreds_of_semantic_features_remain_bounded_and_deterministic() -> None:
    surface, semantic, _ = _semantic_fixture()
    body = next(obj for obj in semantic.objects if "connector-main" in obj.track_ids)
    top = next(obj for obj in semantic.objects if "pin-top" in obj.track_ids)
    bottom = next(obj for obj in semantic.objects if "pin-bottom" in obj.track_ids)

    duplicates = [body, bottom]
    for index in range(250):
        duplicates.append(
            replace(
                top,
                object_id=f"stress-pin-{index:04d}",
                track_ids=(f"stress-pin-{index:04d}",),
            )
        )
    semantic = replace(
        semantic,
        objects=tuple(duplicates),
        statistics=replace(semantic.statistics, persistent_object_count=len(duplicates)),
    )
    tolerance = FeatureTolerance(position_warn=0.08, position_reject=0.16)
    reference = build_reference_assembly(
        surface,
        semantic,
        reference_id="stress-reference",
        default_tolerance=tolerance,
        registration_anchor_feature_ids=("connector-main",),
        maximum_anchor_sample_points=256,
    )
    candidate_surface, candidate_semantic = _transform_candidate(
        surface,
        semantic,
        scale_factor=1.15,
        rotation=_rot_z(7.0),
        translation=np.asarray([0.2, 0.1, -0.1]),
    )
    policy = RegistrationInspectionPolicy(
        maximum_reference_features=512,
        maximum_candidate_objects=512,
        maximum_anchor_sample_points=256,
    )
    first = inspect_semantic_geometry(reference, candidate_surface, candidate_semantic, policy=policy)
    second = inspect_semantic_geometry(reference, candidate_surface, candidate_semantic, policy=policy)

    assert first.decision is InspectionDecision.PASS
    assert first.statistics.reference_feature_count == 252
    assert first.statistics.matched_feature_count == 252
    assert first.statistics.extra_candidate_count == 0
    assert first.as_dict() == second.as_dict()
