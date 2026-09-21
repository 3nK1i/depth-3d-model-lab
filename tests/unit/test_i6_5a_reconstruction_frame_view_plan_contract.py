from dataclasses import replace

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    I65A_CAMERA_AUTHORITY_OWNER,
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewPlanResolution,
    FrameViewPlanRealization,
    FrameViewRecord,
    FrameViewRecordDisposition,
    FrameViewSourceProvenance,
)


SHA = "a" * 64


def _source(**overrides):
    values = dict(
        source_id="image-sequence-1234",
        source_kind="image_sequence",
        source_uri="/dataset/images",
        source_fingerprint_sha256=SHA,
        estimated_total_views=8,
        expected_frame_rate_hz=30.0,
    )
    values.update(overrides)
    return FrameViewSourceProvenance(**values)


def _request(mode=FrameViewPlanMode.ADAPTIVE, requested_max_views=None):
    return FrameViewPlanRequest(mode=mode, requested_max_views=requested_max_views)


def _resolution(**overrides):
    values = dict(
        source=_source(),
        request=_request(),
        effective_target_views=6,
        resource_target_views=6,
        reason_codes=("adaptive_plan_resolved", "source_span_must_be_preserved"),
    )
    values.update(overrides)
    return FrameViewPlanResolution(**values)


def _records():
    values = []
    for index in range(8):
        if index in {0, 1, 3, 5, 6, 7}:
            disposition = FrameViewRecordDisposition.PLANNED
            reasons = ("selected_by_effective_plan",)
        else:
            disposition = FrameViewRecordDisposition.ELIGIBLE_NOT_PLANNED
            reasons = ("not_selected_by_effective_target",)
        values.append(
            FrameViewRecord(
                source_view_id=f"view-{index:03d}",
                sequence_index=index,
                source_locator=f"/dataset/images/{index:03d}.jpg",
                source_timestamp_ns=index * 33_333_333,
                disposition=disposition,
                reason_codes=reasons,
            )
        )
    return tuple(values)


def test_source_provenance_is_dependency_light_and_explicit():
    source = _source()
    payload = source.as_dict()
    assert payload["source_fingerprint_sha256"] == SHA
    assert payload["estimated_total_views"] == 8
    assert payload["expected_frame_rate_hz"] == 30.0


def test_source_fingerprint_must_be_sha256():
    with pytest.raises(ReconstructionValidationError):
        _source(source_fingerprint_sha256="not-a-sha")


def test_manual_limit_request_requires_at_least_three_views():
    with pytest.raises(ReconstructionValidationError):
        _request(FrameViewPlanMode.MANUAL_LIMIT, 2)
    request = _request(FrameViewPlanMode.MANUAL_LIMIT, 5)
    assert request.requested_max_views == 5


def test_adaptive_and_all_eligible_requests_do_not_carry_manual_limits():
    with pytest.raises(ReconstructionValidationError):
        _request(FrameViewPlanMode.ADAPTIVE, 5)
    with pytest.raises(ReconstructionValidationError):
        _request(FrameViewPlanMode.ALL_ELIGIBLE, 5)


def test_resolution_separates_requested_and_effective_target():
    plan = _resolution()
    assert plan.request.requested_max_views is None
    assert plan.effective_target_views == 6
    assert plan.resource_target_views == 6
    assert plan.permanent_global_view_cap is None


def test_manual_limit_resolution_cannot_exceed_request_or_known_source_count():
    request = _request(FrameViewPlanMode.MANUAL_LIMIT, 5)
    with pytest.raises(ReconstructionValidationError):
        _resolution(request=request, effective_target_views=6, resource_target_views=None)
    with pytest.raises(ReconstructionValidationError):
        _resolution(source=_source(estimated_total_views=4), request=request, effective_target_views=5, resource_target_views=None)


def test_all_eligible_resolution_requires_no_effective_cap():
    request = _request(FrameViewPlanMode.ALL_ELIGIBLE)
    with pytest.raises(ReconstructionValidationError):
        _resolution(request=request, effective_target_views=8, resource_target_views=None)
    plan = _resolution(request=request, effective_target_views=None, resource_target_views=None)
    assert plan.effective_target_views is None


def test_view_records_require_explicit_reason_codes():
    with pytest.raises(ReconstructionValidationError):
        FrameViewRecord(
            source_view_id="view-0",
            sequence_index=0,
            source_locator="/dataset/0.jpg",
            source_timestamp_ns=0,
            disposition=FrameViewRecordDisposition.INELIGIBLE,
            reason_codes=(),
        )


def test_realization_tracks_source_eligible_and_planned_counts_without_sfm_claims():
    realized = FrameViewPlanRealization(resolution=_resolution(), records=_records())
    assert realized.observed_source_view_count == 8
    assert realized.eligible_view_count == 8
    assert realized.planned_view_count == 6
    assert realized.planned_source_view_ids == (
        "view-000", "view-001", "view-003", "view-005", "view-006", "view-007"
    )
    payload = realized.as_dict()
    assert payload["authority"]["camera_pose_authority_owner"] == I65A_CAMERA_AUTHORITY_OWNER
    assert payload["authority"]["planned_views_are_registered_cameras"] is False
    assert payload["authority"]["pycolmap_registration_outcome_claimed"] is False


def test_ineligible_views_are_not_counted_as_eligible_or_planned():
    records = list(_records())
    records[2] = replace(
        records[2],
        disposition=FrameViewRecordDisposition.INELIGIBLE,
        reason_codes=("decode_failed",),
    )
    realized = FrameViewPlanRealization(resolution=_resolution(), records=tuple(records))
    assert realized.eligible_view_count == 7
    assert realized.planned_view_count == 6


def test_planned_count_cannot_exceed_effective_target():
    resolution = _resolution(effective_target_views=5, resource_target_views=5)
    with pytest.raises(ReconstructionValidationError):
        FrameViewPlanRealization(resolution=resolution, records=_records())


def test_records_must_preserve_source_sequence_order_and_unique_ids():
    records = list(_records())
    records[1], records[2] = records[2], records[1]
    with pytest.raises(ReconstructionValidationError):
        FrameViewPlanRealization(resolution=_resolution(), records=tuple(records))
    records = list(_records())
    records[1] = replace(records[1], source_view_id=records[0].source_view_id)
    with pytest.raises(ReconstructionValidationError):
        FrameViewPlanRealization(resolution=_resolution(), records=tuple(records))


def test_resolution_digest_is_deterministic_and_provenance_sensitive():
    first = _resolution()
    second = _resolution()
    assert first.plan_input_sha256 == second.plan_input_sha256
    changed = _resolution(reason_codes=("adaptive_plan_resolved", "different_reason"))
    assert changed.plan_input_sha256 != first.plan_input_sha256


def test_realization_digest_is_deterministic_and_record_sensitive():
    first = FrameViewPlanRealization(resolution=_resolution(), records=_records())
    second = FrameViewPlanRealization(resolution=_resolution(), records=_records())
    assert first.plan_provenance_sha256 == second.plan_provenance_sha256
    records = list(_records())
    records[2] = replace(records[2], reason_codes=("alternative_exclusion_reason",))
    changed = FrameViewPlanRealization(resolution=_resolution(), records=tuple(records))
    assert changed.plan_provenance_sha256 != first.plan_provenance_sha256
