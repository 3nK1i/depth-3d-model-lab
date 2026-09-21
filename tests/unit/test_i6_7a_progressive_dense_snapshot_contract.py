from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_provider_selection_contract import (
    PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION,
)
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.reconstruction_progressive_dense_contract import (
    I67A_CAMERA_AUTHORITY_OWNER,
    I67A_SCALE_STATUS,
    I67A_SNAPSHOT_AUTHORITY,
    ProgressiveDenseGeometryPreview,
    ProgressiveDensePointSample,
    ProgressiveDenseSnapshot,
    ProgressiveDenseStage,
    ProgressiveDenseStatistics,
)


SESSION_ID = "11111111-2222-4333-8444-555555555555"
DENSE_RUN_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
REQUEST_SHA = "a" * 64
SPARSE_SHA = "b" * 64


def _point(index: int = 0) -> ProgressiveDensePointSample:
    return ProgressiveDensePointSample(
        sample_index=index,
        position_reconstruction=np.asarray([1.0 + index, 2.0, 3.0]),
        color_rgb=(10, 20, 30),
        confidence=0.8,
        view_support=3,
    )


def _statistics(*, points: int = 100, processed: int = 6) -> ProgressiveDenseStatistics:
    return ProgressiveDenseStatistics(
        input_views=24,
        selected_views=18,
        processed_views=processed,
        current_dense_points=points,
        mean_confidence=0.75,
        mean_view_support=3.5,
    )


def _snapshot(
    *,
    implementation: DenseProviderImplementation = DenseProviderImplementation.COLMAP_MVS,
    revision: int = 1,
    stage: ProgressiveDenseStage = ProgressiveDenseStage.PROVIDER_EXECUTION,
    points: int = 100,
) -> ProgressiveDenseSnapshot:
    return ProgressiveDenseSnapshot(
        source_sfm_session_id=SESSION_ID,
        dense_run_id=DENSE_RUN_ID,
        revision=revision,
        timestamp_utc="2026-09-18T09:00:00+00:00",
        elapsed_ns=10,
        provider_id=PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[implementation],
        implementation=implementation,
        stage=stage,
        stage_detail="qualified progressive dense contract probe",
        request_fingerprint_sha256=REQUEST_SHA,
        source_sparse_result_sha256=SPARSE_SHA,
        reconstruction_frame_id="pycolmap-reconstruction-frame",
        statistics=_statistics(points=points),
        progress_fraction=0.5,
        geometry=ProgressiveDenseGeometryPreview(point_samples=(_point(0),)),
    )


def test_valid_snapshot_round_trip_and_digest() -> None:
    snapshot = _snapshot()
    payload = snapshot.as_dict()
    restored = ProgressiveDenseSnapshot.from_payload(payload)
    assert restored.snapshot_sha256 == snapshot.snapshot_sha256
    assert restored.provider_id == "colmap-mvs-external"
    assert restored.authority_as_dict()["snapshot_authority"] == I67A_SNAPSHOT_AUTHORITY
    assert restored.authority_as_dict()["camera_pose_authority_owner"] == I67A_CAMERA_AUTHORITY_OWNER
    assert restored.authority_as_dict()["scale_status"] == I67A_SCALE_STATUS
    assert restored.provisional is True
    assert restored.final_dense_result_admitted is False
    assert restored.persistent_dense_scene is False
    assert restored.canonical_geometry_authoritative is False


@pytest.mark.parametrize("implementation", tuple(DenseProviderImplementation))
def test_every_canonical_concrete_provider_identity_is_supported(implementation: DenseProviderImplementation) -> None:
    snapshot = _snapshot(implementation=implementation)
    assert snapshot.provider_id == PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[implementation]
    assert snapshot.provider_family.value


def test_auto_or_mismatched_provider_identity_is_rejected() -> None:
    with pytest.raises(ReconstructionValidationError):
        replace(_snapshot(), provider_id="auto")
    with pytest.raises(ReconstructionValidationError):
        replace(_snapshot(), provider_id="depth-anything-3-external")


def test_point_sample_is_validated_and_copied_read_only() -> None:
    source = np.asarray([1.0, 2.0, 3.0])
    point = ProgressiveDensePointSample(sample_index=1, position_reconstruction=source)
    source[:] = 9.0
    assert point.position_reconstruction.tolist() == [1.0, 2.0, 3.0]
    assert point.position_reconstruction.flags.writeable is False

    with pytest.raises(ReconstructionValidationError):
        ProgressiveDensePointSample(sample_index=0, position_reconstruction=[1.0, float("nan"), 3.0])
    with pytest.raises(ReconstructionValidationError):
        ProgressiveDensePointSample(sample_index=0, position_reconstruction=[1, 2, 3], color_rgb=(0, 0, 256))
    with pytest.raises(ReconstructionValidationError):
        ProgressiveDensePointSample(sample_index=0, position_reconstruction=[1, 2, 3], confidence=1.1)
    with pytest.raises(ReconstructionValidationError):
        ProgressiveDensePointSample(sample_index=0, position_reconstruction=[1, 2, 3], view_support=0)


def test_preview_rejects_duplicate_sample_indices() -> None:
    with pytest.raises(ReconstructionValidationError):
        ProgressiveDenseGeometryPreview(point_samples=(_point(0), _point(0)))


def test_preview_cannot_exceed_current_dense_point_total() -> None:
    with pytest.raises(ReconstructionValidationError):
        replace(
            _snapshot(points=1),
            geometry=ProgressiveDenseGeometryPreview(point_samples=(_point(0), _point(1))),
        )


def test_statistics_validate_view_counts_and_optional_quality() -> None:
    with pytest.raises(ReconstructionValidationError):
        replace(_statistics(), selected_views=25)
    with pytest.raises(ReconstructionValidationError):
        replace(_statistics(), processed_views=19)
    with pytest.raises(ReconstructionValidationError):
        replace(_statistics(), mean_confidence=-0.1)
    with pytest.raises(ReconstructionValidationError):
        replace(_statistics(), mean_view_support=-1.0)


def test_provisional_snapshot_cannot_claim_final_persistence_metric_or_mesh_authority() -> None:
    baseline = _snapshot()
    forbidden = (
        {"provisional": False},
        {"final_dense_result_admitted": True},
        {"persistent_dense_scene": True},
        {"canonical_geometry_authoritative": True},
        {"camera_pose_authority_established": True},
        {"metric_scale_claimed": True},
        {"measurement_authoritative": True},
        {"certified_metrology": True},
        {"surface_mesh_generated": True},
        {"provider_promotion_performed": True},
    )
    for values in forbidden:
        with pytest.raises(ReconstructionValidationError):
            replace(baseline, **values)


def test_provisional_snapshot_cannot_claim_completed_progress() -> None:
    with pytest.raises(ReconstructionValidationError):
        replace(_snapshot(), progress_fraction=1.0)


def test_payload_authority_tampering_fails_closed() -> None:
    for field, value in (
        ("final_dense_result_admitted", True),
        ("persistent_dense_scene", True),
        ("canonical_geometry_authoritative", True),
        ("metric_scale_claimed", True),
        ("surface_mesh_generated", True),
    ):
        payload = _snapshot().as_dict()
        payload.pop("snapshot_sha256", None)
        payload["authority"][field] = value
        with pytest.raises(ReconstructionValidationError):
            ProgressiveDenseSnapshot.from_payload(payload)


def test_payload_authority_label_tampering_fails_closed() -> None:
    payload = _snapshot().as_dict()
    payload.pop("snapshot_sha256", None)
    payload["authority"]["camera_pose_authority_owner"] = "dense_provider"
    with pytest.raises(ReconstructionValidationError):
        ProgressiveDenseSnapshot.from_payload(payload)


def test_digest_tampering_is_detected() -> None:
    payload = _snapshot().as_dict()
    payload["stage_detail"] = "tampered"
    with pytest.raises(ReconstructionValidationError):
        ProgressiveDenseSnapshot.from_payload(payload)


def test_geometry_payload_rejects_nonobject_samples() -> None:
    payload = _snapshot().as_dict()
    payload.pop("snapshot_sha256", None)
    payload["geometry"]["point_samples"].append("bad")
    with pytest.raises(ReconstructionValidationError):
        ProgressiveDenseSnapshot.from_payload(payload)
