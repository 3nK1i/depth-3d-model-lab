from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderExecutionResult,
    DenseProviderImplementation,
    DenseProviderProgressEvent,
    DenseProviderProgressStage,
    DenseProviderRunStatus,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_dense_contract import ProgressiveDenseStage
from core.reconstruction.reconstruction_progressive_dense_emission import (
    I67B_MAX_PREVIEW_POINTS,
    I67B_MAX_PROVISIONAL_PROGRESS,
    ProgressiveDenseEmissionPolicy,
    ProgressiveDenseEmissionSession,
    canonical_offline_sfm_result_sha256,
    dense_request_fingerprint_sha256,
    evenly_sample_indices,
    load_dense_geometry_preview,
)
from tests.i4_3b_helpers import explicit_selection, make_sfm_result, write_minimal_ply


class CollectingSnapshotSink:
    def __init__(self, *, fail_revisions: set[int] | None = None) -> None:
        self.snapshots = []
        self.fail_revisions = set() if fail_revisions is None else set(fail_revisions)

    def publish(self, snapshot) -> None:
        if snapshot.revision in self.fail_revisions:
            raise RuntimeError("synthetic sink failure")
        self.snapshots.append(snapshot)


def _session(tmp_path: Path, *, sink: CollectingSnapshotSink | None = None):
    sfm = make_sfm_result(tmp_path, view_count=3)
    request, decision = explicit_selection(3)
    sink = sink or CollectingSnapshotSink()
    session = ProgressiveDenseEmissionSession(
        source_sfm_result=sfm,
        request=request,
        decision=decision,
        reconstruction_frame_id="sfm-frame-1",
        snapshot_sink=sink,
        dense_run_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        policy=ProgressiveDenseEmissionPolicy(max_preview_points=4, target_status_snapshots=8),
        utc_now=lambda: "2026-09-18T10:00:00+00:00",
        monotonic_ns=lambda: 1_000_000_000,
    )
    return session, sink, sfm, request, decision


def test_i67b_policy_is_bounded_and_preview_limit_is_visual_only() -> None:
    policy = ProgressiveDenseEmissionPolicy()
    assert policy.max_preview_points == I67B_MAX_PREVIEW_POINTS == 4096
    assert policy.target_status_snapshots == 24
    assert policy.max_failure_records == 16
    with pytest.raises(ReconstructionValidationError):
        ProgressiveDenseEmissionPolicy(max_preview_points=0)


def test_i67b_even_sampling_is_deterministic_bounded_and_keeps_endpoints() -> None:
    assert evenly_sample_indices(0, 5) == ()
    assert evenly_sample_indices(3, 5) == (0, 1, 2)
    sample = evenly_sample_indices(100, 8)
    assert len(sample) == 8
    assert sample[0] == 0 and sample[-1] == 99
    assert sample == evenly_sample_indices(100, 8)


def test_i67b_source_sparse_digest_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    sfm = make_sfm_result(tmp_path, view_count=3)
    first = canonical_offline_sfm_result_sha256(sfm)
    second = canonical_offline_sfm_result_sha256(sfm)
    changed = canonical_offline_sfm_result_sha256(replace(sfm, warnings=("new warning",)))
    assert len(first) == 64 and first == second and first != changed


def test_i67b_request_fingerprint_binds_selection_sparse_result_and_frame(tmp_path: Path) -> None:
    sfm = make_sfm_result(tmp_path, view_count=3)
    request, decision = explicit_selection(3)
    sparse_sha = canonical_offline_sfm_result_sha256(sfm)
    first = dense_request_fingerprint_sha256(request, decision, sparse_sha, "sfm-frame-1")
    assert first == dense_request_fingerprint_sha256(request, decision, sparse_sha, "sfm-frame-1")
    assert first != dense_request_fingerprint_sha256(request, decision, "f" * 64, "sfm-frame-1")


def test_i67b_load_npz_preview_is_bounded_rgb_confidence_and_support(tmp_path: Path) -> None:
    points = np.column_stack((np.arange(10), np.arange(10) + 1, np.arange(10) + 2)).astype(np.float32)
    colors = np.tile(np.asarray([[10, 20, 30]], dtype=np.uint8), (10, 1))
    confidence = np.linspace(0.1, 1.0, 10, dtype=np.float32)
    support = np.arange(1, 11, dtype=np.int32)
    path = tmp_path / "dense.npz"
    np.savez_compressed(
        path,
        points_reconstruction=points,
        colors_rgb=colors,
        mean_confidence=confidence,
        view_observation_counts=support,
    )
    loaded = load_dense_geometry_preview(path, max_preview_points=4)
    assert loaded.point_count == 10
    assert len(loaded.preview.point_samples) == 4
    assert loaded.preview.point_samples[0].sample_index == 0
    assert loaded.preview.point_samples[-1].sample_index == 9
    assert loaded.preview.point_samples[0].color_rgb == (10, 20, 30)
    assert loaded.preview.point_samples[0].view_support == 1
    assert loaded.mean_confidence == pytest.approx(float(confidence.mean()))
    assert loaded.mean_view_support == pytest.approx(float(support.mean()))


def test_i67b_load_ply_preview_reads_real_point_count_without_full_preview(tmp_path: Path) -> None:
    path = tmp_path / "fused.ply"
    write_minimal_ply(path, count=9)
    loaded = load_dense_geometry_preview(path, max_preview_points=4)
    assert loaded.point_count == 9
    assert len(loaded.preview.point_samples) == 4
    assert loaded.preview.point_samples[0].sample_index == 0
    assert loaded.preview.point_samples[-1].sample_index == 8
    assert all(item.color_rgb is not None for item in loaded.preview.point_samples)


def test_i67b_session_translates_provider_progress_without_authority_promotion(tmp_path: Path) -> None:
    session, sink, *_ = _session(tmp_path)
    session.publish(
        DenseProviderProgressEvent(
            provider_id="colmap-mvs-external",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
            stage_detail="patch_match_stereo",
            progress_fraction=0.4,
            processed_views=1,
            total_views=3,
        )
    )
    assert len(sink.snapshots) == 1
    snapshot = sink.snapshots[0]
    assert snapshot.stage is ProgressiveDenseStage.PROVIDER_EXECUTION
    assert snapshot.statistics.current_dense_points == 0
    assert snapshot.progress_fraction == pytest.approx(0.4)
    assert snapshot.provisional is True
    assert snapshot.final_dense_result_admitted is False
    assert snapshot.persistent_dense_scene is False
    assert snapshot.canonical_geometry_authoritative is False


def test_i67b_provider_progress_at_one_is_kept_provisional(tmp_path: Path) -> None:
    session, sink, *_ = _session(tmp_path)
    session.publish(
        DenseProviderProgressEvent(
            provider_id="colmap-mvs-external",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            stage=DenseProviderProgressStage.FINALIZATION,
            stage_detail="canonical_geometry_ready",
            progress_fraction=1.0,
            processed_views=3,
            total_views=3,
        )
    )
    assert sink.snapshots[0].progress_fraction == I67B_MAX_PROVISIONAL_PROGRESS


def test_i67b_status_throttle_bounds_same_stage_events_but_keeps_stage_changes(tmp_path: Path) -> None:
    session, sink, *_ = _session(tmp_path)
    for index in range(101):
        session.publish(
            DenseProviderProgressEvent(
                provider_id="colmap-mvs-external",
                implementation=DenseProviderImplementation.COLMAP_MVS,
                stage=DenseProviderProgressStage.ALIGNMENT,
                stage_detail="align",
                progress_fraction=0.40 + 0.35 * (index / 100.0),
                processed_views=min(3, index),
                total_views=3,
            )
        )
    before = len(sink.snapshots)
    session.publish(
        DenseProviderProgressEvent(
            provider_id="colmap-mvs-external",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            stage=DenseProviderProgressStage.FUSION_FILTERING,
            stage_detail="fusion",
            progress_fraction=0.8,
            processed_views=3,
            total_views=3,
        )
    )
    assert before <= 10
    assert len(sink.snapshots) == before + 1
    assert session.dropped_status_snapshot_count > 0



def test_i67b_emission_fails_closed_on_provider_stage_regression(tmp_path: Path) -> None:
    session, sink, *_ = _session(tmp_path)
    session.publish(
        DenseProviderProgressEvent(
            provider_id="colmap-mvs-external",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            stage=DenseProviderProgressStage.ALIGNMENT,
            stage_detail="alignment",
            progress_fraction=0.70,
            processed_views=2,
            total_views=3,
        )
    )
    with pytest.raises(ReconstructionValidationError, match="stage cannot regress"):
        session.publish(
            DenseProviderProgressEvent(
                provider_id="colmap-mvs-external",
                implementation=DenseProviderImplementation.COLMAP_MVS,
                stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
                stage_detail="invalid regression",
                progress_fraction=0.60,
                processed_views=2,
                total_views=3,
            )
        )
    assert [item.revision for item in sink.snapshots] == [1]


def test_i67b_snapshot_sink_failure_is_fail_soft_and_revision_gap_is_legal(tmp_path: Path) -> None:
    sink = CollectingSnapshotSink(fail_revisions={1})
    session, _, *_ = _session(tmp_path, sink=sink)
    for stage, progress in (
        (DenseProviderProgressStage.PREPARING, 0.02),
        (DenseProviderProgressStage.VIEW_SELECTION, 0.05),
    ):
        session.publish(
            DenseProviderProgressEvent(
                provider_id="colmap-mvs-external",
                implementation=DenseProviderImplementation.COLMAP_MVS,
                stage=stage,
                stage_detail=stage.value,
                progress_fraction=progress,
                processed_views=0,
                total_views=3,
            )
        )
    assert [item.revision for item in sink.snapshots] == [2]
    assert session.emission_failure_count == 1
    assert session.failure_records


def test_i67b_emit_final_geometry_produces_bounded_actual_dense_preview(tmp_path: Path) -> None:
    session, sink, *_ = _session(tmp_path)
    path = tmp_path / "dense.npz"
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    colors = np.tile(np.asarray([[100, 110, 120]], dtype=np.uint8), (10, 1))
    np.savez_compressed(path, points_reconstruction=points, colors_rgb=colors)
    result = DenseProviderExecutionResult(
        provider_id="colmap-mvs-external",
        implementation=DenseProviderImplementation.COLMAP_MVS,
        status=DenseProviderRunStatus.COMPLETED,
        source_view_count=3,
        selected_view_count=3,
        adaptive_view_selection=False,
        view_selection_reason="all",
        permanent_global_view_cap=None,
        canonical_geometry_ref=str(path),
        canonical_frame_id="sfm-frame-1",
        progress_event_count=1,
    )
    revision = session.emit_final_geometry(result)
    assert revision == 1
    snapshot = sink.snapshots[-1]
    assert snapshot.statistics.current_dense_points == 10
    assert len(snapshot.geometry.point_samples) == 4
    assert snapshot.stage is ProgressiveDenseStage.FINALIZATION
    assert snapshot.final_dense_result_admitted is False


def test_i67b_failed_result_never_emits_geometry(tmp_path: Path) -> None:
    session, sink, *_ = _session(tmp_path)
    result = DenseProviderExecutionResult(
        provider_id="colmap-mvs-external",
        implementation=DenseProviderImplementation.COLMAP_MVS,
        status=DenseProviderRunStatus.FAILED,
        source_view_count=3,
        selected_view_count=3,
        adaptive_view_selection=False,
        view_selection_reason="all",
        permanent_global_view_cap=None,
        canonical_geometry_ref="",
        canonical_frame_id="",
        failure_reason="synthetic failure",
        progress_event_count=1,
        canonical_geometry_authoritative=False,
    )
    assert session.emit_final_geometry(result) == 0
    assert sink.snapshots == []


@pytest.mark.parametrize("implementation", list(DenseProviderImplementation))
def test_i67b_emission_session_supports_every_concrete_provider_identity(tmp_path: Path, implementation) -> None:
    from core.reconstruction.dense_provider_architecture import (
        DenseProviderSelectionDecision,
        DenseProviderSelectionMode,
        DenseProviderSelectionRequest,
    )
    from core.reconstruction.reconstruction_provider_selection_contract import (
        PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION,
    )

    sfm = make_sfm_result(tmp_path, view_count=3)
    provider_id = PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[implementation]
    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=3,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id="sfm-frame-1",
        requested_provider_id=provider_id,
        allow_fallback=False,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id=provider_id,
        selected_provider_id=provider_id,
        selected_implementation=implementation,
        considered_provider_ids=(provider_id,),
        automatic=False,
        engineer_override_applied=False,
        fallback_used=False,
        reason_codes=("explicit_engineer_selection",),
    )
    sink = CollectingSnapshotSink()
    session = ProgressiveDenseEmissionSession(
        source_sfm_result=sfm,
        request=request,
        decision=decision,
        reconstruction_frame_id="sfm-frame-1",
        snapshot_sink=sink,
        dense_run_id="cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa",
    )
    session.publish(
        DenseProviderProgressEvent(
            provider_id=provider_id,
            implementation=implementation,
            stage=DenseProviderProgressStage.PREPARING,
            stage_detail="provider_preparing",
            progress_fraction=0.02,
            processed_views=0,
            total_views=3,
        )
    )
    assert len(sink.snapshots) == 1
    assert sink.snapshots[0].implementation is implementation
    assert sink.snapshots[0].provider_id == provider_id


def test_i67b_colmap_worker_uses_multithreaded_non_cached_fusion_execution_policy() -> None:
    text = Path("scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py").read_text(encoding="utf-8")
    assert "fusion_options.use_cache = False" in text
    assert "fusion_options.num_threads = -1" in text
    assert "fusion_options.use_cache = True" not in text


def test_i67b_colmap_worker_records_fusion_execution_policy_without_changing_scientific_thresholds() -> None:
    text = Path("scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py").read_text(encoding="utf-8")
    assert 'result["stereo_fusion_execution_policy"]' in text
    assert '"use_cache": False' in text
    assert '"num_threads": -1' in text
    assert '"scientific_fusion_parameters_changed": False' in text
    assert "fusion_options.max_image_size = max_image_size" in text
    for scientific_field in (
        "min_num_pixels",
        "max_num_pixels",
        "max_traversal_depth",
        "max_reproj_error",
        "max_depth_error",
        "max_normal_error",
        "check_num_images",
    ):
        assert f"fusion_options.{scientific_field} =" not in text


def test_i67b_real_smoke_reports_provider_failure_before_geometry_absence() -> None:
    text = Path("scripts/run_i6_7b_progressive_dense_emission_smoke.py").read_text(encoding="utf-8")
    status_guard = text.index(
        "if envelope.result.status not in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}:"
    )
    geometry_guard = text.index('if not geometry:')
    assert status_guard < geometry_guard
    assert "failure_reason" in text[status_guard:geometry_guard]
