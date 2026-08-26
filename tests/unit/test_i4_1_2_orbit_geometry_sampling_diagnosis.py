from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import (
    build_i4_1_2_diagnosis,
    coverage_by_plane_mode,
    optical_axis_consensus_target,
    sampling_coverage_diagnosis,
    trajectory_circle_target,
)


def _pose(angle_deg: float, *, radius: float = 5.0, z: float = 0.4, target=None) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    center = np.asarray([radius * np.cos(a), radius * np.sin(a), z], dtype=float)
    target = np.zeros(3) if target is None else np.asarray(target, dtype=float)
    forward = target - center
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0])
    x = np.cross(up, forward)
    if np.linalg.norm(x) < 1e-8:
        up = np.asarray([0.0, 1.0, 0.0])
        x = np.cross(up, forward)
    x /= np.linalg.norm(x)
    y = np.cross(forward, x); y /= np.linalg.norm(y)
    m = np.eye(4); m[:3, :3] = np.column_stack([x, y, forward]); m[:3, 3] = center
    return m


def _rotate(matrices: np.ndarray) -> np.ndarray:
    ax, ay = np.deg2rad(37.0), np.deg2rad(-21.0)
    rx = np.asarray([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]],dtype=float)
    ry = np.asarray([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]],dtype=float)
    r = ry @ rx
    T = np.eye(4); T[:3,:3] = r; T[:3,3] = np.asarray([9.0,-4.0,3.0])
    return np.asarray([T @ m for m in matrices])


def _sfm_report(matrices: np.ndarray, *, frame_count: int = 1520, fps: float = 29.97, sample_hz: float = 2.0):
    images=[]; cameras=[]
    for i,m in enumerate(matrices):
        seq=int(round(i*fps/sample_hz)); ts=int(round(i/sample_hz*1e9)); kid=f"kf-{i}"
        images.append({"order":i,"keyframe_id":kid,"frame_id":f"f-{i}","source_sequence_index":seq,"timestamp_ns":ts})
        cameras.append({"keyframe_id":kid,"T_reconstruction_from_camera":m.tolist()})
    landmarks=[]
    for i,(x,y,z) in enumerate((
        (x,y,z) for x in np.linspace(-1,1,6) for y in np.linspace(-.7,.7,6) for z in np.linspace(-.5,.5,4)
    )):
        landmarks.append({"landmark_id":str(i),"position_reconstruction":[x,y,z]})
    return {"source":{"expected_frame_rate_hz":fps,"metadata":{"reported_frame_count":frame_count}},"sfm_result":{"workspace":{"images":images},"cameras":cameras,"landmarks":landmarks}}


def _base(angle_end: float = 220.0):
    angles=np.linspace(0.0,angle_end,40)
    poses=np.asarray([_pose(a) for a in angles])
    report=_sfm_report(poses)
    ids=tuple(f"kf-{i}" for i in range(40))
    sparse=np.asarray([item["position_reconstruction"] for item in report["sfm_result"]["landmarks"]],dtype=float)
    return poses,report,ids,sparse


def test_plane_modes_are_coordinate_invariant_for_rotated_broad_orbit():
    poses,_,_,_=_base(220.0)
    target=np.zeros(3)
    before=coverage_by_plane_mode(poses,target)
    rotated=_rotate(poses); target2=np.asarray([9.0,-4.0,3.0])
    after=coverage_by_plane_mode(rotated,target2)
    for name in before:
        assert before[name]["azimuth_coverage_degrees"] == pytest.approx(after[name]["azimuth_coverage_degrees"], abs=1e-7)
        assert before[name]["azimuth_coverage_degrees"] > 200.0


def test_optical_axis_consensus_recovers_known_target():
    poses,_,_,_=_base(210.0)
    target,diag=optical_axis_consensus_target(poses)
    assert np.linalg.norm(target) < 1e-7
    assert diag["forward_fraction"] == pytest.approx(1.0)
    assert diag["angular_residual_degrees"]["p90"] < 1e-5


def test_trajectory_circle_recovers_orbit_center_under_arbitrary_transform():
    poses,_,_,_=_base(240.0)
    rotated=_rotate(poses)
    target,diag=trajectory_circle_target(rotated)
    # The circle center lies in the camera orbit plane (z=0.4 before the
    # arbitrary rigid transform); it is a trajectory-center diagnostic, not
    # an object-depth estimator along the plane normal.
    base_center=np.asarray([0.0,0.0,0.4,1.0])
    ax,ay=np.deg2rad(37.0),np.deg2rad(-21.0)
    rx=np.asarray([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]],dtype=float)
    ry=np.asarray([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]],dtype=float)
    T=np.eye(4); T[:3,:3]=ry@rx; T[:3,3]=np.asarray([9.0,-4.0,3.0])
    expected=(T@base_center)[:3]
    assert np.linalg.norm(target-expected) < 1e-5
    assert diag["radial_residual_fraction"]["p90"] < 1e-5


def test_sampling_diagnosis_detects_40_frame_prefix_cap_on_50s_video():
    poses,report,_,_=_base(220.0)
    result=sampling_coverage_diagnosis(report,expected_max_sampled_frames=40)
    assert result["sampled_view_count"] == 40
    assert result["prefix_max_frame_truncation_likely"] is True
    assert 19.4 < result["sampled_timestamp_span_seconds"] < 19.6
    assert result["sampled_temporal_span_fraction"] < 0.40
    assert result["sampled_last_source_index_fraction"] < 0.40


def test_sampling_diagnosis_does_not_flag_short_fully_represented_video():
    poses=np.asarray([_pose(a) for a in np.linspace(0,180,30)])
    report=_sfm_report(poses,frame_count=450,fps=30.0,sample_hz=2.0)
    # 30 samples span 14.5s while source ~15s and cap is not reached.
    result=sampling_coverage_diagnosis(report,expected_max_sampled_frames=40)
    assert result["cap_reached"] is False
    assert result["prefix_max_frame_truncation_likely"] is False


def test_build_diagnosis_prefers_sampling_correction_candidate_for_prefix_limited_source():
    poses,report,ids,sparse=_base(220.0)
    result=build_i4_1_2_diagnosis(
        source_i4_summary={"phase":"I4.1","status":"degraded","failed_stage":"object_orbit_i1_baseline","reason":"I1 baseline did not qualify"},
        sfm_report=report,
        registered_camera_ids=ids,
        registered_camera_matrices=poses,
        sparse_landmarks=sparse,
        aligned_keyframe_ids=ids[:30],
        selected_keyframe_ids=ids[34:37],
    )
    assert result["status"] == "diagnosis_ready"
    assert result["diagnosis"]["sampling_policy_correction_candidate"] is True
    assert result["authority"]["sampling_policy_changed"] is False
    assert result["decision"]["authorizes_sampling_policy_change"] is False
    assert result["decision"]["authorizes_i4_2_detail_preservation"] is False


def test_build_diagnosis_reproduces_current_core_metric_when_core_available():
    poses,report,ids,sparse=_base(200.0)
    core=sparse.copy()
    result=build_i4_1_2_diagnosis(
        source_i4_summary={"phase":"I4.1","status":"degraded","failed_stage":"object_orbit_baseline_quality"},
        sfm_report=report,registered_camera_ids=ids,registered_camera_matrices=poses,sparse_landmarks=sparse,
        aligned_keyframe_ids=ids[:35],selected_keyframe_ids=ids[::3][:12],core_points=core,
    )
    assert result["current_i4_metric_reproduction_exact"] is True


def test_wrong_source_phase_fails_closed():
    poses,report,ids,sparse=_base()
    with pytest.raises(ReconstructionValidationError):
        build_i4_1_2_diagnosis(
            source_i4_summary={"phase":"I3","status":"degraded","failed_stage":"object_orbit_i1_baseline"},
            sfm_report=report,registered_camera_ids=ids,registered_camera_matrices=poses,sparse_landmarks=sparse,
        )


def test_duplicate_registered_ids_fail_closed():
    poses,report,ids,sparse=_base()
    broken=list(ids); broken[-1]=broken[0]
    with pytest.raises(ReconstructionValidationError):
        build_i4_1_2_diagnosis(
            source_i4_summary={"phase":"I4.1","status":"degraded","failed_stage":"object_orbit_i1_baseline"},
            sfm_report=report,registered_camera_ids=broken,registered_camera_matrices=poses,sparse_landmarks=sparse,
        )


def test_no_authority_promotion_on_diagnosis():
    poses,report,ids,sparse=_base()
    result=build_i4_1_2_diagnosis(
        source_i4_summary={"phase":"I4.1","status":"degraded","failed_stage":"object_orbit_i1_baseline"},
        sfm_report=report,registered_camera_ids=ids,registered_camera_matrices=poses,sparse_landmarks=sparse,
    )
    assert result["authority"]["metric_scale_claimed"] is False
    assert result["authority"]["measurement_authoritative"] is False
    assert result["authority"]["surface_mesh_generated"] is False
    assert result["decision"]["authorizes_threshold_relaxation"] is False
