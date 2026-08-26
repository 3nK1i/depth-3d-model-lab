from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run_c_single_frame_canonical_projection import (
    _alignment_evidence_from_payload,
    _write_binary_ply,
)


def _accepted_payload() -> dict:
    return {
        "schema_version": 1,
        "session_id": "00000000-0000-4000-8000-00000000c101",
        "keyframe_id": "kf-c",
        "frame_id": "frame-c",
        "camera_id": "camera-1",
        "camera_model_name": "SIMPLE_PINHOLE",
        "depth_provenance": "unit-test depth",
        "status": "accepted",
        "sfm_scale_status": "arbitrary",
        "sfm_world_alignment_status": "arbitrary",
        "depth_image_size": {"width": 4, "height": 3},
        "minimum_correspondences": 8,
        "maximum_relative_residual": 0.35,
        "candidate_observations": 8,
        "behind_camera_observations": 0,
        "out_of_bounds_observations": 0,
        "invalid_depth_observations": 0,
        "valid_correspondences": 8,
        "inlier_correspondences": 8,
        "robust_relative_residual_limit": 0.1,
        "median_relative_residual": 0.02,
        "p90_relative_residual": 0.05,
        "alignment": {
            "mode": "sparse_correspondence_scale",
            "scale_to_reconstruction_units": 6.78,
            "relative_scale_uncertainty": 0.08,
            "scale_reference_profile_id": "",
            "provenance": "B2 accepted fit",
        },
        "rejection_code": None,
        "message": "accepted",
    }


def test_c_runner_restores_validated_b2_alignment_evidence() -> None:
    evidence = _alignment_evidence_from_payload(_accepted_payload())
    assert evidence.is_accepted
    assert evidence.alignment is not None
    assert evidence.alignment.scale_to_reconstruction_units == 6.78
    assert evidence.as_dict()["metric_scale_claimed"] is False


def test_c_runner_binary_ply_has_expected_vertex_count(tmp_path: Path) -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=np.float64)
    colors = np.asarray([[255, 0, 1], [4, 5, 6]], dtype=np.uint8)
    path = _write_binary_ply(tmp_path / "cloud.ply", points, colors)
    payload = path.read_bytes()
    assert b"format binary_little_endian 1.0" in payload
    assert b"element vertex 2" in payload
    assert len(payload) > 100
