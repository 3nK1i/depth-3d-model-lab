from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError

from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict
from core.reconstruction.single_image_provider_production_qualification import (
    I43G4B_PROVIDER_IMPLEMENTATIONS,
    validate_single_image_provider_run_evidence,
)
from tests.i4_3b_helpers import make_sfm_result
from tests.i4_3g4b_helpers import passing_g4a_snapshot, provider_evidence


def test_offline_sfm_json_roundtrip_preserves_frozen_authority(tmp_path: Path) -> None:
    original = make_sfm_result(tmp_path, view_count=4)
    serialized = json.loads(json.dumps(original.as_dict()))
    restored = offline_sfm_result_from_dict(serialized)
    assert restored.as_dict() == original.as_dict()
    assert restored.backend_id == original.backend_id
    assert [camera.T_reconstruction_from_camera.tolist() for camera in restored.cameras] == [
        camera.T_reconstruction_from_camera.tolist() for camera in original.cameras
    ]


def test_frozen_sfm_accepts_materialized_inputs_beyond_registered_cameras_and_projects_registered_subset(tmp_path: Path) -> None:
    from scripts.run_i4_3g4b_single_image_scenario import (
        _registered_view_projection,
        _verify_frozen_sfm,
    )
    from core.reconstruction.single_image_provider_production_qualification import sha256_file

    complete = make_sfm_result(tmp_path, view_count=4)
    partial = replace(
        complete,
        cameras=complete.cameras[:3],
        statistics=replace(complete.statistics, registered_keyframes=3),
    )
    snapshot = tmp_path / "frozen_sfm.json"
    snapshot.write_text(json.dumps(partial.as_dict(), indent=2) + "\n", encoding="utf-8")
    frozen_sha = sha256_file(snapshot)
    binding = SimpleNamespace(
        scenario_id="g4_single_image_complete_room",
        authoritative_sfm_snapshot_path=snapshot,
        authoritative_sfm_snapshot_sha256=frozen_sha,
        registered_view_count=3,
        landmark_count=0,
    )

    restored = _verify_frozen_sfm(binding)
    assert restored.statistics.input_keyframes == 4
    assert restored.statistics.registered_keyframes == 3
    assert len(restored.workspace.images) == 4

    projected = _registered_view_projection(restored)
    assert projected.statistics.input_keyframes == 3
    assert projected.statistics.registered_keyframes == 3
    assert len(projected.workspace.images) == 3
    assert [item.keyframe_id for item in projected.workspace.images] == [
        camera.keyframe_id for camera in projected.cameras
    ]
    assert [camera.T_reconstruction_from_camera.tolist() for camera in projected.cameras] == [
        camera.T_reconstruction_from_camera.tolist() for camera in restored.cameras
    ]
    assert sha256_file(snapshot) == frozen_sha


def test_registered_view_projection_is_identity_when_every_input_registered(tmp_path: Path) -> None:
    from scripts.run_i4_3g4b_single_image_scenario import _registered_view_projection

    complete = make_sfm_result(tmp_path, view_count=4)
    assert _registered_view_projection(complete) is complete


def test_all_six_provider_evidence_share_one_authority_policy() -> None:
    for implementation in I43G4B_PROVIDER_IMPLEMENTATIONS:
        evidence = provider_evidence(implementation)
        validate_single_image_provider_run_evidence(evidence)
        assert evidence["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
        assert evidence["authority"]["permanent_global_view_cap"] is None
        assert evidence["authority"]["provider_promoted"] is False
        assert evidence["authority"]["metric_scale_claimed"] is False


def test_runner_storage_floor_is_reused_and_enforced(tmp_path: Path, monkeypatch) -> None:
    from scripts.run_i4_3g4b_single_image_scenario import _require_storage_ready, _storage_state

    snapshot = passing_g4a_snapshot()
    snapshot["storage"]["storage_path"] = str(tmp_path)

    class Usage:
        total = 100 * 1024**3
        used = 80 * 1024**3
        free = 20 * 1024**3

    monkeypatch.setattr("scripts.run_i4_3g4b_single_image_scenario.shutil.disk_usage", lambda _path: Usage())
    state = _storage_state(snapshot)
    assert state["storage_ready"] is False
    assert state["threshold_reused_from_g4a"] is True
    with pytest.raises(ReconstructionValidationError, match="storage safety floor failed"):
        _require_storage_ready(snapshot, stage="test")


@pytest.mark.parametrize(
    ("implementation", "manifest_name"),
    (
        ("metric3d_v2", "single_image_depth_result_manifest.json"),
        ("unidepth_v2", "unidepth_v2_result_manifest.json"),
        ("moge_2", "moge2_result_manifest.json"),
        ("depth_pro", "depth_pro_result_manifest.json"),
        ("depth_anything_v2", "depth_anything_v2_result_manifest.json"),
        ("marigold", "marigold_result_manifest.json"),
    ),
)
def test_runner_worker_artifacts_preserve_each_provider_native_outputs(
    tmp_path: Path, implementation: str, manifest_name: str
) -> None:
    from scripts.run_i4_3g4b_single_image_scenario import _worker_artifacts

    provider_out = tmp_path / implementation
    worker = provider_out / "provider" / "worker_output"
    worker.mkdir(parents=True)
    input_manifest = provider_out / "provider" / "input_manifest.json"
    input_manifest.write_text("{}\n", encoding="utf-8")
    native_a = worker / "view-a.npz"
    native_b = worker / "view-b.npz"
    native_a.write_bytes(b"a")
    native_b.write_bytes(b"b")
    (worker / manifest_name).write_text(
        json.dumps({"frame_records": [
            {"evidence_npz_path": str(native_a)},
            {"evidence_npz_path": str(native_b)},
        ]}),
        encoding="utf-8",
    )

    artifacts, count = _worker_artifacts(provider_out, implementation=implementation)
    assert count == 2
    resolved = [path.resolve() for path in artifacts]
    assert input_manifest.resolve() in resolved
    assert native_a.resolve() in resolved
    assert native_b.resolve() in resolved


def test_runner_alignment_metrics_use_only_persisted_semantics() -> None:
    from scripts.run_i4_3g4b_single_image_scenario import _alignment_metrics

    summary = {
        "alignment": {
            "accepted_views": 2,
            "records": [
                {
                    "status": "accepted",
                    "candidate_observations": 100,
                    "valid_correspondences": 80,
                    "median_relative_residual": 0.10,
                },
                {
                    "status": "accepted",
                    "candidate_observations": 100,
                    "valid_correspondences": 60,
                    "median_relative_residual": 0.20,
                },
                {
                    "status": "rejected",
                    "candidate_observations": 100,
                    "valid_correspondences": 10,
                    "median_relative_residual": 0.90,
                },
            ],
        }
    }
    accepted, metrics, reasons = _alignment_metrics(summary, selected=3)
    assert accepted == 2
    assert metrics["registered_view_surface_coverage"] == pytest.approx(2 / 3)
    assert metrics["sparse_correspondence_coverage"] == pytest.approx(150 / 300)
    assert metrics["alignment_median_relative_residual"] == pytest.approx(0.15)
    assert "alignment_p95_relative_residual" not in metrics
    assert "p95" in reasons["alignment_p95_relative_residual"]


def test_runner_alignment_metrics_support_e5_e6_input_correspondence_schema() -> None:
    from scripts.run_i4_3g4b_single_image_scenario import _alignment_metrics

    summary = {
        "alignment": {
            "accepted_views": 2,
            "records": [
                {"status": "accepted", "input_correspondences": 50, "valid_correspondences": 40, "median_relative_residual": 0.1},
                {"status": "accepted", "input_correspondences": 50, "valid_correspondences": 30, "median_relative_residual": 0.2},
            ],
        }
    }
    accepted, metrics, reasons = _alignment_metrics(summary, selected=2)
    assert accepted == 2
    assert metrics["sparse_correspondence_coverage"] == pytest.approx(0.7)
    assert metrics["alignment_median_relative_residual"] == pytest.approx(0.15)
    assert "sparse_correspondence_coverage" not in reasons
