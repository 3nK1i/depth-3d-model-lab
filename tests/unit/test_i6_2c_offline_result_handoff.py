from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.reconstruction_offline_result_handoff import (
    OfflineResultHandoffAvailability,
    build_offline_result_handoff,
)


def _payload(root: Path, session_id: str, *, with_visual: bool = True) -> dict[str, object]:
    artifacts: dict[str, str] = {}
    if with_visual:
        visual = root / "offline_visualization.json"
        visual.write_text("{}", encoding="utf-8")
        artifacts["offline_visualization_json"] = str(visual)
    sparse = root / "sparse_reconstruction.ply"
    sparse.write_text("ply\n", encoding="utf-8")
    artifacts["sparse_reconstruction_ply"] = str(sparse)
    result = {
        "status": "completed",
        "pass": True,
        "session_id": session_id,
        "camera_pose_available": True,
        "canonical_geometry_authoritative": False,
        "persistent_scene_ready": False,
        "certified_metrology": False,
        "artifacts": artifacts,
        "sfm_summary": {"gauge": {"scale_status": "arbitrary"}},
    }
    (root / "offline_result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def test_sparse_completed_result_becomes_partial_handoff(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    sid = str(uuid4())
    payload = _payload(root, sid)

    handoff = build_offline_result_handoff(
        payload,
        root,
        expected_session_id=sid,
        expected_output_dir=root,
    )

    assert handoff.availability is OfflineResultHandoffAvailability.PARTIAL
    assert handoff.session_id == sid
    assert handoff.output_dir == str(root)
    assert handoff.viewer_ready is True
    assert handoff.sparse_export_ready is True
    assert handoff.camera_pose_available is True
    assert handoff.canonical_geometry_authoritative is False
    assert handoff.certified_metrology is False
    assert handoff.scale_status == "arbitrary"
    assert handoff.warnings == ()


def test_missing_visualization_is_partial_but_not_viewer_ready(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    sid = str(uuid4())
    payload = _payload(root, sid, with_visual=False)

    handoff = build_offline_result_handoff(
        payload,
        root,
        expected_session_id=sid,
        expected_output_dir=root,
    )

    assert handoff.availability is OfflineResultHandoffAvailability.PARTIAL
    assert handoff.viewer_ready is False
    assert handoff.sparse_export_ready is True
    assert "visualization" in " ".join(handoff.warnings).lower()


def test_session_and_output_identity_must_match(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    sid = str(uuid4())
    payload = _payload(root, sid)

    with pytest.raises(ValueError, match="session identity"):
        build_offline_result_handoff(
            payload,
            root,
            expected_session_id=str(uuid4()),
            expected_output_dir=root,
        )

    with pytest.raises(ValueError, match="output directory"):
        build_offline_result_handoff(
            payload,
            root,
            expected_session_id=sid,
            expected_output_dir=tmp_path / "other",
        )


def test_artifact_escape_is_quarantined_without_downgrading_result(tmp_path: Path) -> None:
    root = (tmp_path / "run").resolve()
    root.mkdir()
    sid = str(uuid4())
    payload = _payload(root, sid)
    payload["artifacts"] = {
        "offline_visualization_json": str((tmp_path / "outside.json").resolve())
    }

    handoff = build_offline_result_handoff(
        payload,
        root,
        expected_session_id=sid,
        expected_output_dir=root,
    )

    assert handoff.availability is OfflineResultHandoffAvailability.PARTIAL
    assert handoff.viewer_ready is False
    assert handoff.sparse_export_ready is False
    assert "outside the admitted output directory" in " ".join(handoff.warnings)


def test_metrology_and_invalid_terminal_claims_fail_closed(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    sid = str(uuid4())
    payload = _payload(root, sid)
    payload["certified_metrology"] = True
    with pytest.raises(ValueError, match="certified metrology"):
        build_offline_result_handoff(payload, root, expected_session_id=sid, expected_output_dir=root)

    payload = _payload(root, sid)
    payload["pass"] = False
    with pytest.raises(ValueError, match="completed passing"):
        build_offline_result_handoff(payload, root, expected_session_id=sid, expected_output_dir=root)
