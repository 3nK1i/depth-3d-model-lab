from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.reconstruction import (
    SfmWorkspaceError,
    materialize_sfm_workspace,
    validate_sfm_workspace_integrity,
)
from tests.reconstruction_sfm_helpers import make_sfm_ingestion_result


def test_workspace_materializes_every_selected_keyframe_not_just_cache(tmp_path) -> None:
    result = make_sfm_ingestion_result(tmp_path / "sequence", count=6)
    assert len(result.keyframes) == 6
    assert len(result.keyframe_images) == 2  # intentionally bounded Phase 8.2 cache

    workspace = materialize_sfm_workspace(result, tmp_path / "workspace")
    assert len(workspace.images) == 6
    for item in workspace.images:
        staged = Path(workspace.workspace_root) / item.staged_image_path
        assert staged.is_file()
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == item.staged_sha256

    payload = json.loads((Path(workspace.workspace_root) / "manifest.json").read_text())
    assert payload["schema_version"] == 1
    assert len(payload["images"]) == 6


def test_workspace_refuses_nonempty_destination_without_overwrite(tmp_path) -> None:
    result = make_sfm_ingestion_result(tmp_path / "sequence", count=3)
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "keep.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(SfmWorkspaceError):
        materialize_sfm_workspace(result, root)
    assert (root / "keep.txt").read_text(encoding="utf-8") == "do not delete"


def test_workspace_overwrite_is_explicit(tmp_path) -> None:
    result = make_sfm_ingestion_result(tmp_path / "sequence", count=3)
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    workspace = materialize_sfm_workspace(result, root, overwrite=True)
    assert not (root / "old.txt").exists()
    assert len(workspace.images) == 3


def test_recorded_video_workspace_redecodes_selected_frames(tmp_path) -> None:
    import cv2

    from core.reconstruction import (
        FrameQualityPolicy,
        FrameSamplingMode,
        FrameSamplingPolicy,
        IngestionCachePolicy,
        KeyframeSelectionPolicy,
        RecordedVideoReader,
        ReconstructionIngestionPipeline,
    )
    from tests.reconstruction_ingestion_helpers import textured_frame
    from tests.reconstruction_sfm_helpers import SFM_SESSION_ID

    video_path = tmp_path / "sequence.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable in this environment.")
    for index in range(5):
        writer.write(textured_frame(index, width=160, height=120))
    writer.release()

    source = RecordedVideoReader(video_path)
    result = ReconstructionIngestionPipeline(
        session_id=SFM_SESSION_ID,
        source=source,
        sampling_policy=FrameSamplingPolicy(mode=FrameSamplingMode.EVERY_FRAME),
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=1,
            forced_spacing_ns=1,
            minimum_visual_change=0.0,
            minimum_quality_score=0.0,
            maximum_keyframes=5,
        ),
        quality_policy=FrameQualityPolicy(
            minimum_blur_variance=0.0,
            minimum_contrast_stddev=0.0,
            minimum_mean_luma=0.0,
            maximum_mean_luma=255.0,
            maximum_clipped_fraction=1.0,
            minimum_quality_score=0.0,
        ),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=32 * 1024 * 1024,
            maximum_frames=1,
            gpu_staging_bytes=0,
        ),
    ).run()
    assert len(result.keyframes) == 5
    assert len(result.keyframe_images) == 1

    workspace = materialize_sfm_workspace(result, tmp_path / "video-workspace")
    assert len(workspace.images) == 5
    assert all(item.staged_image_path.endswith(".png") for item in workspace.images)
    assert all((Path(workspace.workspace_root) / item.staged_image_path).is_file() for item in workspace.images)


def test_workspace_integrity_detects_staged_image_mutation(tmp_path) -> None:
    result = make_sfm_ingestion_result(tmp_path / "sequence-integrity", count=3)
    workspace = materialize_sfm_workspace(result, tmp_path / "workspace-integrity")
    validate_sfm_workspace_integrity(workspace)

    staged = Path(workspace.workspace_root) / workspace.images[0].staged_image_path
    staged.write_bytes(staged.read_bytes() + b"mutated")
    with pytest.raises(SfmWorkspaceError, match="checksum changed"):
        validate_sfm_workspace_integrity(workspace)


def test_workspace_integrity_detects_missing_staged_image(tmp_path) -> None:
    result = make_sfm_ingestion_result(tmp_path / "sequence-missing", count=3)
    workspace = materialize_sfm_workspace(result, tmp_path / "workspace-missing")
    staged = Path(workspace.workspace_root) / workspace.images[1].staged_image_path
    staged.unlink()
    with pytest.raises(SfmWorkspaceError, match="is missing"):
        validate_sfm_workspace_integrity(workspace)
