from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest

from tests.phase9_3h_helpers import make_phase9_3h_da3_fixture


def test_e2_dependency_light_worker_reuses_da3_shaped_evidence_and_persists_scene(tmp_path: Path) -> None:
    input_path, _result_path, manifest = make_phase9_3h_da3_fixture(tmp_path / "fixture")
    input_payload = json.loads(input_path.read_text(encoding="utf-8"))
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    for index, source_text in enumerate(input_payload["image_paths"]):
        source = Path(source_text)
        (keyframes / f"{index:04d}{source.suffix}").write_bytes(source.read_bytes())
    chunk_path = tmp_path / "chunk-0000.npz"
    source_arrays = np.load(manifest.output_npz_path, allow_pickle=False)
    np.savez_compressed(
        chunk_path,
        depth=source_arrays["depth"],
        confidence=source_arrays["confidence"],
        extrinsics_w2c=source_arrays["extrinsics_w2c"],
        intrinsics=source_arrays["intrinsics"],
    )
    external = {
        "schema_version": 1,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "source_revision": manifest.source_revision,
        "gpu_name": manifest.gpu_name,
        "peak_vram_mib": manifest.peak_vram_mib,
        "inference_duration_s": manifest.inference_duration_s,
        "process_res": manifest.process_res,
        "chunks": [{
            "ordinal": 0,
            "frame_indices": [0, 1, 2],
            "npz_path": str(chunk_path.resolve()),
            "frame_count": 3,
            "height": manifest.height,
            "width": manifest.width,
            "inference_duration_s": manifest.inference_duration_s,
        }],
        "geometry_scale": "relative_arbitrary_scale",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    external_path = tmp_path / "external.json"
    external_path.write_text(json.dumps(external), encoding="utf-8")
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"reuse-hook-does-not-decode-video")
    output = tmp_path / "output"
    job_id = str(uuid4())
    script = Path(__file__).resolve().parents[2] / "scripts/run_phase9_6e2_full_scene_da3_job.py"
    completed = subprocess.run(
        [
            sys.executable, str(script),
            "--job-id", job_id,
            "--video", str(video),
            "--output-dir", str(output),
            "--da3-python", sys.executable,
            "--da3-repo", str(tmp_path),
            "--reuse-keyframes-dir", str(keyframes),
            "--reuse-external-summary", str(external_path),
            "--pixel-stride", "2",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    summary = json.loads((output / "phase9_6e2_full_scene_summary.json").read_text(encoding="utf-8"))
    assert summary["pass"] is True
    assert summary["job_id"] == job_id
    assert summary["selected_keyframes"] == 3
    assert summary["geometry_scale"] == "relative_arbitrary_scale"
    assert summary["unit_to_metre"] is None
    assert summary["loop_closure"] is False
    assert summary["quality_certified"] is False
    assert summary["certified_metrology"] is False
    assert Path(summary["persistent_scene_manifest"]).is_file()
    assert '"result_path"' in completed.stdout


def _pose_w2c(rotation_c2w: np.ndarray, center: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = rotation_c2w.T
    output[:3, 3] = -rotation_c2w.T @ center
    return output


def test_e2_chunk_consolidation_aligns_second_window_into_first_arbitrary_gauge(tmp_path: Path) -> None:
    from scripts.run_phase9_6e2_full_scene_da3_job import _consolidate_chunks

    height, width = 16, 20
    def _k(focal: float) -> np.ndarray:
        return np.array([[focal, 0.0, 9.5], [0.0, focal, 7.5], [0.0, 0.0, 1.0]], np.float32)

    # Real full-scene DA3 predicts intrinsics independently in every bounded
    # window.  Exercise >10% focal jitter here: E2 must preserve the raw
    # predictions as provenance while consolidating one provisional fixed-camera
    # estimate before the unchanged downstream F3 single-camera guard.
    intrinsics0 = np.stack([_k(18.0), _k(14.3), _k(20.0)])
    intrinsics1 = np.stack([_k(19.0), _k(16.0), _k(22.0)])
    confidence = np.ones((3, height, width), np.float32)
    confidence = np.ones((3, height, width), np.float32)
    global_depths = [np.full((height, width), 2.0 + 0.2 * index, np.float32) for index in range(4)]
    global_centres = [np.array([0.25 * index, 0.03 * index, 0.0]) for index in range(4)]
    global_rotation = np.eye(3)
    global_poses = [_pose_w2c(global_rotation, centre) for centre in global_centres]

    chunk0 = tmp_path / "chunk-0000.npz"
    np.savez_compressed(
        chunk0,
        depth=np.stack(global_depths[:3]),
        confidence=confidence,
        # Real DA3 emits OpenCV W2C extrinsics as 3x4. Keep this fixture in
        # that form so multi-window consolidation covers homogeneous-shape
        # normalization before later transformed windows become 4x4.
        extrinsics_w2c=np.stack(global_poses[:3])[:, :3, :],
        intrinsics=intrinsics0,
    )

    scale = 2.0
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([1.2, -0.4, 0.1])
    local_poses = []
    local_depths = []
    for frame_index in (1, 2, 3):
        global_center = global_centres[frame_index]
        local_center = rotation.T @ ((global_center - translation) / scale)
        local_rotation = rotation.T @ global_rotation
        local_poses.append(_pose_w2c(local_rotation, local_center))
        local_depths.append(global_depths[frame_index] / scale)
    chunk1 = tmp_path / "chunk-0001.npz"
    np.savez_compressed(
        chunk1,
        depth=np.stack(local_depths),
        confidence=confidence,
        extrinsics_w2c=np.stack(local_poses)[:, :3, :],
        intrinsics=intrinsics1,
    )

    summary = {
        "chunks": [
            {"frame_indices": [0, 1, 2], "npz_path": str(chunk0)},
            {"frame_indices": [1, 2, 3], "npz_path": str(chunk1)},
        ]
    }
    output, metadata = _consolidate_chunks(summary, 4, tmp_path / "out")
    arrays = np.load(output, allow_pickle=False)
    assert arrays["depth"].shape[0] == 4
    assert arrays["depth"][3] == pytest.approx(global_depths[3], rel=1e-5)
    assert arrays["extrinsics_w2c"].shape == (4, 4, 4)
    assert arrays["extrinsics_w2c"][3] == pytest.approx(global_poses[3], abs=1e-5)
    assert arrays["intrinsics"].shape == (4, 3, 3)
    assert np.max(np.abs(arrays["intrinsics"] - arrays["intrinsics"][0])) == pytest.approx(0.0)
    assert arrays["raw_intrinsics_per_frame"].shape == (4, 3, 3)
    assert metadata["intrinsics_consolidation"] == "robust_median_fixed_camera"
    assert metadata["raw_focal_max_relative_deviation"] > 0.20
    assert metadata["camera_calibrated"] is False
    assert metadata["window_count"] == 2
    assert metadata["loop_closure"] is False
    assert metadata["quality_certified"] is False


def test_e2_video_selection_spans_complete_clip_instead_of_midpoint_burst(tmp_path: Path) -> None:
    from types import SimpleNamespace
    import cv2
    from scripts.run_phase9_6e2_full_scene_da3_job import _select_video_keyframes

    video = tmp_path / "walkthrough.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (96, 64),
    )
    assert writer.isOpened()
    for index in range(80):
        yy, xx = np.mgrid[0:64, 0:96]
        base = (((xx // 8 + yy // 8 + index // 3) % 2) * 150 + 45).astype(np.uint8)
        frame = np.stack([base, np.roll(base, index % 17, axis=1), 255 - base], axis=2)
        writer.write(frame)
    writer.release()

    args = SimpleNamespace(
        video=str(video),
        target_fps=4.0,
        max_keyframes=10,
        minimum_spacing_seconds=0.4,
        forced_spacing_seconds=1.5,
        minimum_visual_change=0.03,
        minimum_quality_score=0.05,
    )
    paths, source_indices, timestamps, metadata = _select_video_keyframes(args, tmp_path / "selection")
    assert 3 <= len(paths) <= 10
    assert source_indices[0] <= 5
    assert source_indices[-1] >= 70
    assert timestamps == sorted(timestamps)
    assert metadata["selected_keyframe_count"] == len(paths)
    assert metadata["reported_frame_count"] >= 75
