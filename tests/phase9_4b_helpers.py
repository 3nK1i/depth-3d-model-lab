from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from app.reconstruction_canonical_cameras import build_canonical_camera_trajectory_snapshot
from core.reconstruction import SurfaceMeshResult, persistent_scene_from_pipeline_result
from core.reconstruction.da3 import DA3_EFFICIENT_MODEL_ID, DA3ResultManifest
from core.reconstruction.da3_provider import DA3InputManifest, write_da3_input_manifest
from core.reconstruction.gaussian_input_bridge import GaussianInputFrameEvidence
from core.reconstruction.orchestration import ReconstructionArtifactKind
from tests.reconstruction_f5_helpers import run_f5_fixture


def _source_hash(image_bgr: np.ndarray) -> str:
    image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    digest = sha256()
    digest.update(str(tuple(int(value) for value in image.shape)).encode("ascii"))
    digest.update(b"\0uint8\0")
    digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


def _write_source_images(root: Path, count: int, *, width: int = 96, height: int = 72):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    hashes = []
    for index in range(count):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = (np.arange(width, dtype=np.uint8)[None, :] + index * 7) % 255
        image[..., 1] = 40 + index * 30
        image[..., 2] = 180 - index * 10
        path = (root / f"source-{index:04d}.png").resolve()
        assert cv2.imwrite(str(path), image)
        paths.append(path)
        hashes.append(_source_hash(image))
    return tuple(paths), tuple(hashes)


def _poses(count: int) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], count, axis=0)
    for index in range(count):
        matrices[index, 0, 3] = -0.2 + 0.1 * index
        matrices[index, 2, 3] = 0.03 * index
    return matrices


def _surface_snapshot(surface: SurfaceMeshResult) -> dict:
    return {
        "metric_geometry_status": surface.metric_geometry_status.value,
        "unit_to_metre": surface.unit_to_metre,
        "relative_scale_uncertainty": surface.relative_scale_uncertainty,
        "scale_profile_id": surface.scale_profile_id,
        "world_alignment_status": surface.world_alignment_status.value,
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }


def make_phase9_4b_explicit_scene(tmp_path: Path):
    fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase9-4b-explicit"
    )
    surface = fixture.store.find_unique(SurfaceMeshResult)
    count = 5
    paths, hashes = _write_source_images(tmp_path / "sources", count)
    poses = _poses(count)
    base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    camera_artifact = next(
        item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    )
    frame_ids = tuple(f"canonical-{index:04d}" for index in range(count))
    updated = dict(snapshots)
    updated["surface_mesh"] = _surface_snapshot(surface)
    updated["dense_geometry"] = {"scale_profile_id": surface.scale_profile_id}
    updated["camera_trajectory"] = build_canonical_camera_trajectory_snapshot(
        matrices_reconstruction=poses,
        camera_ids=frame_ids,
        pose_layers=("optimized",) * count,
        provider_id=camera_artifact.provider_id,
        reconstruction_frame_id=surface.reconstruction_frame_id,
        source_frame_sha256=hashes,
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)
    frames = tuple(
        GaussianInputFrameEvidence(
            source_image_path=str(path),
            source_frame_sha256=digest,
            width=96,
            height=72,
            fx=82.0 + index,
            fy=83.0 + index,
            cx=48.0,
            cy=36.0,
        )
        for index, (path, digest) in enumerate(zip(paths, hashes))
    )
    return scene, frames, poses


def make_phase9_4b_da3_scene(tmp_path: Path):
    fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase9-4b-da3"
    )
    surface = fixture.store.find_unique(SurfaceMeshResult)
    count, width, height = 3, 80, 60
    paths, hashes = _write_source_images(tmp_path / "da3-sources", count, width=120, height=90)
    poses = _poses(count)
    extrinsics_w2c = np.linalg.inv(poses)
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], count, axis=0)
    for index in range(count):
        intrinsics[index, 0, 0] = 70.0 + index
        intrinsics[index, 1, 1] = 71.0 + index
        intrinsics[index, 0, 2] = width / 2.0
        intrinsics[index, 1, 2] = height / 2.0
    npz = (tmp_path / "da3-output.npz").resolve()
    np.savez_compressed(
        npz,
        depth=np.full((count, height, width), 2.0, dtype=np.float32),
        confidence=np.full((count, height, width), 0.9, dtype=np.float32),
        extrinsics_w2c=extrinsics_w2c,
        intrinsics=intrinsics,
    )
    request_id = "phase9-4b-da3-sidecars"
    input_manifest = DA3InputManifest(request_id=request_id, image_paths=tuple(str(p) for p in paths))
    input_path = write_da3_input_manifest((tmp_path / "da3-input.json").resolve(), input_manifest)
    result_manifest = DA3ResultManifest(
        request_id=request_id,
        model_id=DA3_EFFICIENT_MODEL_ID,
        model_revision="1" * 40,
        source_revision="2" * 40,
        output_npz_path=str(npz),
        frame_count=count,
        width=width,
        height=height,
        process_res=392,
        inference_duration_s=1.0,
        peak_vram_mib=128.0,
        gpu_name="fixture-gpu",
    )
    result_path = (tmp_path / "da3-result.json").resolve()
    result_path.write_text(__import__("json").dumps(result_manifest.as_dict(), indent=2, sort_keys=True) + "\n")

    base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    camera_artifact = next(
        item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    )
    updated = dict(snapshots)
    updated["surface_mesh"] = _surface_snapshot(surface)
    updated["dense_geometry"] = {"scale_profile_id": surface.scale_profile_id}
    updated["camera_trajectory"] = build_canonical_camera_trajectory_snapshot(
        matrices_reconstruction=poses,
        camera_ids=tuple(f"da3-{index:04d}" for index in range(count)),
        pose_layers=("da3_joint_pose",) * count,
        provider_id=camera_artifact.provider_id,
        reconstruction_frame_id=surface.reconstruction_frame_id,
        source_frame_sha256=hashes,
    )
    updated["depth_evidence"] = {
        "da3_input_manifest_path": str(input_path),
        "da3_result_manifest_path": str(result_path),
        "da3_output_npz_path": str(npz),
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)
    return scene, paths, poses, intrinsics
