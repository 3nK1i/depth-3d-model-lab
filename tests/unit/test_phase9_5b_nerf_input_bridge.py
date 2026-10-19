from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.contracts.validation import metadata_as_dict
from core.reconstruction import (
    MetricGeometryStatus,
    NerfInputFrameEvidence,
    NerfSceneRole,
    materialize_nerf_input_from_canonical_scene,
    materialize_nerf_input_from_da3_persistent_scene,
    read_neural_scene_dataset_manifest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import ReconstructionArtifactKind
from scripts.nerf_workers.common import export_nerfstudio_dataset
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene, make_phase9_4b_explicit_scene


def _to_nerf_evidence(frames):
    return tuple(
        NerfInputFrameEvidence(
            source_image_path=item.source_image_path,
            source_frame_sha256=item.source_frame_sha256,
            width=item.width,
            height=item.height,
            fx=item.fx,
            fy=item.fy,
            cx=item.cx,
            cy=item.cy,
        )
        for item in frames
    )


def test_5b_explicit_bridge_materializes_existing_neural_scene_dataset(tmp_path: Path) -> None:
    scene, gaussian_frames, poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    frames = _to_nerf_evidence(gaussian_frames)
    result = materialize_nerf_input_from_canonical_scene(scene, frames, output_dir=tmp_path / "nerf-input")
    assert result.scene_id == scene.scene_id
    assert result.frame_count == len(frames) == 5
    assert result.scene_role is NerfSceneRole.APPEARANCE_ONLY
    assert result.canonical_geometry_authoritative is False
    assert result.metric_scale_authoritative is False
    assert result.measurement_authoritative is False
    assert result.qc_authoritative is False
    assert result.certified_metrology is False
    assert result.training_started is False
    assert result.rendering_started is False
    assert result.dataset.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert result.dataset.unit_to_metre is None
    assert np.allclose(result.dataset.frames[3].T_reconstruction_from_camera, poses[3])
    assert Path(result.manifest_path).is_file()


def test_5b_manifest_and_nerfstudio_export_preserve_camera_evidence(tmp_path: Path) -> None:
    scene, gaussian_frames, poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    result = materialize_nerf_input_from_canonical_scene(
        scene, _to_nerf_evidence(gaussian_frames), output_dir=tmp_path / "bridge"
    )
    loaded = read_neural_scene_dataset_manifest(result.manifest_path)
    transforms = export_nerfstudio_dataset(
        loaded,
        tmp_path / "nerfstudio",
        training_frame_ids=tuple(frame.frame_id for frame in loaded.frames[:-1]),
        validation_frame_ids=(loaded.frames[-1].frame_id,),
    )
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    assert payload["orientation_override"] == "none"
    assert payload["depth3d_pose_source_convention"] == "T_reconstruction_from_camera"
    assert payload["depth3d_nerfstudio_camera_convention"] == "OpenGL_c2w"
    assert len(payload["frames"]) == 5
    converted = np.asarray(payload["frames"][2]["transform_matrix"], dtype=np.float64)
    assert np.allclose(converted[:, 0], poses[2][:, 0])
    assert np.allclose(converted[:3, 1:3], -poses[2][:3, 1:3])
    assert np.allclose(converted[:3, 3], poses[2][:3, 3])


def test_5b_source_order_cannot_override_canonical_camera_binding(tmp_path: Path) -> None:
    scene, gaussian_frames, poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    frames = tuple(reversed(_to_nerf_evidence(gaussian_frames)))
    result = materialize_nerf_input_from_canonical_scene(scene, frames, output_dir=tmp_path / "bridge")
    assert result.source_frame_sha256 == tuple(item.source_frame_sha256 for item in gaussian_frames)
    assert np.allclose(result.dataset.frames[0].T_reconstruction_from_camera, poses[0])


def test_5b_rejects_source_pixel_drift(tmp_path: Path) -> None:
    scene, gaussian_frames, _poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    frames = _to_nerf_evidence(gaussian_frames)
    image = cv2.imread(frames[0].source_image_path, cv2.IMREAD_COLOR)
    image[0, 0] = (255, 255, 255)
    assert cv2.imwrite(frames[0].source_image_path, image)
    with pytest.raises(ReconstructionValidationError, match="fingerprint mismatch"):
        materialize_nerf_input_from_canonical_scene(scene, frames, output_dir=tmp_path / "bridge")


def test_5b_requires_exact_complete_source_coverage(tmp_path: Path) -> None:
    scene, gaussian_frames, _poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    frames = _to_nerf_evidence(gaussian_frames)
    with pytest.raises(ReconstructionValidationError, match="exactly cover"):
        materialize_nerf_input_from_canonical_scene(scene, frames[:-1], output_dir=tmp_path / "bridge")


def test_5b_da3_adapter_reuses_persisted_sidecars_without_backend_execution(tmp_path: Path) -> None:
    scene, _paths, poses, intrinsics = make_phase9_4b_da3_scene(tmp_path / "fixture")
    result = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=tmp_path / "nerf-input")
    assert result.source_kind == "persisted_da3_sidecars"
    assert result.frame_count == 3
    assert result.dataset.frames[2].fx == pytest.approx(intrinsics[2, 0, 0])
    assert result.dataset.frames[2].fy == pytest.approx(intrinsics[2, 1, 1])
    assert np.allclose(result.dataset.frames[2].T_reconstruction_from_camera, poses[2])
    assert all(Path(frame.image_path).is_file() for frame in result.dataset.frames)
    assert result.training_started is False and result.rendering_started is False


def test_5b_da3_adapter_rejects_pose_drift(tmp_path: Path) -> None:
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "fixture")
    depth_artifact = next(item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.DEPTH_EVIDENCE)
    snapshot = metadata_as_dict(depth_artifact.payload_snapshot)  # type: ignore[arg-type]
    npz_path = Path(str(snapshot["da3_output_npz_path"]))
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["extrinsics_w2c"][0, 0, 3] += 0.2
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(ReconstructionValidationError, match="poses drift"):
        materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=tmp_path / "bridge")


def test_5b_result_contract_cannot_claim_geometry_or_metrology_authority(tmp_path: Path) -> None:
    scene, gaussian_frames, _poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    result = materialize_nerf_input_from_canonical_scene(
        scene, _to_nerf_evidence(gaussian_frames), output_dir=tmp_path / "bridge"
    )
    with pytest.raises(ReconstructionValidationError, match="cannot claim authority"):
        replace(result, canonical_geometry_authoritative=True)
    with pytest.raises(ReconstructionValidationError, match="cannot claim authority"):
        replace(result, certified_metrology=True)
