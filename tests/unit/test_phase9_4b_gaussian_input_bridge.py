from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.contracts.validation import metadata_as_dict
from core.reconstruction import (
    GaussianInputFrameEvidence,
    MetricGeometryStatus,
    read_neural_scene_dataset_manifest,
    materialize_gaussian_input_from_canonical_scene,
    materialize_gaussian_input_from_da3_persistent_scene,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.gaussian_splatting import GaussianSceneRole
from core.reconstruction.orchestration import ReconstructionArtifactKind
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene, make_phase9_4b_explicit_scene


def test_4b_explicit_bridge_materializes_existing_neural_scene_dataset(tmp_path: Path) -> None:
    scene, frames, poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    result = materialize_gaussian_input_from_canonical_scene(
        scene, frames, output_dir=tmp_path / "gaussian-input"
    )

    assert result.scene_id == scene.scene_id
    assert result.frame_count == len(frames) == 5
    assert result.scene_role is GaussianSceneRole.APPEARANCE_ONLY
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


def test_4b_manifest_round_trip_preserves_fingerprint_and_camera_evidence(tmp_path: Path) -> None:
    scene, frames, _poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    result = materialize_gaussian_input_from_canonical_scene(
        scene, frames, output_dir=tmp_path / "bridge"
    )
    loaded = read_neural_scene_dataset_manifest(result.manifest_path)
    assert loaded.as_dict() == result.dataset.as_dict()
    assert [frame.fx for frame in loaded.frames] == [frame.fx for frame in result.dataset.frames]
    assert result.dataset_fingerprint_sha256


def test_4b_source_evidence_order_does_not_override_canonical_camera_binding(tmp_path: Path) -> None:
    scene, frames, poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    reversed_frames = tuple(reversed(frames))
    result = materialize_gaussian_input_from_canonical_scene(
        scene, reversed_frames, output_dir=tmp_path / "bridge"
    )
    assert result.source_frame_sha256 == tuple(item.source_frame_sha256 for item in frames)
    assert np.allclose(result.dataset.frames[0].T_reconstruction_from_camera, poses[0])
    assert result.dataset.frames[0].fx == frames[0].fx


def test_4b_rejects_source_pixel_drift_instead_of_pairing_wrong_image(tmp_path: Path) -> None:
    scene, frames, _poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    image = cv2.imread(frames[0].source_image_path, cv2.IMREAD_COLOR)
    image[0, 0] = (255, 255, 255)
    assert cv2.imwrite(frames[0].source_image_path, image)
    with pytest.raises(ReconstructionValidationError, match="fingerprint mismatch"):
        materialize_gaussian_input_from_canonical_scene(
            scene, frames, output_dir=tmp_path / "bridge"
        )


def test_4b_requires_exact_complete_source_fingerprint_coverage(tmp_path: Path) -> None:
    scene, frames, _poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    with pytest.raises(ReconstructionValidationError, match="exactly cover"):
        materialize_gaussian_input_from_canonical_scene(
            scene, frames[:-1], output_dir=tmp_path / "bridge"
        )


def test_4b_frame_evidence_rejects_relative_paths_and_invalid_intrinsics(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError, match="absolute"):
        GaussianInputFrameEvidence(
            source_image_path="relative.png",
            source_frame_sha256="a" * 64,
            width=64,
            height=48,
            fx=50.0,
            fy=50.0,
            cx=32.0,
            cy=24.0,
        )
    with pytest.raises(ReconstructionValidationError, match="positive focal"):
        GaussianInputFrameEvidence(
            source_image_path=str((tmp_path / "source.png").resolve()),
            source_frame_sha256="a" * 64,
            width=64,
            height=48,
            fx=0.0,
            fy=50.0,
            cx=32.0,
            cy=24.0,
        )


def test_4b_da3_adapter_reuses_persisted_sidecars_without_backend_execution(tmp_path: Path) -> None:
    scene, _paths, poses, intrinsics = make_phase9_4b_da3_scene(tmp_path / "fixture")
    result = materialize_gaussian_input_from_da3_persistent_scene(
        scene, output_dir=tmp_path / "gaussian-input"
    )
    assert result.source_kind == "persisted_da3_sidecars"
    assert result.frame_count == 3
    assert result.dataset.frames[0].width == 80
    assert result.dataset.frames[0].height == 60
    assert result.dataset.frames[2].fx == pytest.approx(intrinsics[2, 0, 0])
    assert result.dataset.frames[2].fy == pytest.approx(intrinsics[2, 1, 1])
    assert np.allclose(result.dataset.frames[2].T_reconstruction_from_camera, poses[2])
    assert all(Path(frame.image_path).is_file() for frame in result.dataset.frames)
    assert result.training_started is False and result.rendering_started is False


def test_4b_da3_adapter_rejects_pose_drift_between_sidecar_and_canonical_scene(tmp_path: Path) -> None:
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "fixture")
    depth_artifact = next(item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.DEPTH_EVIDENCE)
    snapshot = metadata_as_dict(depth_artifact.payload_snapshot)  # type: ignore[arg-type]
    npz_path = Path(str(snapshot["da3_output_npz_path"]))
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["extrinsics_w2c"][0, 0, 3] += 0.2
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(ReconstructionValidationError, match="poses drift"):
        materialize_gaussian_input_from_da3_persistent_scene(
            scene, output_dir=tmp_path / "bridge"
        )


def test_4b_da3_adapter_rejects_source_image_drift(tmp_path: Path) -> None:
    scene, paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "fixture")
    image = cv2.imread(str(paths[1]), cv2.IMREAD_COLOR)
    image[-1, -1] = (1, 2, 3)
    assert cv2.imwrite(str(paths[1]), image)
    with pytest.raises(ReconstructionValidationError, match="fingerprint"):
        materialize_gaussian_input_from_da3_persistent_scene(
            scene, output_dir=tmp_path / "bridge"
        )


def test_4b_result_contract_cannot_be_promoted_to_geometry_authority(tmp_path: Path) -> None:
    scene, frames, _poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    result = materialize_gaussian_input_from_canonical_scene(
        scene, frames, output_dir=tmp_path / "bridge"
    )
    with pytest.raises(ReconstructionValidationError, match="cannot claim authority"):
        replace(result, canonical_geometry_authoritative=True)
    with pytest.raises(ReconstructionValidationError, match="cannot claim authority"):
        replace(result, certified_metrology=True)
