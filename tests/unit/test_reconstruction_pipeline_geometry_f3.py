from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    DA3_EFFICIENT_MODEL_ID,
    DA3ResultManifest,
    DenseDepthUnit,
    DenseSceneProfile,
    PipelineCameraTrajectoryEvidence,
    PipelineDepthEvidence,
    PipelineIngestionEvidence,
    PipelinePayloadStore,
    ReconstructionArtifactAuthority,
    ReconstructionProviderRole,
    arbitrary_scale_reference,
    bound_evidence_provider_descriptor,
    canonical_geometry_provider_descriptors,
    da3_camera_profile_from_result,
    da3_supporting_evidence_from_result,
    estimate_camera_geometry_from_horizontal_fov,
    normalize_da3_confidence_for_fusion,
    reconstruction_unit_depth_alignment,
)
from core.reconstruction.errors import ReconstructionValidationError


def _session() -> str:
    return str(uuid4())


def _camera(width: int = 8, height: int = 6):
    return estimate_camera_geometry_from_horizontal_fov(
        profile_id="f3-unit-camera",
        camera_id="f3-unit-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=70.0,
        focal_length_relative_uncertainty=0.2,
        reconstruction_frame_id="f3-unit-map",
    )


def test_payload_store_is_typed_and_rejects_duplicate_type_ref():
    store = PipelinePayloadStore()
    ref = store.put("value", payload_ref="memory://f3/value")
    assert store.get(ref, str) == "value"
    with pytest.raises(ReconstructionValidationError):
        store.get(ref, bytes)
    with pytest.raises(ReconstructionValidationError):
        store.put("again", payload_ref=ref)


def test_ingestion_evidence_requires_camera_pixel_space_match():
    camera = _camera()
    scale = arbitrary_scale_reference(profile_id="f3-unit-scale")
    with pytest.raises(ReconstructionValidationError):
        PipelineIngestionEvidence(
            session_id=_session(),
            frame_ids=("f0",),
            timestamps_ns=(1,),
            rgb_frames=(np.zeros((5, 8, 3), np.uint8),),
            camera_profile=camera,
            scale_profile=scale,
            reconstruction_frame_id="f3-unit-map",
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        )


def test_trajectory_rejects_nonrigid_pose():
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(ReconstructionValidationError):
        PipelineCameraTrajectoryEvidence(
            session_id=_session(),
            frame_ids=("f0",),
            T_reconstruction_from_camera=(bad,),
            pose_layers=("tracking",),
            source_provider_id="pose",
        )


def test_depth_evidence_requires_bounded_fusion_confidence():
    alignment = reconstruction_unit_depth_alignment(provenance="unit")
    with pytest.raises(ReconstructionValidationError):
        PipelineDepthEvidence(
            session_id=_session(),
            frame_ids=("f0",),
            depth_frames=(np.ones((4, 5), np.float32),),
            depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
            depth_alignments=(alignment,),
            confidence_frames=(np.full((4, 5), 1.1, np.float32),),
            source_provider_id="depth",
            geometry_scale="arbitrary",
        )


def test_normalize_da3_confidence_preserves_probability_range():
    source = np.asarray([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)
    result = normalize_da3_confidence_for_fusion(source)
    np.testing.assert_allclose(result, source)
    assert not result.flags.writeable


def test_normalize_da3_confidence_scales_unbounded_scores():
    source = np.asarray([[0.0, 2.0], [4.0, 10.0]], dtype=np.float32)
    result = normalize_da3_confidence_for_fusion(source)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
    assert result[1, 1] >= result[1, 0] >= result[0, 1]


def _da3_manifest(tmp_path: Path, *, spread: float = 0.0) -> DA3ResultManifest:
    n, h, w = 2, 4, 6
    depth = np.full((n, h, w), 2.0, dtype=np.float32)
    confidence = np.stack(
        [np.linspace(0.0, 2.0, h * w, dtype=np.float32).reshape(h, w) for _ in range(n)]
    )
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    extrinsics[1, 0, 3] = -0.1
    intrinsics = np.tile(
        np.asarray([[5.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]], np.float32),
        (n, 1, 1),
    )
    intrinsics[1, 0, 0] *= 1.0 + spread
    npz = tmp_path / "da3.npz"
    np.savez_compressed(
        npz,
        depth=depth,
        confidence=confidence,
        extrinsics_w2c=extrinsics,
        intrinsics=intrinsics,
    )
    return DA3ResultManifest(
        request_id="f3-da3",
        model_id=DA3_EFFICIENT_MODEL_ID,
        model_revision="1" * 40,
        source_revision="2" * 40,
        output_npz_path=str(npz.resolve()),
        frame_count=n,
        width=w,
        height=h,
        process_res=392,
        inference_duration_s=0.5,
        peak_vram_mib=100.0,
        gpu_name="validation-gpu",
    )


def test_da3_camera_profile_is_provisional_and_uses_output_pixel_space(tmp_path: Path):
    manifest = _da3_manifest(tmp_path)
    camera = da3_camera_profile_from_result(manifest, reconstruction_frame_id="f3-da3-map")
    assert camera.is_provisional
    assert not camera.is_calibrated
    assert camera.camera.image_size == ImageSize(6, 4)
    assert camera.camera.camera_frame.parent_frame_id == "f3-da3-map"


def test_da3_camera_profile_rejects_large_per_frame_intrinsics_drift(tmp_path: Path):
    manifest = _da3_manifest(tmp_path, spread=0.5)
    with pytest.raises(ReconstructionValidationError):
        da3_camera_profile_from_result(
            manifest,
            reconstruction_frame_id="f3-da3-map",
            maximum_intrinsics_relative_spread=0.10,
        )


def test_da3_supporting_evidence_never_claims_metric_or_canonical_authority(tmp_path: Path):
    manifest = _da3_manifest(tmp_path)
    session = _session()
    trajectory, depth = da3_supporting_evidence_from_result(
        manifest,
        session_id=session,
        frame_ids=("f0", "f1"),
    )
    assert trajectory.session_id == session
    assert trajectory.source_provider_id == "depth-anything-3-external"
    assert depth.depth_unit is DenseDepthUnit.SOURCE_MODEL_UNITS
    assert depth.geometry_scale == "relative_arbitrary_scale"
    assert all(item.mode.value == "explicit_scale_factor" for item in depth.depth_alignments)
    assert all(item.scale_reference_profile_id == "" for item in depth.depth_alignments)
    assert all(value is not None and float(value.max()) <= 1.0 for value in depth.confidence_frames)


def test_bound_evidence_descriptor_is_real_in_process_not_mock():
    descriptor = bound_evidence_provider_descriptor()
    assert descriptor.available
    assert descriptor.execution.value == "in_process"
    assert set(descriptor.roles) == {
        ReconstructionProviderRole.INGESTION,
        ReconstructionProviderRole.CAMERA_POSE,
        ReconstructionProviderRole.DEPTH,
    }
    assert dict(descriptor.metadata)["development_mock"] is False


def test_canonical_provider_descriptors_preserve_stage_authorities():
    descriptors = canonical_geometry_provider_descriptors()
    assert [item.provider_id for item in descriptors] == [
        "canonical-dense-fusion",
        "quality-aware-surface-mesh",
        "semantic-geometry-core",
        "registration-inspection-qc",
    ]
    assert all(item.execution.value == "in_process" for item in descriptors)
    assert dict(descriptors[0].metadata)["authority"] == ReconstructionArtifactAuthority.CANONICAL_GEOMETRY.value
    assert dict(descriptors[1].metadata)["authority"] == ReconstructionArtifactAuthority.CANONICAL_GEOMETRY.value
    assert dict(descriptors[2].metadata)["authority"] == ReconstructionArtifactAuthority.SEMANTIC_CONTEXT.value
    assert dict(descriptors[3].metadata)["authority"] == ReconstructionArtifactAuthority.INSPECTION_DECISION.value
