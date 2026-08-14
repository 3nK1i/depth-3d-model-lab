from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.inspection import load_inspection_bundle
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SemanticClassCatalog,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    assign_workspace_class,
    set_workspace_review_status,
)
from core.perception.reviewed_analysis import reviewed_selection_from_workspace, save_reviewed_semantic_selection


ROOT = Path(__file__).resolve().parents[2]


def _handoff(tmp_path: Path) -> tuple[Path, Path]:
    height, width = 72, 96
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(30, 180, width, dtype=np.uint8)
    image[..., 1] = 90
    image_path = tmp_path / "frame.npy"
    np.save(image_path, image)

    frame = FrameReference(7, 8, 0.0, ImageSize(width, height))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    catalog, plant = SemanticClassCatalog().with_custom_class("plant")
    catalog, suitcase = catalog.with_custom_class("suitcase")

    def proposal(object_id: str, mask_id: str, bounds: tuple[int, int, int, int]) -> SemanticWorkspaceObject:
        y0, y1, x0, x1 = bounds
        dense = np.zeros((height, width), dtype=bool)
        dense[y0:y1, x0:x1] = True
        mask = InstanceMask(
            mask_id=mask_id,
            frame_reference=frame,
            pixel_space=space,
            payload=DenseMaskPayload(dense),
            source=MaskSource.NATIVE_INSTANCE,
            model_provenance=ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
        )
        return SemanticWorkspaceObject(object_id, proposal_mask=mask)

    workspace = SemanticWorkspace(
        "phase9-2g-integration",
        frame,
        catalog,
        (
            proposal("proposal:plant", "plant-mask", (8, 62, 7, 41)),
            proposal("proposal:suitcase", "suitcase-mask", (14, 67, 55, 90)),
        ),
    )
    workspace = assign_workspace_class(workspace, "proposal:plant", plant, instance_name="plant_001")
    workspace = set_workspace_review_status(workspace, "proposal:plant", SemanticReviewStatus.ACCEPTED)
    workspace = assign_workspace_class(workspace, "proposal:suitcase", suitcase, instance_name="suitcase_001")
    workspace = set_workspace_review_status(workspace, "proposal:suitcase", SemanticReviewStatus.ACCEPTED)
    selection = reviewed_selection_from_workspace(workspace, image)
    metadata_path, _ = save_reviewed_semantic_selection(selection, tmp_path / "reviewed")
    return image_path, metadata_path


def test_9_2g_isolated_measurement_job_uses_reviewed_masks_and_skips_perception_model(tmp_path: Path) -> None:
    image_path, reviewed_path = _handoff(tmp_path)
    depth_config = tmp_path / "depth.json"
    depth_config.write_text(
        json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0.0}),
        encoding="utf-8",
    )
    output = tmp_path / "result"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_measurement_inspection_job.py"),
            "--input", str(image_path),
            "--output", str(output),
            "--depth-model", "development-mock",
            "--depth-config", str(depth_config),
            # Deliberately invalid: success proves reviewed masks bypass perception inference.
            "--perception-model", "this-model-must-never-load",
            "--reviewed-semantics", str(reviewed_path),
            "--max-objects", "4",
            "--minimum-mask-pixels", "64",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "perception model inference skipped" in completed.stdout
    summary, arrays = load_inspection_bundle(output)
    assert summary.perception_model_id == "reviewed-auto-segment"
    assert summary.perception_model_name == "Reviewed Auto Segment"
    assert summary.semantic_source == "reviewed_auto_segment"
    assert summary.semantic_workspace_id == "phase9-2g-integration"
    assert len(summary.objects) == 2
    assert [obj.instance_name for obj in summary.objects] == ["plant_001", "suitcase_001"]
    assert all(obj.semantic_source == "reviewed_auto_segment" for obj in summary.objects)
    assert all(obj.semantic_review_status == "accepted" for obj in summary.objects)
    assert all(np.count_nonzero(arrays[obj.mask_key]) > 0 for obj in summary.objects)


def test_9_2g_job_fails_closed_when_reviewed_masks_belong_to_different_image(tmp_path: Path) -> None:
    image_path, reviewed_path = _handoff(tmp_path)
    image = np.load(image_path, allow_pickle=False)
    image[0, 0] = (255, 0, 0)
    np.save(image_path, image)
    depth_config = tmp_path / "depth.json"
    depth_config.write_text(json.dumps({"device": "cpu", "precision": "fp32"}), encoding="utf-8")
    output = tmp_path / "result"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_measurement_inspection_job.py"),
            "--input", str(image_path),
            "--output", str(output),
            "--depth-model", "development-mock",
            "--depth-config", str(depth_config),
            "--reviewed-semantics", str(reviewed_path),
            "--max-objects", "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "different source frame" in completed.stdout


def test_9_2g_reviewed_masks_are_rectified_with_camera_before_3d_analysis(tmp_path: Path) -> None:
    from core.contracts import (
        AxisConvention,
        CalibrationStatus,
        CameraIntrinsics,
        CoordinateFrame,
        CoordinateFrameKind,
    )
    from core.models.metadata import IntrinsicsSource
    from core.reconstruction.calibration_profile import (
        CameraCalibrationMethod,
        CameraCalibrationProfile,
        CameraCalibrationValidationState,
        save_camera_calibration_profile,
    )
    from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera

    image_path, reviewed_path = _handoff(tmp_path)
    size = ImageSize(96, 72)
    profile = CameraCalibrationProfile(
        profile_id="phase9-2g-integration-calibration",
        camera=ReconstructionCamera(
            camera_id="camera",
            image_size=size,
            intrinsics=CameraIntrinsics(
                np.array([[82.0, 0.0, 48.0], [0.0, 83.0, 36.0], [0.0, 0.0, 1.0]]),
                "rgb-rectified",
                IntrinsicsSource.CALIBRATED,
                CalibrationStatus.CALIBRATED,
                calibration_record_id="phase9-2g-integration-record",
            ),
            camera_frame=CoordinateFrame(
                "camera-frame",
                CoordinateFrameKind.CAMERA_GENERIC,
                AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
                parent_frame_id="scene",
            ),
            distortion_model=CameraDistortionModel.BROWN_CONRADY,
            distortion_coefficients=(-0.06, 0.01, 0.0, 0.0, 0.0),
        ),
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.UNVALIDATED,
        provenance="phase9-2g-integration",
        created_at_utc="2026-08-14T12:00:00+00:00",
    )
    profile_path = save_camera_calibration_profile(profile, tmp_path / "calibration.json")
    depth_config = tmp_path / "depth.json"
    depth_config.write_text(json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0.0}), encoding="utf-8")
    output = tmp_path / "calibrated-result"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_measurement_inspection_job.py"),
            "--input", str(image_path),
            "--output", str(output),
            "--depth-model", "development-mock",
            "--depth-config", str(depth_config),
            "--perception-model", "this-model-must-never-load",
            "--reviewed-semantics", str(reviewed_path),
            "--calibration-profile", str(profile_path),
            "--max-objects", "4",
            "--minimum-mask-pixels", "64",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary, _arrays = load_inspection_bundle(output)
    assert summary.semantic_source == "reviewed_auto_segment"
    assert summary.perception_model_id == "reviewed-auto-segment"
    assert summary.calibration_status == "calibrated"
    assert summary.calibration_rectified is True
    assert summary.calibration_profile_id == "phase9-2g-integration-calibration"
    assert all(obj.semantic_source == "reviewed_auto_segment" for obj in summary.objects)
