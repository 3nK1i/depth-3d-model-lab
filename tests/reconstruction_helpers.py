from __future__ import annotations

import numpy as np

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
    RigidTransform3D,
)
from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.models.metadata import IntrinsicsSource
from core.reconstruction import (
    CameraDistortionModel,
    CameraPose,
    InputSourceKind,
    InputSourceMetadata,
    ReconstructionCamera,
    ReconstructionFrame,
    ReconstructionKeyframe,
    ReconstructionQualityProfile,
    ReconstructionSession,
    TimestampOrigin,
    KeyframeSelectionReason,
)

SESSION_ID = "00000000-0000-4000-8000-000000000001"
CREATED_AT = "2026-08-06T15:00:00+00:00"
IMAGE_SIZE = ImageSize(640, 480)


def make_source(kind: InputSourceKind = InputSourceKind.IMAGE_SEQUENCE) -> InputSourceMetadata:
    uri = "sequence://synthetic" if kind is InputSourceKind.IMAGE_SEQUENCE else "video://synthetic"
    if kind is InputSourceKind.USB_CAMERA:
        uri = "usb://0"
    elif kind is InputSourceKind.RTSP_STREAM:
        uri = "rtsp://example.invalid/stream"
    return InputSourceMetadata(
        source_id="source-1",
        kind=kind,
        display_name="Synthetic source",
        source_uri=uri,
        expected_image_size=IMAGE_SIZE,
        expected_frame_rate_hz=30.0,
        timestamp_origin=TimestampOrigin.SYNTHETIC,
        fingerprint="abc123",
        metadata={"purpose": "test"},
    )


def make_camera() -> ReconstructionCamera:
    intrinsics = CameraIntrinsics(
        np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "rgb",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    frame = CoordinateFrame(
        "camera-1-frame",
        CoordinateFrameKind.CAMERA_GENERIC,
        AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        parent_frame_id="world",
    )
    return ReconstructionCamera(
        camera_id="camera-1",
        image_size=IMAGE_SIZE,
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=CameraDistortionModel.NONE,
    )


def make_pose(index: int, *, timestamp_ns: int | None = None) -> CameraPose:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = index * 0.1
    matrix[1, 3] = index * 0.02
    transform = RigidTransform3D(
        "camera-1-frame",
        "world",
        matrix,
        method="synthetic",
    )
    return CameraPose(
        pose_id=f"pose-{index}",
        camera_id="camera-1",
        frame_id=f"frame-{index}",
        timestamp_ns=timestamp_ns if timestamp_ns is not None else (index + 1) * 1_000_000,
        T_world_from_camera=transform,
        tracking_confidence=0.95,
    )


def make_session(*, frame_count: int = 3) -> ReconstructionSession:
    session = ReconstructionSession(
        make_source(),
        ReconstructionQualityProfile.BALANCED,
        session_id=SESSION_ID,
        created_at_utc=CREATED_AT,
    )
    session.prepare()
    for index in range(frame_count):
        session.add_frame(
            ReconstructionFrame(
                frame_id=f"frame-{index}",
                session_id=SESSION_ID,
                source_id="source-1",
                sequence_index=index,
                timestamp_ns=(index + 1) * 1_000_000,
                image_size=IMAGE_SIZE,
            )
        )
    for index in range(min(frame_count, 2)):
        session.add_keyframe(
            ReconstructionKeyframe(
                keyframe_id=f"keyframe-{index}",
                session_id=SESSION_ID,
                frame_id=f"frame-{index}",
                selection_sequence=index,
                reason=(
                    KeyframeSelectionReason.FIRST_FRAME
                    if index == 0
                    else KeyframeSelectionReason.MOTION
                ),
                score=1.0 - index * 0.1,
                pose_id=f"pose-{index}",
            )
        )
    return session


def make_hardware(
    *,
    total_gib: float | None,
    free_gib: float | None = None,
    name: str = "Test GPU",
    capability: tuple[int, int] = (7, 5),
    cuda_ready: bool = True,
) -> HardwareInfo:
    backend = AcceleratorBackend.CPU if total_gib is None else AcceleratorBackend.CUDA
    if free_gib is None and total_gib is not None:
        free_gib = total_gib * 0.8
    return HardwareInfo(
        backend=backend,
        device="cpu" if backend is AcceleratorBackend.CPU else "cuda:0",
        device_index=None if backend is AcceleratorBackend.CPU else 0,
        name="CPU" if backend is AcceleratorBackend.CPU else name,
        total_memory_bytes=None if total_gib is None else int(total_gib * 1024**3),
        free_memory_bytes=None if free_gib is None else int(free_gib * 1024**3),
        compute_capability=None if backend is AcceleratorBackend.CPU else capability,
        supports_fp16=backend is AcceleratorBackend.CUDA,
        supports_bf16=backend is AcceleratorBackend.CUDA and capability[0] >= 8,
        torch_available=True,
        torch_cuda_available=backend is AcceleratorBackend.CUDA and cuda_ready,
        torch_version="test",
        torch_cuda_version=None if backend is AcceleratorBackend.CPU else "test",
        compiled_architectures=() if backend is AcceleratorBackend.CPU else ("sm_75",),
        torch_build_compatible=None if backend is AcceleratorBackend.CPU else cuda_ready,
    )
