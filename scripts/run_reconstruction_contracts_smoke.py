#!/usr/bin/env python3
"""Dependency-light smoke test for the Phase 8.1 reconstruction foundation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    BackendProvenance,
    CameraDistortionModel,
    CameraPose,
    CheckpointMetadata,
    EventSeverity,
    InputSourceKind,
    InputSourceMetadata,
    KeyframeSelectionReason,
    ReconstructionCamera,
    ReconstructionEventType,
    ReconstructionFrame,
    ReconstructionKeyframe,
    ReconstructionMetadataBundle,
    ReconstructionProvenance,
    ReconstructionQualityProfile,
    ReconstructionSession,
    ReconstructionStage,
    ReconstructionTrajectory,
    SparseLandmark,
    SparseObservation,
    TimestampOrigin,
    metadata_bundle_from_json,
    metadata_bundle_to_json,
    resolve_reconstruction_profile,
)
from core.version import __version__

_SESSION_ID = "00000000-0000-4000-8000-000000008001"
_CREATED_AT = "2026-08-06T15:00:00+00:00"
_GIB = 1024**3


def _hardware(name: str, total_gib: float, free_gib: float, capability: tuple[int, int]) -> HardwareInfo:
    return HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device="cuda:0",
        device_index=0,
        name=name,
        total_memory_bytes=int(total_gib * _GIB),
        free_memory_bytes=int(free_gib * _GIB),
        compute_capability=capability,
        supports_fp16=True,
        supports_bf16=capability[0] >= 8,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="smoke",
        torch_cuda_version="smoke",
        compiled_architectures=(f"sm_{capability[0]}{capability[1]}",),
        torch_build_compatible=True,
    )


def _pose(index: int) -> CameraPose:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = index * 0.25
    matrix[1, 3] = index * 0.02
    return CameraPose(
        pose_id=f"pose-{index}",
        camera_id="camera-1",
        frame_id=f"frame-{index}",
        timestamp_ns=(index + 1) * 1_000_000_000,
        T_world_from_camera=RigidTransform3D(
            "camera-1-frame",
            "world",
            matrix,
            method="phase8_1_smoke",
        ),
        tracking_confidence=0.95,
    )


def _build_bundle() -> tuple[ReconstructionSession, ReconstructionMetadataBundle]:
    size = ImageSize(1280, 720)
    source = InputSourceMetadata(
        source_id="source-1",
        kind=InputSourceKind.IMAGE_SEQUENCE,
        display_name="Synthetic Phase 8.1 sequence",
        source_uri="sequence://phase8-1-smoke",
        expected_image_size=size,
        expected_frame_rate_hz=1.0,
        timestamp_origin=TimestampOrigin.SYNTHETIC,
        fingerprint="phase8-1-smoke",
    )
    session = ReconstructionSession(
        source,
        ReconstructionQualityProfile.BALANCED,
        session_id=_SESSION_ID,
        created_at_utc=_CREATED_AT,
    )
    session.prepare()
    for index in range(3):
        session.add_frame(
            ReconstructionFrame(
                frame_id=f"frame-{index}",
                session_id=_SESSION_ID,
                source_id=source.source_id,
                sequence_index=index,
                timestamp_ns=(index + 1) * 1_000_000_000,
                image_size=size,
            )
        )
    for index in range(2):
        session.add_keyframe(
            ReconstructionKeyframe(
                keyframe_id=f"keyframe-{index}",
                session_id=_SESSION_ID,
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

    intrinsics = CameraIntrinsics(
        np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "rgb",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    camera = ReconstructionCamera(
        camera_id="camera-1",
        image_size=size,
        intrinsics=intrinsics,
        camera_frame=CoordinateFrame(
            "camera-1-frame",
            CoordinateFrameKind.CAMERA_GENERIC,
            AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
            parent_frame_id="world",
        ),
        distortion_model=CameraDistortionModel.NONE,
    )
    trajectory = ReconstructionTrajectory((_pose(0), _pose(1)))
    landmark = SparseLandmark(
        "landmark-1",
        "track-1",
        np.array([0.2, 0.1, 2.5], dtype=np.float64),
        (
            SparseObservation(
                "observation-0",
                "landmark-1",
                "keyframe-0",
                (600.0, 350.0),
                feature_index=10,
                reprojection_error_px=0.3,
                confidence=0.95,
            ),
            SparseObservation(
                "observation-1",
                "landmark-1",
                "keyframe-1",
                (570.0, 351.0),
                feature_index=12,
                reprojection_error_px=0.4,
                confidence=0.93,
            ),
        ),
        color_rgb=(180, 190, 200),
    )

    session.start()
    session.events.emit(
        event_type=ReconstructionEventType.STATUS,
        stage=ReconstructionStage.FOUNDATION,
        message="Reconstruction contract smoke started.",
        elapsed_ns=1,
    )
    session.events.emit(
        event_type=ReconstructionEventType.PROGRESS,
        stage=ReconstructionStage.KEYFRAME_SELECTION,
        message="Synthetic keyframes prepared.",
        elapsed_ns=2,
        completed_work=2,
        total_work=2,
        progress_fraction=1.0,
    )
    session.events.emit(
        event_type=ReconstructionEventType.WARNING,
        stage=ReconstructionStage.FOUNDATION,
        message="Synthetic metadata only; no real reconstruction backend executed.",
        elapsed_ns=3,
        severity=EventSeverity.WARNING,
    )
    session.pause()

    snapshot = session.snapshot()
    checkpoint = CheckpointMetadata.create(
        session_id=_SESSION_ID,
        application_version=__version__,
        session_state=snapshot.state,
        completed_stage=ReconstructionStage.KEYFRAME_SELECTION,
        profile=ReconstructionQualityProfile.BALANCED,
        source_fingerprint=source.fingerprint,
        backend_versions=(("phase8-1-contract-smoke", "1"),),
        last_frame_id="frame-2",
        last_keyframe_id="keyframe-1",
    )
    bundle = ReconstructionMetadataBundle(
        session=snapshot,
        cameras=(camera,),
        trajectory=trajectory,
        sparse_landmarks=(landmark,),
        checkpoint=checkpoint,
        provenance=ReconstructionProvenance(
            application_version=__version__,
            source_fingerprint=source.fingerprint,
            hardware_label="Synthetic Phase 8.1 smoke hardware",
            backends=(BackendProvenance("phase8-1-contract-smoke", "1"),),
            created_at_utc=_CREATED_AT,
            metadata={"real_backend_executed": False},
        ),
    )
    return session, bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/tmp/phase8_1_reconstruction_metadata.json",
        help="Path for the serialized synthetic reconstruction metadata.",
    )
    args = parser.parse_args()

    try:
        session, bundle = _build_bundle()
        gpu_2060 = resolve_reconstruction_profile(
            _hardware("NVIDIA GeForce RTX 2060", 6.0, 5.2, (7, 5))
        )
        gpu_5090 = resolve_reconstruction_profile(
            _hardware("NVIDIA GeForce RTX 5090", 32.0, 30.0, (12, 0))
        )
        forced_profiles = {
            profile.display_name: resolve_reconstruction_profile(
                _hardware("NVIDIA GeForce RTX 5090", 32.0, 30.0, (12, 0)),
                profile,
            ).effective.display_name
            for profile in ReconstructionQualityProfile
        }

        payload = metadata_bundle_to_json(bundle)
        restored = metadata_bundle_from_json(payload)
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")

        assert restored.session.session_id == _SESSION_ID
        assert len(restored.session.frames) == 3
        assert len(restored.session.keyframes) == 2
        assert len(restored.trajectory.samples) == 2
        assert len(restored.sparse_landmarks) == 1
        assert gpu_2060.effective is ReconstructionQualityProfile.EFFICIENT
        assert gpu_5090.effective is ReconstructionQualityProfile.HIGH_QUALITY
        assert session.state.value == "paused"
        session.resume()
        session.request_cancel()
        session.acknowledge_cancelled()
        assert session.state.value == "cancelled"

        print("Phase 8.1 reconstruction contracts smoke: PASS")
        print(f"session_id={restored.session.session_id}")
        print(
            f"frames={len(restored.session.frames)} "
            f"keyframes={len(restored.session.keyframes)} "
            f"trajectory_samples={len(restored.trajectory.samples)} "
            f"landmarks={len(restored.sparse_landmarks)}"
        )
        print(
            "RTX 2060 6 GB -> "
            f"{gpu_2060.effective.display_name}, "
            f"GPU budget={gpu_2060.memory_policy.gpu_budget_bytes / _GIB:.2f} GiB"
        )
        print(
            "RTX 5090 32 GB -> "
            f"{gpu_5090.effective.display_name}, "
            f"GPU budget={gpu_5090.memory_policy.gpu_budget_bytes / _GIB:.2f} GiB"
        )
        print("Explicit profile selection=" + json.dumps(forced_profiles, sort_keys=True))
        print(f"events={len(session.events.events)} final_state={session.state.value}")
        print(f"metadata={output}")
        return 0
    except Exception as exc:
        print(f"Phase 8.1 reconstruction contracts smoke: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
