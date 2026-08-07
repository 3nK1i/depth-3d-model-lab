#!/usr/bin/env python3
"""Real Phase 8.4B isolated stella_vslam worker smoke on EuRoC MH_01.

This does not wire the application capture path yet (that is Phase 8.4C).  It
feeds calibrated EuRoC frames through the production shared-memory/AF_UNIX
worker boundary and validates real arbitrary-scale live observations.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from time import monotonic

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capture.frame import SourceFrame
from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
)
from core.models.metadata import IntrinsicsSource
from core.reconstruction import (
    BackendOperationContext,
    CameraDistortionModel,
    ExternalStellaLiveSlamBackend,
    InputSourceKind,
    InputSourceMetadata,
    LiveSharedMemoryFrameRing,
    LiveSlamRequest,
    LiveSourceFrameAdapter,
    LiveTrackingState,
    ReconstructionCamera,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    STELLA_LIVE_BACKEND_ID,
    TimestampOrigin,
    new_reconstruction_session_id,
)


def _load_euroc_index(mav0: Path, max_frames: int) -> list[tuple[int, Path]]:
    csv_path = mav0 / "cam0" / "data.csv"
    data_dir = mav0 / "cam0" / "data"
    if not csv_path.is_file() or not data_dir.is_dir():
        raise RuntimeError(f"EuRoC cam0 data is incomplete under {mav0}")
    records: list[tuple[int, Path]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            if raw_line.lstrip().startswith("#") or not raw_line.strip():
                continue
            row = next(csv.reader([raw_line]))
            if len(row) < 2:
                continue
            timestamp_ns = int(row[0].strip())
            image_path = data_dir / row[1].strip()
            if not image_path.is_file():
                raise RuntimeError(f"Missing EuRoC image: {image_path}")
            records.append((timestamp_ns, image_path))
            if len(records) >= max_frames:
                break
    if len(records) < 20:
        raise RuntimeError("EuRoC smoke needs at least 20 ordered frames.")
    return records


def _euroc_camera() -> ReconstructionCamera:
    image_size = ImageSize(752, 480)
    intrinsics = CameraIntrinsics(
        np.array(
            [[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "euroc-cam0-752x480",
        IntrinsicsSource.CALIBRATED,
        CalibrationStatus.CALIBRATED,
        calibration_record_id="euroc-mh01-cam0-official",
    )
    frame = CoordinateFrame(
        "euroc-cam0-frame",
        CoordinateFrameKind.CAMERA_GENERIC,
        AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        parent_frame_id="euroc-live-reconstruction",
    )
    return ReconstructionCamera(
        camera_id="euroc-cam0",
        image_size=image_size,
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stella-prefix",
        type=Path,
        default=Path.home() / "Desktop" / "eia-stella-vslam-probe",
    )
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path.home() / "Desktop" / "eia-stella-live-worker" / "bin" / "eia_stella_live_worker",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path.home() / "Desktop" / "phase8_4_stella_probe_data" / "orb_vocab.fbow",
    )
    parser.add_argument(
        "--euroc",
        type=Path,
        default=Path.home() / "Desktop" / "phase8_4_stella_probe_data" / "MH_01_easy" / "mav0",
    )
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/phase8_4b_live_stella_worker")
    )
    args = parser.parse_args()
    if args.max_frames < 20:
        parser.error("--max-frames must be at least 20")

    mav0 = args.euroc.expanduser().resolve()
    records = _load_euroc_index(mav0, args.max_frames)
    camera = _euroc_camera()
    session_id = new_reconstruction_session_id()
    source_fingerprint = sha256(str(mav0).encode("utf-8")).hexdigest()
    source = InputSourceMetadata(
        source_id=f"euroc-mh01-simulated-live-{source_fingerprint[:12]}",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="EuRoC MH_01 cam0 simulated-live stream",
        source_uri=f"euroc-simulated-live://{mav0}",
        expected_image_size=camera.image_size,
        expected_frame_rate_hz=20.0,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=source_fingerprint,
        metadata={"dataset": "EuRoC MH_01_easy", "camera": "cam0"},
    )
    profile = ReconstructionQualityProfile.EFFICIENT
    request = LiveSlamRequest(
        session_id=session_id,
        source=source,
        camera=camera,
        profile=profile,
        backend_id=STELLA_LIVE_BACKEND_ID,
        simulated_live=True,
        enable_mapping=True,
        enable_loop_closure=False,
        enable_global_optimization=False,
        metadata={"validation": "phase8.4b-real-worker"},
    )
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=camera.image_size.width * camera.image_size.height * 3,
    )
    adapter = LiveSourceFrameAdapter(
        session_id=session_id,
        source=source,
        camera=camera,
        ring=ring,
    )
    events = []
    control = ReconstructionOperationControl()
    context = BackendOperationContext(
        session_id=session_id,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=profile,
        control=control,
        emit_event=events.append,
    )
    backend = ExternalStellaLiveSlamBackend(
        install_prefix=args.stella_prefix,
        worker_executable=args.worker,
        vocabulary_path=args.vocab,
        frame_ring_spec=ring.spec,
        startup_timeout_s=20.0,
        frame_timeout_s=5.0,
        shutdown_timeout_s=10.0,
    )

    # SourceFrame.timestamp is defined by the stable capture contract as a
    # monotonic capture-clock timestamp.  Preserve EuRoC inter-frame timing
    # while anchoring the simulated-live replay to the local monotonic clock.
    dataset_origin_ns = records[0][0]
    capture_clock_origin_s = monotonic()

    observations = []
    try:
        backend.prepare(context)
        backend.start_live_session(request, context)
        for index, (timestamp_ns, image_path) in enumerate(records):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not decode {image_path}")
            if image.shape[:2] != (camera.image_size.height, camera.image_size.width):
                raise RuntimeError(f"Unexpected EuRoC image size: {image.shape}")
            capture_timestamp_s = capture_clock_origin_s + (
                (timestamp_ns - dataset_origin_ns) / 1_000_000_000.0
            )
            source_frame = SourceFrame(
                image=image,
                frame_id=index,
                timestamp=capture_timestamp_s,
            )
            live_frame = adapter.publish(source_frame)
            observations.append(backend.track_live_frame(live_frame, context))
            if (index + 1) % 100 == 0:
                pose_count = sum(item.pose is not None for item in observations)
                keyframes = max(
                    (
                        item.map_snapshot.keyframe_count
                        for item in observations
                        if item.map_snapshot is not None
                    ),
                    default=0,
                )
                print(
                    f"frames={index + 1} poses={pose_count} keyframes={keyframes}",
                    flush=True,
                )
        backend.stop_live_session(context)
    finally:
        backend.shutdown()
        ring.shutdown()

    state_counts = {state.value: 0 for state in LiveTrackingState}
    for observation in observations:
        state_counts[observation.tracking_state.value] += 1
    pose_count = sum(item.pose is not None for item in observations)
    keyframe_observations = sum(item.is_keyframe for item in observations)
    max_keyframes = max(
        (item.map_snapshot.keyframe_count for item in observations if item.map_snapshot),
        default=0,
    )
    max_landmarks = max(
        (item.map_snapshot.landmark_count for item in observations if item.map_snapshot),
        default=0,
    )
    latencies_ms = [item.quality.processing_latency_ns / 1_000_000.0 for item in observations]
    arbitrary_gauge_ok = all(
        item.pose is None or item.pose.gauge.unit_to_metre is None for item in observations
    )
    passed = (
        len(observations) == len(records)
        and pose_count >= 20
        and max_keyframes >= 1
        and max_landmarks > 0
        and arbitrary_gauge_ok
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "8.4B",
        "real_stella_executed": True,
        "worker_process_isolated": True,
        "frames_submitted": len(records),
        "observations_received": len(observations),
        "pose_observations": pose_count,
        "keyframe_observations": keyframe_observations,
        "max_keyframes": max_keyframes,
        "max_landmarks": max_landmarks,
        "tracking_states": state_counts,
        "median_tracking_latency_ms": statistics.median(latencies_ms) if latencies_ms else None,
        "mean_tracking_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else None,
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": "arbitrary",
        "unit_to_metre": None,
        "loop_closure_enabled": False,
        "global_optimization_feature_enabled": False,
        "environment_probe": None if backend.environment_probe is None else backend.environment_probe.as_dict(),
        "pass": passed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Phase 8.4B real isolated stella worker smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"frames={len(records)} observations={len(observations)} poses={pose_count} "
        f"keyframes={max_keyframes} landmarks={max_landmarks}"
    )
    print(f"tracking_states={state_counts}")
    if latencies_ms:
        print(
            f"median_tracking_latency_ms={statistics.median(latencies_ms):.3f} "
            f"mean_tracking_latency_ms={statistics.fmean(latencies_ms):.3f}"
        )
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    print("loop_closure_enabled=False global_optimization_feature_enabled=False")
    print(f"summary={output_dir / 'summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
