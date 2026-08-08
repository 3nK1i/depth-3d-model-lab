#!/usr/bin/env python3
"""Real Phase 8.6A stella smoke using explicitly provisional camera intrinsics.

This proves that the existing isolated stella worker can consume an estimated
pinhole camera only when the caller opts into the provisional policy.  The
result remains arbitrary-scale and is not a metric-calibration claim.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from time import monotonic

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capture.frame import SourceFrame
from core.contracts import ImageSize
from core.reconstruction import (
    BackendOperationContext,
    CameraIntrinsicsPolicy,
    ExternalStellaLiveSlamBackend,
    InputSourceKind,
    InputSourceMetadata,
    LiveSharedMemoryFrameRing,
    LiveSlamRequest,
    LiveSourceFrameAdapter,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    STELLA_LIVE_BACKEND_ID,
    TimestampOrigin,
    estimate_camera_geometry_from_horizontal_fov,
    new_reconstruction_session_id,
)
from scripts.run_live_stella_worker_smoke import _load_euroc_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stella-prefix", type=Path, default=Path.home() / "Desktop" / "eia-stella-vslam-probe")
    parser.add_argument("--worker", type=Path, default=Path.home() / "Desktop" / "eia-stella-live-worker" / "bin" / "eia_stella_live_worker")
    parser.add_argument("--vocab", type=Path, default=Path.home() / "Desktop" / "phase8_4_stella_probe_data" / "orb_vocab.fbow")
    parser.add_argument("--euroc", type=Path, default=Path.home() / "Desktop" / "phase8_4_stella_probe_data" / "MH_01_easy" / "mav0")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--horizontal-fov", type=float, default=79.0)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_6a_estimated_camera_stella"))
    args = parser.parse_args()
    if args.max_frames < 100:
        parser.error("--max-frames must be at least 100")

    mav0 = args.euroc.expanduser().resolve()
    records = _load_euroc_index(mav0, args.max_frames)
    profile = estimate_camera_geometry_from_horizontal_fov(
        profile_id="euroc-estimated-fov-phase8-6a",
        camera_id="euroc-cam0-estimated",
        image_size=ImageSize(752, 480),
        horizontal_fov_degrees=args.horizontal_fov,
        focal_length_relative_uncertainty=0.20,
        camera_frame_id="euroc-cam0-estimated-frame",
        reconstruction_frame_id="euroc-live-reconstruction",
        notes="Validation-only assumed-FOV profile; not EuRoC calibration",
    )
    camera = profile.camera
    session_id = new_reconstruction_session_id()
    fingerprint = sha256(str(mav0).encode("utf-8")).hexdigest()
    source = InputSourceMetadata(
        source_id=f"euroc-estimated-phase8-6a-{fingerprint[:12]}",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="EuRoC Phase 8.6A estimated-camera simulation",
        source_uri=f"euroc-estimated://{mav0}",
        expected_image_size=camera.image_size,
        expected_frame_rate_hz=20.0,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=fingerprint,
        metadata={"dataset": "EuRoC MH_01_easy", "phase": "8.6A", "camera_geometry": "estimated"},
    )
    request = LiveSlamRequest(
        session_id=session_id,
        source=source,
        camera=camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id=STELLA_LIVE_BACKEND_ID,
        simulated_live=True,
        camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
        enable_mapping=True,
        enable_loop_closure=False,
        enable_global_optimization=False,
        metadata={"validation": "phase8.6a-estimated-camera-real-stella"},
    )
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=camera.image_size.width * camera.image_size.height * 3,
    )
    adapter = LiveSourceFrameAdapter(session_id=session_id, source=source, camera=camera, ring=ring)
    control = ReconstructionOperationControl()
    context = BackendOperationContext(
        session_id=session_id,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=control,
        emit_event=lambda _event: None,
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

    observations = []
    dataset_origin_ns = records[0][0]
    capture_origin_s = monotonic()
    try:
        backend.prepare(context)
        backend.start_live_session(request, context)
        for index, (timestamp_ns, path) in enumerate(records):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (480, 752):
                raise RuntimeError(f"Invalid EuRoC image: {path}")
            source_frame = SourceFrame(
                image=image,
                frame_id=index,
                timestamp=capture_origin_s + (timestamp_ns - dataset_origin_ns) / 1_000_000_000.0,
            )
            live_frame = adapter.publish(source_frame)
            observations.append(backend.track_live_frame(live_frame, context))
            if (index + 1) % 100 == 0:
                print(f"frames={index + 1} poses={sum(item.pose is not None for item in observations)}", flush=True)
        backend.stop_live_session(context)
    finally:
        backend.shutdown()
        ring.shutdown()

    pose_count = sum(item.pose is not None for item in observations)
    max_keyframes = max((item.map_snapshot.keyframe_count for item in observations if item.map_snapshot), default=0)
    max_landmarks = max((item.map_snapshot.landmark_count for item in observations if item.map_snapshot), default=0)
    latencies = [item.quality.processing_latency_ns / 1_000_000.0 for item in observations]
    arbitrary_ok = all(item.pose is None or item.pose.gauge.unit_to_metre is None for item in observations)
    min_poses = max(50, int(len(records) * 0.60))
    passed = (
        len(observations) == len(records)
        and pose_count >= min_poses
        and max_keyframes >= 1
        and max_landmarks > 0
        and arbitrary_ok
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "8.6A",
        "pass": passed,
        "camera_geometry_mode": profile.mode.value,
        "camera_geometry_method": profile.method.value,
        "horizontal_fov_degrees": profile.horizontal_fov_degrees,
        "focal_length_relative_uncertainty": profile.focal_length_relative_uncertainty,
        "intrinsics_source": camera.intrinsics.source.value,
        "calibration_status": camera.intrinsics.calibration_status.value,
        "camera_intrinsics_policy": request.camera_intrinsics_policy.value,
        "frames_submitted": len(records),
        "observations_received": len(observations),
        "pose_observations": pose_count,
        "max_keyframes": max_keyframes,
        "max_landmarks": max_landmarks,
        "median_tracking_latency_ms": statistics.median(latencies) if latencies else None,
        "real_stella_executed": True,
        "worker_process_isolated": True,
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": "arbitrary",
        "unit_to_metre": None,
        "metric_accuracy_claimed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    print(f"Phase 8.6A real stella estimated-camera smoke: {'PASS' if passed else 'FAIL'}")
    print(f"frames={len(records)} observations={len(observations)} poses={pose_count} keyframes={max_keyframes} landmarks={max_landmarks}")
    print(f"camera_mode={profile.mode.value} method={profile.method.value} hFOV={profile.horizontal_fov_degrees}")
    print(f"intrinsics_source={camera.intrinsics.source.value} calibration_status={camera.intrinsics.calibration_status.value}")
    print(f"policy={request.camera_intrinsics_policy.value} metric_accuracy_claimed=False")
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    print(f"summary={output / 'summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
