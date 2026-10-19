from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
WORKER_SOURCE = ROOT / "native" / "live_slam_worker" / "main.cc"
BUILD_SCRIPT = ROOT / "scripts" / "build_stella_live_worker.sh"
REAL_SMOKE_SCRIPT = ROOT / "scripts" / "run_live_stella_worker_smoke.py"


def test_native_worker_source_enforces_process_and_phase_boundaries() -> None:
    text = WORKER_SOURCE.read_text(encoding="utf-8")
    assert "shm_open" in text
    assert "mmap" in text
    assert "AF_UNIX" in text
    assert "feed_monocular_frame" in text
    assert "disable_loop_detector" in text
    assert "loop_detector_is_enabled" in text
    assert "T_camera_from_reconstruction.inverse()" in text
    assert '"T_reconstruction_from_camera"' in text
    assert '"unit_to_metre", nullptr' in text
    assert '"phase8_global_optimization_feature_enabled", false' in text
    assert "enable_loop_detector" in text
    assert "request_loop_closure" in text
    assert "loop_BA_is_running" in text
    assert "get_pose_wc" in text
    assert '"loop_closure_supported_in_phase8_5", true' in text
    assert '"global_optimization_supported_in_phase8_5", true' in text


def test_native_worker_never_owns_usb_rtsp_or_video_capture() -> None:
    text = WORKER_SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "VideoCapture",
        "CAP_FFMPEG",
        "CAP_V4L2",
        "rtsp://",
        "/dev/video",
    )
    assert all(token not in text for token in forbidden)


def test_native_worker_build_script_is_shell_syntax_clean_and_external_build_only() -> None:
    subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "$HOME/Desktop/depth3d-stella-live-worker-build" in text
    assert "$HOME/Desktop/depth3d-stella-live-worker" in text
    assert "unset CONDA_PREFIX" in text
    assert "STELLA LIVE WORKER BUILD: PASS" in text

    smoke_text = REAL_SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "timestamp_origin=TimestampOrigin.CAPTURE_CLOCK" in smoke_text
    assert "timestamp_origin=TimestampOrigin.PRESENTATION_TIMESTAMP" not in smoke_text
    assert "capture_clock_origin_s = monotonic()" in smoke_text
    assert "timestamp=capture_timestamp_s" in smoke_text
