#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

python -m compileall -q \
  core/capture/rtsp_source.py \
  app/live_slam_controller.py \
  tests/unit/test_phase8_9_pre_h_rtsp_fps_fix.py \
  tests/integration/test_phase8_9_pre_h_rtsp_fps_pipeline.py

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase8_9_pre_h_rtsp_fps_fix.py \
      tests/integration/test_phase8_9_pre_h_rtsp_fps_pipeline.py
    ;;
  affected)
    "$0" quick
    python -m pytest -q \
      tests/unit/test_capture_sources.py \
      tests/unit/test_capture_live_source_identity.py \
      tests/unit/test_reconstruction_live_source_metadata.py \
      tests/unit/test_reconstruction_live_stella_environment.py \
      tests/unit/test_phase8_8a2_reconstruction_controller.py \
      tests/integration/test_phase8_8a2_live_controller_pipeline.py
    ./scripts/run_phase8_4c_tests.sh quick
    ./scripts/run_phase8_4d_tests.sh quick
    ./scripts/run_phase8_6a_tests.sh quick
    ;;
  *)
    echo "Usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
