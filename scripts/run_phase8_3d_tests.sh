#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_reconstruction_sfm_quality.py \
      tests/unit/test_reconstruction_sfm_worker_hardening.py \
      tests/unit/test_reconstruction_sfm_worker_protocol.py \
      tests/unit/test_reconstruction_sfm_pycolmap.py \
      tests/unit/test_phase8_3d_backward_compatibility.py \
      tests/unit/test_phase8_3c_backward_compatibility.py \
      tests/unit/test_reconstruction_sfm_contracts.py \
      tests/unit/test_reconstruction_sfm_coordinates.py \
      tests/unit/test_reconstruction_sfm_workspace.py \
      tests/unit/test_reconstruction_sfm_environment.py \
      tests/unit/test_reconstruction_sfm_mock.py \
      tests/unit/test_phase8_3b_backward_compatibility.py \
      tests/unit/test_reconstruction_ingestion_pipeline.py \
      tests/unit/test_phase8_2_backward_compatibility.py \
      tests/unit/test_reconstruction_camera_poses.py \
      tests/unit/test_phase8_1_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_reconstruction_recorded_video_sfm_hardening.py \
      tests/integration/test_reconstruction_pycolmap_worker_lifecycle.py \
      tests/integration/test_reconstruction_sfm_foundation_lifecycle.py \
      tests/integration/test_reconstruction_ingestion_lifecycle.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|full}" >&2
    exit 2
    ;;
esac
