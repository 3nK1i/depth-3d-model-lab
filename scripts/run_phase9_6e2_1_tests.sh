#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_6e21_offline_progress_fsm.py \
      tests/integration/test_phase9_6e21_offline_progress_contract.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_6e21_offline_progress_contract.py \
      tests/integration/test_reconstruction_recorded_video_sfm_hardening.py \
      tests/integration/test_reconstruction_pycolmap_worker_lifecycle.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_reconstruction_sparse_events.py \
      tests/unit/test_reconstruction_sfm_worker_hardening.py \
      tests/unit/test_reconstruction_sfm_worker_protocol.py \
      tests/unit/test_reconstruction_sfm_pycolmap.py \
      tests/integration/test_reconstruction_recorded_video_sfm_hardening.py \
      tests/integration/test_phase8_8a3_offline_controller_process.py \
      tests/unit/test_phase9_6e2_da3_full_scene.py \
      tests/integration/test_phase9_6e2_da3_full_scene_contract.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
