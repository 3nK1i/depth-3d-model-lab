#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"

quick() {
  python -m pytest -q \
    tests/unit/test_phase8_9_pre_h_ui3_contract.py \
    tests/ui/test_phase8_9_pre_h_ui3_reconstruction_fullscreen.py
}

affected() {
  quick
  python -m pytest -q \
    tests/ui/test_viewport_controls.py \
    tests/ui/test_media_toolbars.py
  python -m pytest -q \
    tests/ui/test_phase8_8c1_live_visualizer_foundation.py \
    tests/ui/test_phase8_8c2_live_trajectory_visualization.py \
    tests/ui/test_phase8_8c3_live_sparse_map_visualization.py \
    tests/ui/test_phase8_8c4_live_keyframe_map_structure.py \
    tests/ui/test_phase8_8c5_tracking_diagnostics_hud.py \
    tests/ui/test_phase8_8d4_offline_spatial_coverage.py \
    tests/ui/test_phase8_8d5_semantic_object_presentation.py
}

case "$mode" in
  quick) quick ;;
  affected) affected ;;
  *)
    echo "usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
