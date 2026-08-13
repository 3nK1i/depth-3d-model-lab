#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"

quick() {
  python -m pytest -q \
    tests/unit/test_phase8_9_pre_h_ui2_visual_contract.py \
    tests/ui/test_phase8_9_pre_h_ui2_responsiveness.py
}

affected() {
  quick
  python -m pytest -q \
    tests/ui/test_viewport_layout.py \
    tests/ui/test_viewport_controls.py \
    tests/ui/test_snapshot_page.py \
    tests/ui/test_media_toolbars.py \
    tests/ui/test_stable_video_page.py
  python -m pytest -q \
    tests/unit/test_phase8_8c1_reconstruction_visualization.py \
    tests/unit/test_phase8_8c2_live_trajectory_visualization.py \
    tests/ui/test_phase8_8c1_live_visualizer_foundation.py \
    tests/ui/test_phase8_8c2_live_trajectory_visualization.py \
    tests/ui/test_phase8_8c4_live_keyframe_map_structure.py
}

case "$mode" in
  quick) quick ;;
  affected) affected ;;
  *)
    echo "usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
