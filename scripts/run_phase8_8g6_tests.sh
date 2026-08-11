#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    ./scripts/run_phase8_8g5_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8g6_contract.py \
      tests/ui/test_phase8_8g6_full_reconstruction_ux_review.py
    ;;
  protocol)
    ./scripts/run_phase8_8g5_tests.sh protocol
    python -m pytest -q tests/unit/test_phase8_8g6_contract.py
    ;;
  backward)
    ./scripts/run_phase8_8g5_tests.sh backward
    # G6 is the final Reconstruction UX review. Re-run the earlier B/C/D UI
    # owners plus G5 itself; G5 backward already covers E/F/G1-G4 owners.
    python -m pytest -q \
      tests/ui/test_phase8_8b1_reconstruction_page.py \
      tests/ui/test_phase8_8b2_reconstruction_source_controls.py \
      tests/ui/test_phase8_8b3_reconstruction_runtime_controls.py \
      tests/ui/test_phase8_8b4_status_progress.py \
      tests/ui/test_phase8_8b5_reconstruction_navigation.py \
      tests/ui/test_phase8_8c1_live_visualizer_foundation.py \
      tests/ui/test_phase8_8c2_live_trajectory_visualization.py \
      tests/ui/test_phase8_8c3_live_sparse_map_visualization.py \
      tests/ui/test_phase8_8c4_live_keyframe_map_structure.py \
      tests/ui/test_phase8_8c5_tracking_diagnostics_hud.py \
      tests/ui/test_phase8_8d1_offline_reconstruction_visualization.py \
      tests/ui/test_phase8_8d2_offline_camera_inspection.py \
      tests/ui/test_phase8_8d3_offline_sfm_health.py \
      tests/ui/test_phase8_8d4_offline_spatial_coverage.py \
      tests/ui/test_phase8_8d5_semantic_object_presentation.py \
      tests/ui/test_phase8_8g5_progressive_disclosure_ux.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac
