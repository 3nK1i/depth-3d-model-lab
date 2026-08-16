#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_6e4_workspace_ux_contract.py \
      tests/ui/test_phase9_6e4_workspace_ux.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase9_6e4_workspace_ux_contract.py
    ;;
  backward)
    python -m pytest -q \
      tests/ui/test_phase8_8g5_progressive_disclosure_ux.py \
      tests/ui/test_phase8_8g6_full_reconstruction_ux_review.py \
      tests/ui/test_phase9_6e1_processing_job_ui.py \
      tests/ui/test_phase9_6e2_da3_full_scene_ui.py \
      tests/ui/test_phase9_6e3_nerf_build_ui.py \
      tests/unit/test_phase9_6e1_processing_jobs.py \
      tests/unit/test_phase9_6e2_da3_full_scene.py \
      tests/unit/test_phase9_6e3_nerf_build.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
