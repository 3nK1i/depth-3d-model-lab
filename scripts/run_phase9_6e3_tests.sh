#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q tests/unit/test_phase9_6e3_nerf_build.py tests/ui/test_phase9_6e3_nerf_build_ui.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase9_6e3_nerf_build_contract.py tests/integration/test_phase9_5d_persistent_nerf_representation.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase9_5b_nerf_input_bridge.py \
      tests/unit/test_phase9_5c_nerf_optimization_execution.py \
      tests/unit/test_phase9_5d_nerf_persistence.py \
      tests/unit/test_phase9_5e_nerf_interactive.py \
      tests/unit/test_phase9_5f_nerf_display_controls.py \
      tests/unit/test_phase9_5g_nerf_resource_lifecycle.py \
      tests/unit/test_phase9_6e1_processing_jobs.py \
      tests/unit/test_phase9_6e2_da3_full_scene.py
    ;;
  full)
    "$0" quick
    "$0" integration
    "$0" backward
    ;;
  *) echo "usage: $0 {quick|integration|backward|full}" >&2; exit 2 ;;
esac
