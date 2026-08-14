#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/ui/test_phase9_2d_auto_segment_workspace.py \
      tests/ui/test_phase9_2d_measure_inspect_entry.py \
      tests/unit/test_phase9_2c_auto_segment_engine.py \
      tests/unit/test_phase9_2b_semantic_project.py \
      tests/unit/test_phase9_2a_semantic_workspace.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_2d_auto_segment_job.py \
      tests/integration/test_phase9_2c_auto_segment_project_flow.py \
      tests/integration/test_phase9_2b_semantic_project_workspace.py \
      tests/integration/test_phase9_2a_semantic_workspace_compatibility.py
    ;;
  backward)
    python -m pytest -q \
      tests/ui/test_phase9_1g_measurement_calibration_ui.py \
      tests/ui/test_phase9_1h_persisted_authority_ui.py \
      tests/unit/test_phase8_8b3_runtime_contract.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
