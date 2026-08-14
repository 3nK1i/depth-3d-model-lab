#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_2h_project_workspace_persistence.py \
      tests/ui/test_phase9_2h_workspace_class_management_ui.py \
      tests/unit/test_phase9_2g_reviewed_semantic_analysis.py \
      tests/ui/test_phase9_2g_reviewed_measure_inspect_ui.py \
      tests/unit/test_phase9_2f_semantic_merge.py \
      tests/ui/test_phase9_2f_mask_merge_ui.py \
      tests/unit/test_phase9_2e_interactive_prompt_contracts.py \
      tests/ui/test_phase9_2e_prompt_refinement_ui.py \
      tests/ui/test_phase9_2d_auto_segment_workspace.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_2h_workspace_resume_measure_inspect.py \
      tests/integration/test_phase9_2g_reviewed_measurement_pipeline.py \
      tests/integration/test_phase9_2f_merge_downstream_compatibility.py \
      tests/integration/test_phase9_2e_prompted_segment_job.py \
      tests/integration/test_phase9_2d_auto_segment_job.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase9_2b_semantic_project.py \
      tests/unit/test_phase9_2a_semantic_workspace.py \
      tests/unit/test_perception_pipeline.py \
      tests/unit/test_ultralytics_perception_adapters.py \
      tests/ui/test_phase9_1g_measurement_calibration_ui.py \
      tests/unit/test_phase8_8b3_runtime_contract.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
