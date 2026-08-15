#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_pre9_4e_auto_segment_proposal_deletion.py \
      tests/ui/test_pre9_4e_auto_segment_proposal_deletion_ui.py
    ;;
  integration)
    pytest -q tests/integration/test_pre9_4e_auto_segment_proposal_deletion_persistence.py
    python scripts/run_pre9_4e_auto_segment_deletion_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_2a_semantic_workspace.py \
      tests/unit/test_phase9_2f_semantic_merge.py \
      tests/unit/test_phase9_2h_project_workspace_persistence.py \
      tests/integration/test_phase9_2a_semantic_workspace_compatibility.py \
      tests/integration/test_phase9_2b_semantic_project_workspace.py \
      tests/integration/test_phase9_2f_merge_downstream_compatibility.py \
      tests/integration/test_phase9_2h_workspace_resume_measure_inspect.py \
      tests/ui/test_phase9_2d_auto_segment_workspace.py \
      tests/ui/test_phase9_2e_prompt_refinement_ui.py \
      tests/ui/test_phase9_2f_mask_merge_ui.py \
      tests/ui/test_phase9_2h_workspace_class_management_ui.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
