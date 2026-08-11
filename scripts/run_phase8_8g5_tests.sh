#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    ./scripts/run_phase8_8g4_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8g5_contract.py \
      tests/ui/test_phase8_8g5_progressive_disclosure_ux.py
    ;;
  protocol)
    ./scripts/run_phase8_8g4_tests.sh protocol
    python -m pytest -q tests/unit/test_phase8_8g5_contract.py
    ;;
  backward)
    ./scripts/run_phase8_8g4_tests.sh backward
    # G5 changes the shared settings-group container, so re-run the bounded UI
    # owners that populate those groups rather than repeating the whole app suite.
    python -m pytest -q \
      tests/ui/test_phase8_8e1_scene_object_selection.py \
      tests/ui/test_phase8_8e2_reconstruction_measurement_ui.py \
      tests/ui/test_phase8_8e3_measurement_validity_confidence.py \
      tests/ui/test_phase8_8e4_inspection_qc_presentation.py \
      tests/ui/test_phase8_8e5_reference_comparison_workflow.py \
      tests/ui/test_phase8_8f1_appearance_representation_controls.py \
      tests/ui/test_phase8_8f2_gaussian_splatting_visualization.py \
      tests/ui/test_phase8_8g1_reconstruction_session_model.py \
      tests/ui/test_phase8_8g2_session_save_reload_interaction.py \
      tests/ui/test_phase8_8g3_checkpoint_resume_ux.py \
      tests/ui/test_phase8_8g4_failure_recovery_ux.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac
