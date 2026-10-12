#!/usr/bin/env bash
# I6.8.5C bounded contract/dispatch and existing workflow regression checks.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m py_compile \
  app/reconstruction_method_execution.py \
  app/widgets/reconstruction_page.py \
  tests/unit/test_i6_8_5c_reconstruction_method_execution.py \
  tests/ui/test_i6_8_5c_method_execution_ui.py
python -m pytest -q \
  tests/unit/test_i6_8_5c_reconstruction_method_execution.py \
  tests/ui/test_i6_8_5c_method_execution_ui.py \
  tests/unit/test_i6_8_5b_reconstruction_method_contract.py \
  tests/unit/test_i6_4a_provider_availability_selection_contract.py \
  tests/unit/test_phase9_3h_da3_application_contract.py \
  tests/unit/test_phase9_6e2_da3_full_scene.py \
  tests/ui/test_phase9_6e2_da3_full_scene_ui.py \
  tests/unit/test_i6_7e2a_sparse_dense_handoff_gate.py
printf 'I6_8_5C_FOCUSED_VALIDATION=PASS\n'
