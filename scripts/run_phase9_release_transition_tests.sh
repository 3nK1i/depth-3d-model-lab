#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"

python -m compileall -q core/version.py core/runtime/release.py main.py tests/unit/test_phase9_release_transition_contract.py

quick_tests=(
  tests/unit/test_phase9_release_transition_contract.py
  tests/unit/test_release_metadata.py
  tests/ui/test_release_header.py
)

affected_tests=(
  tests/unit/test_phase6_release_metadata.py
  tests/unit/test_phase7_release_metadata.py
  tests/unit/test_phase75_step1_backward_compatibility.py
  tests/unit/test_phase75_step2_backward_compatibility.py
  tests/unit/test_phase75_step3_backward_compatibility.py
  tests/unit/test_phase75_step4_backward_compatibility.py
  tests/unit/test_phase75_step5_backward_compatibility.py
  tests/unit/test_phase75_step6_backward_compatibility.py
  tests/unit/test_phase75_step7_backward_compatibility.py
  tests/unit/test_phase75_step8_backward_compatibility.py
  tests/unit/test_phase75_step10_backward_compatibility.py
  tests/unit/test_phase8_1_backward_compatibility.py
  tests/unit/test_phase8_2_backward_compatibility.py
  tests/unit/test_phase8_3b_backward_compatibility.py
  tests/unit/test_phase8_3c_backward_compatibility.py
  tests/unit/test_phase8_3d_backward_compatibility.py
  tests/unit/test_phase8_4a_backward_compatibility.py
  tests/unit/test_phase8_7g4_readiness.py
  tests/unit/test_phase8_9g1_regression_inventory.py
  tests/unit/test_phase8_9h2_release_prep_contract.py
  tests/unit/test_phase8_9h3_release_candidate_contract.py
  tests/unit/test_phase8_9i2_release_transition_contract.py
  tests/ui/test_phase7_release_modes.py
  tests/ui/test_release_header.py
  tests/integration/test_phase8_9g1_regression_contract.py
  tests/integration/test_phase8_9g2_full_regression_contract.py
  tests/integration/test_phase9_1i_phase_closure.py
  tests/integration/test_phase9_2i_phase_closure.py
)

case "$MODE" in
  quick)
    python -m pytest -q "${quick_tests[@]}"
    echo "Phase 9 v0.9.0 release transition: PASS (quick)"
    ;;
  affected)
    python -m pytest -q "${quick_tests[@]}" "${affected_tests[@]}"
    echo "Phase 9 v0.9.0 release transition: PASS (affected)"
    ;;
  full)
    python -m pytest -q tests/unit
    python -m pytest -q tests/ui
    python -m pytest -q tests/integration
    python -m pytest -q tests/stress
    echo "Phase 9 v0.9.0 release transition: PASS (full)"
    echo "heavy_real_backend_campaign_reexecuted=False"
    echo "certified_metrology=False"
    ;;
  *)
    echo "Usage: $0 {quick|affected|full}" >&2
    exit 2
    ;;
esac
