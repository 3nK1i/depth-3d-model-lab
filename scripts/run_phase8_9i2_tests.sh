#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

case "$MODE" in
  quick|affected|full) ;;
  *)
    echo "usage: $0 {quick|affected|full}" >&2
    exit 2
    ;;
esac

python scripts/run_phase8_9i2_release_transition.py

QUICK_TESTS=(
  tests/unit/test_phase8_9i2_release_transition_contract.py
  tests/unit/test_release_metadata.py
  tests/unit/test_phase6_release_metadata.py
  tests/unit/test_phase7_release_metadata.py
  tests/ui/test_release_header.py
  tests/ui/test_phase7_release_modes.py
)

AFFECTED_TESTS=(
  "${QUICK_TESTS[@]}"
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
  tests/unit/test_phase8_8d5_contract.py
  tests/unit/test_phase8_9g1_regression_inventory.py
  tests/unit/test_phase8_9g2_full_regression.py
  tests/unit/test_phase8_9g3_backward_compatibility.py
  tests/unit/test_phase8_9h2_release_prep_contract.py
  tests/unit/test_phase8_9h3_release_candidate_contract.py
  tests/integration/test_phase8_9g1_regression_contract.py
  tests/integration/test_phase8_9g2_full_regression_contract.py
)

if [[ "$MODE" == "quick" ]]; then
  python -m pytest -q "${QUICK_TESTS[@]}"
  python scripts/run_phase8_9i2_release_transition.py
  echo "Phase 8.9I2 v0.8.0 release transition: PASS (quick)"
  exit 0
fi

if [[ "$MODE" == "affected" ]]; then
  python -m pytest -q "${AFFECTED_TESTS[@]}"
  python scripts/run_phase8_9i2_release_transition.py
  echo "Phase 8.9I2 v0.8.0 release transition: PASS (affected)"
  exit 0
fi

python -m compileall -q app core tests scripts main.py
python -m pytest -q tests/unit
python -m pytest -q tests/ui
python -m pytest -q tests/integration
python -m pytest -q tests/stress
python scripts/run_phase8_9i2_release_transition.py

echo "Phase 8.9I2 v0.8.0 final current-tree release gate: PASS (full)"
echo "heavy_real_sfm_gaussian_nerf_campaigns_reexecuted=False"
echo "v0.8.0_tag_created=False"
