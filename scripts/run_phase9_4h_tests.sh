#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

QUICK_TESTS=(
  tests/unit/test_phase9_4h_gaussian_phase_closure.py
  tests/ui/test_phase9_4h_gaussian_phase_closure_ui.py
)

INTEGRATION_TESTS=(
  tests/integration/test_phase9_4h_gaussian_phase_closure.py
)

STRESS_TESTS=(
  tests/stress/test_phase9_4h_gaussian_phase_closure_stress.py
  tests/stress/test_phase9_4g_gaussian_resource_lifecycle_stress.py
)

BACKWARD_TESTS=(
  tests/unit/test_phase9_4a_gaussian_runtime_isolation.py
  tests/unit/test_phase9_4b_gaussian_input_bridge.py
  tests/unit/test_phase9_4c_gaussian_optimization_execution.py
  tests/unit/test_phase9_4d_gaussian_persistence.py
  tests/unit/test_phase9_4e_interactive_gaussian_scene.py
  tests/unit/test_phase9_4f_gaussian_display_controls.py
  tests/unit/test_phase9_4g_gaussian_resource_lifecycle.py
  tests/unit/test_pre9_4e_interactive_distance.py
  tests/unit/test_pre9_4e_projected_surface_pick.py
  tests/ui/test_phase9_4e_interactive_gaussian_viewer_ui.py
  tests/ui/test_phase9_4f_gaussian_controls_layers_ui.py
  tests/ui/test_phase9_4g_gaussian_resource_lifecycle_ui.py
  tests/ui/test_pre9_4e_interactive_measurement_ui.py
  tests/integration/test_phase9_4a_gaussian_runtime_isolation.py
  tests/integration/test_phase9_4b_gaussian_input_bridge.py
  tests/integration/test_phase9_4c_gaussian_optimization_execution.py
  tests/integration/test_phase9_4d_persistent_gaussian_representation.py
  tests/integration/test_phase9_4e_interactive_gaussian_viewer.py
  tests/integration/test_phase9_4f_gaussian_controls_layers.py
  tests/integration/test_phase9_4g_gaussian_resource_lifecycle.py
  tests/integration/test_pre9_4e_interactive_measurement_bundle.py
)

case "$MODE" in
  quick)
    python -m pytest -q "${QUICK_TESTS[@]}"
    ;;
  integration)
    python -m pytest -q "${INTEGRATION_TESTS[@]}"
    python scripts/run_phase9_4h_gaussian_closure_smoke.py
    ;;
  stress)
    python -m pytest -q "${STRESS_TESTS[@]}"
    ;;
  backward)
    python -m pytest -q "${BACKWARD_TESTS[@]}"
    ;;
  full)
    python -m compileall -q app core tests scripts main.py
    python -m pytest -q tests/unit
    python -m pytest -q tests/ui
    python -m pytest -q tests/integration
    python -m pytest -q tests/stress
    python scripts/run_phase9_4h_gaussian_closure_smoke.py
    echo "Phase 9.4H current-tree automated closure regression: PASS"
    echo "real_gaussian_closure_campaign_must_be_run_separately=True"
    echo "real_physical_scale_calibration_may_remain_deferred=True"
    echo "certified_metrology_remains_false=True"
    ;;
  *)
    echo "usage: $0 {quick|integration|stress|backward|full}" >&2
    exit 2
    ;;
esac
