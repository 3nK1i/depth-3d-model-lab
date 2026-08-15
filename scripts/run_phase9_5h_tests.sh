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
  tests/unit/test_phase9_5h_nerf_phase_closure.py
  tests/ui/test_phase9_5h_nerf_phase_closure_ui.py
)

INTEGRATION_TESTS=(
  tests/integration/test_phase9_5h_nerf_phase_closure.py
)

STRESS_TESTS=(
  tests/stress/test_phase9_5h_nerf_phase_closure_stress.py
  tests/stress/test_phase9_5g_nerf_resource_lifecycle_stress.py
)

BACKWARD_TESTS=(
  tests/unit/test_phase9_5a_nerf_runtime_isolation.py
  tests/unit/test_phase9_5b_nerf_input_bridge.py
  tests/unit/test_phase9_5c_nerf_optimization_execution.py
  tests/unit/test_phase9_5d_nerf_persistence.py
  tests/unit/test_phase9_5e_nerf_interactive.py
  tests/unit/test_phase9_5f_nerf_display_controls.py
  tests/unit/test_phase9_5g_nerf_resource_lifecycle.py
  tests/integration/test_phase9_5a_nerf_runtime_isolation.py
  tests/integration/test_phase9_5b_nerf_input_bridge.py
  tests/integration/test_phase9_5c_nerf_optimization_execution.py
  tests/integration/test_phase9_5d_persistent_nerf_representation.py
  tests/integration/test_phase9_5e_interactive_nerf_viewer.py
  tests/integration/test_phase9_5f_nerf_controls_layers.py
  tests/integration/test_phase9_5g_nerf_resource_lifecycle.py
  tests/ui/test_phase9_5e_interactive_nerf_viewer_ui.py
  tests/ui/test_phase9_5f_nerf_controls_layers_ui.py
  tests/ui/test_phase9_5g_nerf_resource_lifecycle_ui.py
  tests/unit/test_phase8_8f3_contract.py
  tests/unit/test_pre9_4e_interactive_distance.py
  tests/unit/test_pre9_4e_projected_surface_pick.py
  tests/ui/test_pre9_4e_interactive_measurement_ui.py
  tests/integration/test_pre9_4e_interactive_measurement_bundle.py
)

case "$MODE" in
  quick)
    python -m pytest -q "${QUICK_TESTS[@]}"
    ;;
  integration)
    python -m pytest -q "${INTEGRATION_TESTS[@]}"
    python scripts/run_phase9_5h_nerf_closure_smoke.py
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
    python scripts/run_phase9_5h_nerf_closure_smoke.py
    echo "Phase 9.5H current-tree automated closure regression: PASS"
    echo "real_nerf_closure_campaign_must_be_run_separately=True"
    echo "single_view_quality_is_characterization_not_production_certification=True"
    echo "real_physical_scale_calibration_may_remain_deferred=True"
    echo "certified_metrology_remains_false=True"
    ;;
  *)
    echo "usage: $0 {quick|integration|stress|backward|full}" >&2
    exit 2
    ;;
esac
