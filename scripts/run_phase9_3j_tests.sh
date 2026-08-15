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
  tests/unit/test_phase9_3j_measurement_registration.py
  tests/ui/test_phase9_3j_measurement_canonical_registration.py
)

INTEGRATION_TESTS=(
  tests/integration/test_phase9_3j_measurement_canonical_bridge.py
)

BACKWARD_TESTS=(
  tests/unit/test_phase9_3c_canonical_camera_presentation.py
  tests/unit/test_phase9_3b_canonical_geometry_loader.py
  tests/unit/test_phase9_3e_canonical_inspection_overlays.py
  tests/unit/test_phase9_3f_scene_layers.py
  tests/unit/test_phase9_3g_dense_provider_presentation.py
  tests/unit/test_phase9_3h_da3_application_contract.py
  tests/unit/test_phase9_3h_da3_canonical_scene.py
  tests/unit/test_phase9_3i_scene_runtime_health.py
  tests/unit/test_inspection_bundle.py
  tests/ui/test_phase9_3e_canonical_measurement_overlay.py
  tests/ui/test_phase9_3f_scene_layers.py
  tests/ui/test_phase9_3g_dense_provider_ui.py
  tests/ui/test_phase9_3h_da3_canonical_integration.py
  tests/ui/test_phase9_3i_scene_runtime_hardening.py
)

case "$MODE" in
  quick)
    python -m pytest -q "${QUICK_TESTS[@]}"
    ;;
  integration)
    python -m pytest -q "${INTEGRATION_TESTS[@]}"
    python scripts/run_phase9_3j_measurement_registration_smoke.py
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
    python scripts/run_phase9_3j_measurement_registration_smoke.py
    echo "Phase 9.3J current-tree automated closure regression: PASS"
    echo "real_da3_qualification_must_be_run_separately=True"
    echo "real_physical_calibration_optional_and_may_remain_deferred=True"
    ;;
  *)
    echo "usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac
