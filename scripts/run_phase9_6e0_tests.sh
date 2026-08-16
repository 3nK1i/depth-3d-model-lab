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
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_6e0_estimated_metric_measurement.py \
      tests/ui/test_phase9_6e0_estimated_metric_ruler_ui.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase9_6e0_estimated_metric_measurement.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_pre9_4e_interactive_distance.py \
      tests/ui/test_pre9_4e_interactive_measurement_ui.py \
      tests/integration/test_pre9_4e_interactive_measurement_bundle.py \
      tests/integration/test_phase9_4h_gaussian_phase_closure.py \
      tests/integration/test_phase9_5h_nerf_phase_closure.py \
      tests/unit/test_phase9_3j_measurement_registration.py \
      tests/ui/test_phase9_3e_canonical_measurement_overlay.py \
      tests/unit/test_phase9_6d_digital_twin_interaction.py \
      tests/ui/test_phase9_6d_unified_interaction_ui.py \
      tests/integration/test_phase9_6d_unified_interaction_policy.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
