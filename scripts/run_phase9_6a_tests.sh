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
      tests/unit/test_phase9_6a_digital_twin_workspace.py \
      tests/ui/test_phase9_6a_digital_twin_workspace_ui.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase9_6a_digital_twin_workspace.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase9_3a_canonical_scene_presentation.py \
      tests/ui/test_phase9_3f_scene_layers.py \
      tests/integration/test_phase9_3j_measurement_canonical_bridge.py \
      tests/unit/test_phase9_4h_gaussian_phase_closure.py \
      tests/ui/test_phase9_4h_gaussian_phase_closure_ui.py \
      tests/unit/test_phase9_5h_nerf_phase_closure.py \
      tests/ui/test_phase9_5h_nerf_phase_closure_ui.py \
      tests/integration/test_pre9_4e_interactive_measurement_bundle.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
