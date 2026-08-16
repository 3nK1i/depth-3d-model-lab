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
      tests/unit/test_phase9_6d_digital_twin_interaction.py \
      tests/ui/test_phase9_6d_unified_interaction_ui.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase9_6d_unified_interaction_policy.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase9_6b_digital_twin_scene_composition.py \
      tests/ui/test_phase9_6b_unified_scene_composition_ui.py \
      tests/integration/test_phase9_6b_unified_scene_composition.py \
      tests/unit/test_phase9_3i_scene_runtime_health.py \
      tests/ui/test_phase9_3i_scene_runtime_hardening.py \
      tests/integration/test_phase9_3i_scene_runtime_health.py \
      tests/unit/test_phase9_4f_gaussian_display_controls.py \
      tests/ui/test_phase9_4e_interactive_gaussian_viewer_ui.py \
      tests/ui/test_phase9_4f_gaussian_controls_layers_ui.py \
      tests/unit/test_phase9_5e_nerf_interactive.py \
      tests/ui/test_phase9_5e_interactive_nerf_viewer_ui.py \
      tests/unit/test_phase9_5f_nerf_display_controls.py \
      tests/ui/test_phase9_5f_nerf_controls_layers_ui.py \
      tests/integration/test_pre9_4e_interactive_measurement_bundle.py \
      tests/unit/test_phase9_3d_canonical_semantic_presentation.py \
      tests/ui/test_phase9_3d_semantic_canonical_viewer.py \
      tests/unit/test_phase9_3e_canonical_inspection_overlays.py \
      tests/ui/test_phase9_3e_canonical_measurement_overlay.py \
      tests/unit/test_phase9_3j_measurement_registration.py \
      tests/ui/test_pre9_4e_interactive_measurement_ui.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
