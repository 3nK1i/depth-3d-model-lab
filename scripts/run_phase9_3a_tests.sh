#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

compile_3a() {
  python -m compileall -q \
    app/reconstruction_canonical_scene_presentation.py \
    app/widgets/reconstruction_page.py \
    scripts/run_phase9_3a_canonical_scene_smoke.py \
    scripts/run_phase9_3a_canonical_scene_preview.py \
    tests/unit/test_phase9_3a_canonical_scene_presentation.py \
    tests/ui/test_phase9_3a_canonical_scene_presentation.py \
    tests/integration/test_phase9_3a_canonical_scene_compatibility.py
  bash -n scripts/run_phase9_3a_tests.sh
}

run_backward_sanity() {
  python -m pytest -q \
    tests/unit/test_phase8_8c1_visualizer_contract.py \
    tests/ui/test_phase8_8c1_live_visualizer_foundation.py \
    tests/ui/test_phase9_1h_persisted_authority_ui.py \
    tests/integration/test_phase8_9f5_persistent_real_scene_contract.py \
    tests/integration/test_phase9_1a_calibration_compatibility.py \
    tests/integration/test_phase9_2a_semantic_workspace_compatibility.py
}

case "$MODE" in
  quick)
    compile_3a
    python -m pytest -q \
      tests/unit/test_phase9_3a_canonical_scene_presentation.py \
      tests/ui/test_phase9_3a_canonical_scene_presentation.py
    ;;
  integration)
    compile_3a
    python -m pytest -q tests/integration/test_phase9_3a_canonical_scene_compatibility.py
    python scripts/run_phase9_3a_canonical_scene_smoke.py
    ;;
  backward)
    compile_3a
    run_backward_sanity
    ;;
  full)
    compile_3a
    python -m pytest -q \
      tests/unit/test_phase9_3a_canonical_scene_presentation.py \
      tests/ui/test_phase9_3a_canonical_scene_presentation.py \
      tests/integration/test_phase9_3a_canonical_scene_compatibility.py
    run_backward_sanity
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac
