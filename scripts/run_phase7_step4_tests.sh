#!/usr/bin/env bash
# Phase 7 release test gate. Test groups run in separate Python processes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-full}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_release_metadata.py \
      tests/unit/test_phase6_release_metadata.py \
      tests/unit/test_phase7_release_metadata.py \
      tests/unit/test_phase7_operational_exports.py \
      tests/unit/test_release_check_runner.py
    pytest -q tests/ui/test_phase7_mode_switching.py
    pytest -q tests/stress/test_phase7_release_artifacts_stress.py
    ;;
  full)
    printf '\n==================================================\nUnit tests\n==================================================\n'
    pytest -q tests/unit
    printf '\n==================================================\nUI tests\n==================================================\n'
    pytest -q tests/ui
    printf '\n==================================================\nIntegration tests\n==================================================\n'
    pytest -q tests/integration
    printf '\n==================================================\nStress tests\n==================================================\n'
    pytest -q tests/stress
    ;;
  *)
    echo "Usage: $0 {quick|full}" >&2
    exit 2
    ;;
esac
