#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

quick() {
  python -m compileall -q \
    scripts/run_phase8_9_pre_h_final_requalification.py \
    tests/unit/test_phase8_9_pre_h_final_requalification_contract.py
  bash -n scripts/run_phase8_9_pre_h_final_requalification_tests.sh
  python -m pytest -q tests/unit/test_phase8_9_pre_h_final_requalification_contract.py
  python scripts/run_phase8_9_pre_h_final_requalification.py --plan-only
}

full() {
  quick
  rm -rf /tmp/eia_pre_h_final_requal
  python scripts/run_phase8_9_pre_h_final_requalification.py \
    --output-dir /tmp/eia_pre_h_final_requal
}

case "$MODE" in
  quick) quick ;;
  full) full ;;
  *)
    echo "Usage: $0 {quick|full}" >&2
    exit 2
    ;;
esac
