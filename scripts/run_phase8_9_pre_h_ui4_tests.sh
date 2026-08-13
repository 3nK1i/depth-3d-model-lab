#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"

quick() {
  python -m compileall -q \
    tests/ui/test_phase8_9_pre_h_ui4_combined_ui_qualification.py \
    tests/unit/test_phase8_9_pre_h_ui4_contract.py
  python -m pytest -q \
    tests/unit/test_phase8_9_pre_h_ui4_contract.py \
    tests/ui/test_phase8_9_pre_h_ui4_combined_ui_qualification.py
}

affected() {
  quick

  # Re-run the complete already-qualified UI-2 and UI-3 affected surfaces as
  # one composition gate. This is deliberately broader than either substep's
  # quick gate and intentionally contains some overlap.
  ./scripts/run_phase8_9_pre_h_ui2_tests.sh affected
  ./scripts/run_phase8_9_pre_h_ui3_tests.sh affected

  # Small backward-compatibility / lifecycle sanity around the shared shell.
  python -m pytest -q \
    tests/ui/test_main_window_shell.py \
    tests/ui/test_phase7_mode_switching.py \
    tests/ui/test_shutdown_lifecycle.py \
    tests/unit/test_phase8_9_pre_h_rtsp_fps_fix.py
}

case "$mode" in
  quick) quick ;;
  affected) affected ;;
  *)
    echo "usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
