#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"

run_quick() {
  python -m pytest -q \
    tests/unit/test_phase8_8b1_reconstruction_page_contract.py \
    tests/ui/test_phase8_8b1_reconstruction_page.py \
    tests/ui/test_phase8_8a1_reconstruction_controller.py \
    tests/ui/test_phase8_8a5_controller_readiness.py
}

run_backward() {
  ./scripts/run_phase8_8a5_tests.sh quick
  ./scripts/run_phase8_8a5_tests.sh integration
  python -m pytest -q \
    tests/ui/test_phase7_mode_switching.py \
    tests/ui/test_phase8_7g4_ui_readiness.py
}

case "$MODE" in
  quick|ui)
    run_quick
    ;;
  backward)
    run_backward
    ;;
  full)
    run_quick
    run_backward
    ;;
  *)
    echo "Usage: $0 {quick|ui|backward|full}" >&2
    exit 2
    ;;
esac
