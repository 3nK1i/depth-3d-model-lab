#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    ./scripts/run_phase8_8h4_tests.sh quick
    python -m compileall -q \
      scripts/run_phase8_8h5_real_ui_acceptance.py \
      tests/unit/test_phase8_8h5_contract.py
    bash -n scripts/run_phase8_8h5_tests.sh
    python -m pytest -q tests/unit/test_phase8_8h5_contract.py
    ;;
  supporting)
    "$0" quick
    python -m pytest -q \
      tests/ui/test_phase8_8g6_full_reconstruction_ux_review.py \
      tests/ui/test_phase8_8h1_whole_reconstruction_workflow.py \
      tests/ui/test_phase8_8h3_failure_recovery_integration.py
    python scripts/run_phase8_8h1_whole_workflow_smoke.py
    python scripts/run_phase8_8h3_failure_recovery_smoke.py
    ;;
  *)
    echo "usage: $0 {quick|supporting}" >&2
    exit 2
    ;;
esac
