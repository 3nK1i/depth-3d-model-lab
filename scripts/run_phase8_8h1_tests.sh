#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    ./scripts/run_phase8_8g6_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8h1_contract.py \
      tests/integration/test_phase8_8h1_whole_workflow_integration.py \
      tests/ui/test_phase8_8h1_whole_reconstruction_workflow.py
    ;;
  protocol)
    ./scripts/run_phase8_8g6_tests.sh protocol
    python -m pytest -q \
      tests/unit/test_phase8_8h1_contract.py \
      tests/integration/test_phase8_8h1_whole_workflow_integration.py
    ;;
  backward)
    ./scripts/run_phase8_8g6_tests.sh backward
    python -m pytest -q \
      tests/ui/test_phase8_8g6_full_reconstruction_ux_review.py \
      tests/ui/test_phase8_8h1_whole_reconstruction_workflow.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac
