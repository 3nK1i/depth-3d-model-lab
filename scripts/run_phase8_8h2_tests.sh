#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    ./scripts/run_phase8_8h1_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8h2_contract.py \
      tests/integration/test_phase8_8h2_resource_coordination.py \
      tests/ui/test_phase8_8h2_application_resource_coordination.py
    ;;
  protocol)
    ./scripts/run_phase8_8h1_tests.sh protocol
    python -m pytest -q \
      tests/unit/test_phase8_8h2_contract.py \
      tests/integration/test_phase8_8h2_resource_coordination.py
    ;;
  backward)
    ./scripts/run_phase8_8h1_tests.sh backward
    python -m pytest -q \
      tests/ui/test_phase8_8a5_controller_readiness.py \
      tests/ui/test_phase8_8h1_whole_reconstruction_workflow.py \
      tests/ui/test_phase8_8h2_application_resource_coordination.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac
