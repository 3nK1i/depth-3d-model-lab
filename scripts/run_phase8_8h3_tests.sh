#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    ./scripts/run_phase8_8h2_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8h3_contract.py \
      tests/unit/test_phase8_8h3_live_failure_cleanup.py \
      tests/integration/test_phase8_8h3_failure_cancellation_recovery.py \
      tests/ui/test_phase8_8h3_failure_recovery_integration.py
    ;;
  protocol)
    ./scripts/run_phase8_8h2_tests.sh protocol
    python -m pytest -q \
      tests/unit/test_phase8_8h3_contract.py \
      tests/unit/test_phase8_8h3_live_failure_cleanup.py \
      tests/integration/test_phase8_8h3_failure_cancellation_recovery.py
    python -m pytest -q \
      tests/unit/test_reconstruction_live_hardening.py \
      tests/stress/test_reconstruction_live_hardening_stress.py
    ;;
  backward)
    ./scripts/run_phase8_8h2_tests.sh backward
    python -m pytest -q \
      tests/ui/test_phase8_8g4_failure_recovery_ux.py \
      tests/ui/test_phase8_8h3_failure_recovery_integration.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac
