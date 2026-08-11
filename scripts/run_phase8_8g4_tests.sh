#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    ./scripts/run_phase8_8g3_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8g4_contract.py \
      tests/unit/test_phase8_8g4_reconstruction_failure_recovery.py \
      tests/ui/test_phase8_8g4_failure_recovery_ux.py
    ;;
  protocol)
    ./scripts/run_phase8_8g3_tests.sh protocol
    python -m pytest -q tests/unit/test_phase8_8g4_contract.py
    ;;
  backward)
    ./scripts/run_phase8_8g3_tests.sh backward
    python -m pytest -q \
      tests/ui/test_phase8_8g2_session_save_reload_interaction.py \
      tests/ui/test_phase8_8g3_checkpoint_resume_ux.py \
      tests/unit/test_phase8_8g3_reconstruction_checkpoint_resume.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac
