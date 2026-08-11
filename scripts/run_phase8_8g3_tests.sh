#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    ./scripts/run_phase8_8g2_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8g3_contract.py \
      tests/unit/test_phase8_8g3_reconstruction_checkpoint_resume.py \
      tests/ui/test_phase8_8g3_checkpoint_resume_ux.py
    ;;
  protocol)
    ./scripts/run_phase8_8g2_tests.sh protocol
    python -m pytest -q tests/unit/test_phase8_8g3_contract.py
    ;;
  backward)
    ./scripts/run_phase8_8g2_tests.sh backward
    python -m pytest -q \
      tests/unit/test_reconstruction_engine_checkpoint_g2.py \
      tests/unit/test_phase8_7g2_backward_compatibility.py \
      tests/unit/test_phase8_8a4_reconstruction_persistence_controller.py \
      tests/unit/test_phase8_8a5_controller_hardening.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac
