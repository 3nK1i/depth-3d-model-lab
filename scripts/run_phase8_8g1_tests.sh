#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    ./scripts/run_phase8_8f5_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8g1_contract.py \
      tests/unit/test_phase8_8g1_reconstruction_session_model.py \
      tests/ui/test_phase8_8g1_reconstruction_session_model.py
    ;;
  protocol)
    ./scripts/run_phase8_8f5_tests.sh protocol
    python -m pytest -q tests/unit/test_phase8_8g1_contract.py
    ;;
  backward)
    ./scripts/run_phase8_8f5_tests.sh backward
    python -m pytest -q \
      tests/unit/test_reconstruction_contracts_session.py \
      tests/unit/test_reconstruction_checkpoint_serialization.py \
      tests/unit/test_reconstruction_engine_checkpoint_g2.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac
