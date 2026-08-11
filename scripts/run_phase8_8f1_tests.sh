#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"

run_quick() {
  ./scripts/run_phase8_8e5_tests.sh quick
  python -m pytest -q \
    tests/unit/test_phase8_8f1_reconstruction_appearance_presentation.py \
    tests/unit/test_phase8_8f1_contract.py \
    tests/ui/test_phase8_8f1_appearance_representation_controls.py
}

run_protocol() {
  ./scripts/run_phase8_8e5_tests.sh protocol
  # F1 is metadata/application control only. It must not launch or import
  # Gaussian/NeRF runtimes, render, or change appearance-only authority.
  python -m pytest -q tests/unit/test_phase8_8f1_contract.py
}

run_backward() {
  ./scripts/run_phase8_8e5_tests.sh backward
  # Small backward sanity for the Phase 8.7F5 persistent scene contract that
  # F1 consumes. Heavy backends remain mocked/isolated.
  python -m pytest -q \
    tests/unit/test_reconstruction_persistent_scene_f5.py \
    tests/integration/test_reconstruction_persistent_scene_f5.py
}

case "$MODE" in
  quick|ui)
    run_quick
    ;;
  protocol)
    run_protocol
    ;;
  backward)
    run_backward
    ;;
  full)
    run_quick
    run_protocol
    run_backward
    ;;
  *)
    echo "Usage: $0 {quick|ui|protocol|backward|full}" >&2
    exit 2
    ;;
esac
