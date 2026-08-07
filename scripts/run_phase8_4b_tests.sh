#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"

case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase8_4b_backward_compatibility.py \
      tests/unit/test_reconstruction_live_stella_environment.py \
      tests/unit/test_reconstruction_live_stella_protocol.py \
      tests/unit/test_reconstruction_live_stella_worker_source.py \
      tests/integration/test_reconstruction_live_stella_worker_failures.py \
      tests/integration/test_reconstruction_live_stella_worker_lifecycle.py
    ;;
  integration)
    pytest -q \
      tests/integration/test_reconstruction_live_stella_worker_failures.py \
      tests/integration/test_reconstruction_live_stella_worker_lifecycle.py
    ;;
  stress)
    pytest -q tests/stress/test_reconstruction_live_stella_worker_stress.py
    ;;
  full)
    echo "==> pytest group: tests/unit"
    pytest -q tests/unit
    echo "==> pytest group: tests/ui"
    pytest -q tests/ui
    echo "==> pytest group: tests/integration"
    pytest -q tests/integration
    echo "==> pytest group: tests/stress"
    pytest -q tests/stress
    ;;
  *)
    echo "Usage: $0 {quick|integration|stress|full}" >&2
    exit 2
    ;;
esac
