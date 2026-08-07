#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_reconstruction_live_contracts.py \
      tests/unit/test_reconstruction_live_transport.py \
      tests/unit/test_reconstruction_live_frames.py \
      tests/unit/test_reconstruction_live_mailbox.py \
      tests/unit/test_reconstruction_live_protocol.py \
      tests/unit/test_reconstruction_live_process.py \
      tests/unit/test_reconstruction_live_mock.py \
      tests/unit/test_phase8_4a_backward_compatibility.py \
      tests/unit/test_reconstruction_profiles_backends.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_reconstruction_live_foundation_lifecycle.py \
      tests/integration/test_reconstruction_contract_lifecycle.py \
      tests/integration/test_reconstruction_ingestion_lifecycle.py
    ;;
  stress)
    python -m pytest -q \
      tests/stress/test_reconstruction_live_transport_stress.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|stress|full}" >&2
    exit 2
    ;;
esac
