#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_known_size_validation.py \
      tests/unit/test_phase75_release_gate.py \
      tests/unit/test_phase75_step10_backward_compatibility.py \
      tests/unit/test_inspection_export.py \
      tests/unit/test_thin_structure_preservation.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_known_size_validation_export.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|full}" >&2
    exit 2
    ;;
esac
