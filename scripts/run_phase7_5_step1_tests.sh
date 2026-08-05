#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
PYTHON_BIN="${PYTHON:-python}"

STEP1_TESTS=(
  tests/unit/test_common_contracts.py
  tests/unit/test_spatial_contracts.py
  tests/unit/test_perception_metadata.py
  tests/unit/test_perception_registry.py
  tests/unit/test_perception_contracts.py
  tests/unit/test_measurement_contracts.py
  tests/unit/test_measurement_export_schema.py
  tests/unit/test_tracking_contracts.py
  tests/unit/test_phase75_step1_backward_compatibility.py
)

compile_gate() {
  "$PYTHON_BIN" -m compileall -q app core scripts tests
}

quick_gate() {
  compile_gate
  "$PYTHON_BIN" -m pytest -q "${STEP1_TESTS[@]}"
}

full_gate() {
  local status=0
  compile_gate || status=1
  # Each test group runs in its own Python/pytest process. This avoids sharing
  # PyQt6/Qt native state across the complete suite. All groups are attempted
  # so one collection failure cannot hide the status of later groups.
  for group in tests/unit tests/ui tests/integration tests/stress; do
    echo "==> pytest group: $group"
    "$PYTHON_BIN" -m pytest -q "$group" || status=1
  done
  return "$status"
}

case "$MODE" in
  quick)
    quick_gate
    ;;
  full)
    full_gate
    ;;
  *)
    echo "Usage: $0 {quick|full}" >&2
    exit 2
    ;;
esac
