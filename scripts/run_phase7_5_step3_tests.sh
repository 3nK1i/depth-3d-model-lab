#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
PYTHON_BIN="${PYTHON:-python}"

STEP3_TESTS=(
  tests/unit/test_alignment_contracts.py
  tests/unit/test_alignment_transforms.py
  tests/unit/test_alignment_resampling.py
  tests/unit/test_alignment_intrinsics.py
  tests/unit/test_alignment_pipeline.py
  tests/unit/test_alignment_validation.py
  tests/unit/test_alignment_factories.py
  tests/unit/test_phase75_step3_backward_compatibility.py
)

compile() {
  "$PYTHON_BIN" -m compileall -q app core scripts tests
}

quick() {
  compile
  "$PYTHON_BIN" -m pytest -q "${STEP3_TESTS[@]}"
}

integration() {
  compile
  "$PYTHON_BIN" -m pytest -q tests/integration/test_depth_perception_alignment.py
}

full() {
  local status=0
  compile || status=1
  for group in tests/unit tests/ui tests/integration tests/stress; do
    echo "==> pytest group: $group"
    "$PYTHON_BIN" -m pytest -q "$group" || status=1
  done
  return "$status"
}

case "$MODE" in
  quick) quick ;;
  integration) integration ;;
  full) full ;;
  *)
    echo "Usage: $0 {quick|integration|full}" >&2
    exit 2
    ;;
esac
