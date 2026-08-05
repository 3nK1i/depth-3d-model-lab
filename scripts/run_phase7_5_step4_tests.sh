#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
PYTHON_BIN="${PYTHON:-python}"

STEP4_TESTS=(
  tests/unit/test_object_geometry_contracts.py
  tests/unit/test_object_geometry_masks.py
  tests/unit/test_object_geometry_projection.py
  tests/unit/test_object_geometry_filters.py
  tests/unit/test_object_geometry_clustering.py
  tests/unit/test_object_geometry_sampling.py
  tests/unit/test_object_geometry_pipeline.py
  tests/unit/test_object_geometry_validation.py
  tests/unit/test_object_geometry_factories.py
  tests/unit/test_phase75_step4_backward_compatibility.py
)

compile() {
  "$PYTHON_BIN" -m compileall -q app core scripts tests
}

quick() {
  compile
  "$PYTHON_BIN" -m pytest -q "${STEP4_TESTS[@]}"
}

integration() {
  compile
  "$PYTHON_BIN" -m pytest -q tests/integration/test_alignment_object_point_extraction.py
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
