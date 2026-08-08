#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_reconstruction_semantic_geometry.py \
      tests/unit/test_phase8_7b_backward_compatibility.py
    ;;
  integration)
    pytest -q tests/integration/test_reconstruction_semantic_geometry_pipeline.py
    ;;
  stress)
    pytest -q tests/stress/test_reconstruction_semantic_geometry_stress.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|stress|full}" >&2
    exit 2
    ;;
esac
