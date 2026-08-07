#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase8_5a_backward_compatibility.py \
      tests/unit/test_reconstruction_global_consistency.py \
      tests/unit/test_reconstruction_pose_graph.py \
      tests/unit/test_reconstruction_relocalization.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_reconstruction_global_consistency_pipeline.py
    ;;
  stress)
    python -m pytest -q \
      tests/stress/test_reconstruction_pose_graph_stress.py
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
