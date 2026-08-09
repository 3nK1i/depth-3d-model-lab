#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/unit/test_phase8_7f5_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_reconstruction_persistent_scene_f5.py
    ;;
  stress)
    python -m pytest -q tests/stress/test_reconstruction_persistent_scene_f5_stress.py
    ;;
  *)
    echo "usage: $0 {quick|integration|stress}" >&2
    exit 2
    ;;
esac
