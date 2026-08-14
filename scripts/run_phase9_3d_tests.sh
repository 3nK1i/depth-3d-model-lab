#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase9_3d_canonical_semantic_presentation.py \
      tests/ui/test_phase9_3d_semantic_canonical_viewer.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_3d_persistent_semantic_scene.py
    python scripts/run_phase9_3d_semantic_scene_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase8_8d5_offline_semantic_presentation.py \
      tests/ui/test_phase8_8d5_semantic_object_presentation.py \
      tests/unit/test_phase9_3b_canonical_geometry_loader.py \
      tests/unit/test_phase9_3c_canonical_camera_presentation.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
