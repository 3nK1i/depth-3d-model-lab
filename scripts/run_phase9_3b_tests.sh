#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase9_3b_canonical_geometry_loader.py \
      tests/ui/test_phase9_3b_canonical_geometry_viewer.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_3b_persistent_canonical_scene_viewer.py
    python scripts/run_phase9_3b_canonical_mesh_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_3a_canonical_scene_presentation.py \
      tests/ui/test_phase9_3a_canonical_scene_presentation.py \
      tests/unit/test_phase8_8c2_visualizer_contract.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/unit/test_phase8_8f4_reconstruction_representation_switching.py \
      tests/ui/test_phase8_8f4_representation_switching.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
