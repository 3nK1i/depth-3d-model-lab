#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"

case "$mode" in
  quick)
    python -m compileall -q core/reconstruction/export.py core/reconstruction/__init__.py \
      tests/unit/test_phase8_9a1_reconstruction_export.py \
      tests/integration/test_phase8_9a1_reconstruction_export.py
    pytest -q \
      tests/unit/test_phase8_9a1_reconstruction_export.py \
      tests/integration/test_phase8_9a1_reconstruction_export.py
    ;;
  affected)
    python -m compileall -q core/reconstruction/export.py core/reconstruction/__init__.py
    pytest -q \
      tests/unit/test_phase8_9a1_reconstruction_export.py \
      tests/integration/test_phase8_9a1_reconstruction_export.py \
      tests/unit/test_reconstruction_surface_mesh.py \
      tests/unit/test_reconstruction_inspection_qc.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/unit/test_phase8_7a_backward_compatibility.py \
      tests/unit/test_phase8_7c_backward_compatibility.py \
      tests/integration/test_reconstruction_surface_mesh_pipeline.py \
      tests/integration/test_reconstruction_inspection_qc_pipeline.py \
      tests/integration/test_reconstruction_persistent_scene_f5.py
    ;;
  *)
    echo "Usage: $0 [quick|affected]" >&2
    exit 2
    ;;
esac
