#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase9_3e_canonical_inspection_overlays.py \
      tests/ui/test_phase9_3e_canonical_measurement_overlay.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_3e_canonical_inspection_overlay.py
    python scripts/run_phase9_3e_canonical_inspection_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase8_8e2_reconstruction_measurement_presentation.py \
      tests/unit/test_phase8_8e3_reconstruction_measurement_validity_presentation.py \
      tests/unit/test_phase8_8e4_reconstruction_inspection_qc_presentation.py \
      tests/unit/test_phase9_3d_canonical_semantic_presentation.py \
      tests/integration/test_phase9_3d_persistent_semantic_scene.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
