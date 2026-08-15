#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-quick}"

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_pre9_4e_interactive_distance.py \
      tests/unit/test_pre9_4e_projected_surface_pick.py \
      tests/ui/test_pre9_4e_interactive_measurement_ui.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_pre9_4e_interactive_measurement_bundle.py
    python scripts/run_pre9_4e_interactive_measurement_smoke.py
    ;;
  backward)
    python -m pytest -q \
      tests/ui/test_inspection_page.py \
      tests/ui/test_inspection_canvas_painting.py \
      tests/ui/test_phase9_3b_canonical_geometry_viewer.py \
      tests/ui/test_phase9_3j_measurement_canonical_registration.py \
      tests/unit/test_measurement_uncertainty_pipeline.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
