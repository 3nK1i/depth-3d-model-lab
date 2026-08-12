#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"

python -m compileall -q \
  app/widgets/reconstruction_page.py \
  core/reconstruction/export.py \
  tests/unit/test_phase8_9a2_reconstruction_export_ui_contract.py \
  tests/ui/test_phase8_9a2_reconstruction_export_ui.py

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase8_9a2_reconstruction_export_ui_contract.py \
      tests/unit/test_phase8_9a1_reconstruction_export.py \
      tests/ui/test_phase8_9a2_reconstruction_export_ui.py
    ;;
  affected)
    python -m pytest -q \
      tests/unit/test_phase8_9a2_reconstruction_export_ui_contract.py \
      tests/unit/test_phase8_9a1_reconstruction_export.py \
      tests/integration/test_phase8_9a1_reconstruction_export.py \
      tests/unit/test_phase8_8g5_contract.py \
      tests/unit/test_phase8_8g6_contract.py \
      tests/ui/test_phase8_8g5_progressive_disclosure_ux.py \
      tests/ui/test_phase8_8g6_full_reconstruction_ux_review.py \
      tests/ui/test_phase8_8h1_whole_reconstruction_workflow.py \
      tests/ui/test_phase8_9a2_reconstruction_export_ui.py
    ;;
  *)
    echo "Usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
