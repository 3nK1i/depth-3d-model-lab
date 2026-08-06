#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_inspection_export.py \
      tests/unit/test_inspection_export_manifest.py \
      tests/unit/test_phase75_step9_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_measurement_inspection_export.py
    ;;
  ui)
    python -m pytest -q tests/ui/test_inspection_export_ui.py tests/ui/test_inspection_page.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *) echo "Usage: $0 {quick|integration|ui|full}"; exit 2;;
esac
