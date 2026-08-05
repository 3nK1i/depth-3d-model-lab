#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-quick}"
python -m compileall -q app core tests scripts main.py
pytest -q tests/unit/test_snapshot_export.py tests/unit/test_snapshot_cli.py
pytest -q tests/ui/test_snapshot_controller.py tests/ui/test_snapshot_page.py
if [[ "$MODE" == "full" ]]; then
  ./scripts/run_phase5_tests.sh
fi
