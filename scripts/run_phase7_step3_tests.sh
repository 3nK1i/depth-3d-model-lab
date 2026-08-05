#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pytest -q tests/unit/test_stable_video_export.py

if python - <<'PYTEST_DEPS'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("PyQt6") and importlib.util.find_spec("vispy") else 1)
PYTEST_DEPS
then
  pytest -q tests/ui/test_stable_video_controller.py tests/ui/test_stable_video_page.py
else
  echo "SKIP: PyQt6/VisPy unavailable; UI tests must run in the project environment."
fi

if python - <<'PYTEST_DEPS'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("PyQt6") else 1)
PYTEST_DEPS
then
  pytest -q tests/stress/test_stable_video_stress.py
else
  echo "SKIP: PyQt6 unavailable; Stable Video stress test must run in the project environment."
fi

if [[ "$MODE" == "full" ]]; then
  ./scripts/run_phase5_tests.sh
fi
