#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$MODE" in
  quick)
    python -m pytest -q tests/unit/test_phase9_5a_nerf_runtime_isolation.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase9_5a_nerf_runtime_isolation.py
    python scripts/run_phase9_5a_nerf_runtime_smoke.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_reconstruction_nerf.py \
      tests/unit/test_reconstruction_nerf_e2.py \
      tests/unit/test_reconstruction_nerf_e3.py \
      tests/unit/test_reconstruction_nerf_e4.py \
      tests/integration/test_reconstruction_nerf_external_pipeline.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
