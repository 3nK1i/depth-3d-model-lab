#!/usr/bin/env bash
# Reproducible Phase 7 operational-mode release gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-full}"
shift || true

case "$MODE" in
  quick|full)
    exec "$PYTHON_BIN" scripts/release_check.py --mode "$MODE" "$@"
    ;;
  benchmark)
    exec "$PYTHON_BIN" scripts/run_phase6_benchmark.py "$@"
    ;;
  *)
    echo "Usage: $0 {quick|full|benchmark} [options]" >&2
    exit 2
    ;;
esac
