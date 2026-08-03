#!/usr/bin/env bash
# Reproducible Phase 5 release gate.
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
  *)
    echo "Usage: $0 {quick|full} [release_check options]" >&2
    exit 2
    ;;
esac
