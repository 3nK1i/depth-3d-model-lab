#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
mode="${1:-full}"
shift || true
case "$mode" in
  quick|full)
    exec python scripts/phase7_5_release_check.py --mode "$mode" "$@"
    ;;
  template)
    exec python scripts/create_known_size_benchmark_template.py "$@"
    ;;
  validate)
    exec python scripts/run_known_size_validation.py "$@"
    ;;
  *)
    echo "Usage: $0 {quick|full|template|validate} [options]" >&2
    exit 2
    ;;
esac
