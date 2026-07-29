#!/usr/bin/env bash
# Launch EIA TECH — Depth & 3D Model Lab.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python3 main.py "$@"
