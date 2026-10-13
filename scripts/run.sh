#!/usr/bin/env bash
# Launch Depth & 3D Model Lab with the active environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Depth & 3D Model Lab requires Python 3.11 or 3.12; got {sys.version.split()[0]}"
    )
PY

exec "$PYTHON_BIN" main.py "$@"
