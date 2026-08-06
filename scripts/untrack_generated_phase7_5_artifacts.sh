#!/usr/bin/env bash
# Remove accidentally committed runtime artifacts from Git while preserving local copies.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: this command must run inside the project Git checkout" >&2
  exit 2
fi

git rm -r --cached --ignore-unmatch \
  yolo11n-seg.pt \
  outputs/snapshots

echo "Generated checkpoint/snapshot files are now untracked but remain on disk."
echo "The updated .gitignore prevents them from being staged again."
