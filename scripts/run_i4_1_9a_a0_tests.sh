#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m pytest -q \
  tests/unit/test_i4_1_9a_mvs_environment.py \
  tests/unit/test_reconstruction_sfm_environment.py
python -m py_compile \
  core/reconstruction/mvs_environment.py \
  scripts/run_i4_1_9a_mvs_preflight.py
