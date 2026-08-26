#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    python -m py_compile \
      scripts/run_b2_sfm_dense_real_alignment.py \
      tests/unit/test_b2_sfm_dense_real_alignment.py
    pytest -q \
      tests/unit/test_reconstruction_sfm_dense_alignment.py \
      tests/unit/test_b2_sfm_dense_real_alignment.py
    ;;
  affected)
    pytest -q \
      tests/unit/test_reconstruction_sfm_dense_alignment.py \
      tests/unit/test_b2_sfm_dense_real_alignment.py \
      tests/unit/test_reconstruction_dense_fusion.py \
      tests/unit/test_reconstruction_sfm_contracts.py \
      tests/unit/test_reconstruction_sfm_coordinates.py \
      tests/unit/test_reconstruction_sfm_pycolmap.py \
      tests/unit/test_reconstruction_da3.py \
      tests/unit/test_reconstruction_da3_provider.py \
      tests/unit/test_reconstruction_pipeline_geometry_f3.py \
      tests/integration/test_reconstruction_dense_fusion_pipeline.py
    ;;
  *)
    echo "Usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
