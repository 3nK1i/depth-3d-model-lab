#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase8_9f2_real_geometry_campaign.py \
      tests/integration/test_phase8_9f2_real_geometry_contract.py
    ;;
  affected)
    pytest -q \
      tests/unit/test_phase8_9f2_real_geometry_campaign.py \
      tests/integration/test_phase8_9f2_real_geometry_contract.py \
      tests/unit/test_reconstruction_sfm_pycolmap.py \
      tests/unit/test_reconstruction_sfm_quality.py \
      tests/unit/test_reconstruction_da3.py \
      tests/unit/test_reconstruction_da3_provider.py \
      tests/integration/test_reconstruction_da3_provider.py \
      tests/unit/test_reconstruction_pipeline_geometry_f3.py \
      tests/integration/test_reconstruction_pipeline_geometry_f3.py \
      tests/unit/test_phase8_7f2_backward_compatibility.py \
      tests/unit/test_phase8_7f3_backward_compatibility.py \
      tests/unit/test_phase8_9f1_external_backend_preflight.py \
      tests/integration/test_phase8_9f1_external_backend_contract.py
    ;;
  *)
    echo "usage: $0 [quick|affected]" >&2
    exit 2
    ;;
esac
