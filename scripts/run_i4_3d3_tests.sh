#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

focused=(
  tests/unit/test_i4_3d1_video_depth_anything_dense_provider.py
  tests/integration/test_i4_3d1_video_depth_anything_provider_contract.py
  tests/unit/test_i4_3d2_rolling_depth_dense_provider.py
  tests/integration/test_i4_3d2_rolling_depth_provider_contract.py
  tests/unit/test_i4_3d3_temporal_provider_closure.py
  tests/integration/test_i4_3d3_temporal_provider_closure_contract.py
)
stress=(
  tests/stress/test_i4_3d1_video_depth_anything_provider_stress.py
  tests/stress/test_i4_3d2_rolling_depth_provider_stress.py
  tests/stress/test_i4_3d3_temporal_provider_closure_stress.py
)

case "${1:-all}" in
  focused) python -m pytest -q "${focused[@]}" ;;
  stress) python -m pytest -q "${stress[@]}" ;;
  all) python -m pytest -q "${focused[@]}" "${stress[@]}" ;;
  *) echo "usage: $0 [focused|stress|all]" >&2; exit 2 ;;
esac
