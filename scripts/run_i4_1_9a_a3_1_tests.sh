#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mode="${1:-all}"

focused_tests=(
  tests/unit/test_i4_1_9a_mvs_environment.py
  tests/unit/test_i4_1_9a_mvs_camera_bridge.py
  tests/unit/test_i4_1_9a_mvs_execution.py
  tests/unit/test_i4_1_9a_mvs_visual_ab.py
  tests/unit/test_i4_1_9a_mvs_visual_ab_closure.py
  tests/integration/test_i4_1_9a_mvs_camera_bridge_contract.py
  tests/integration/test_i4_1_9a_mvs_execution_contract.py
  tests/integration/test_i4_1_9a_mvs_visual_ab_contract.py
  tests/integration/test_i4_1_9a_mvs_visual_ab_closure_contract.py
)
stress_tests=(
  tests/stress/test_i4_1_9a_mvs_camera_bridge_stress.py
  tests/stress/test_i4_1_9a_mvs_execution_stress.py
  tests/stress/test_i4_1_9a_mvs_visual_ab_stress.py
  tests/stress/test_i4_1_9a_mvs_visual_ab_closure_stress.py
)

case "$mode" in
  focused)
    python -m pytest -q "${focused_tests[@]}"
    ;;
  stress)
    python -m pytest -q "${stress_tests[@]}"
    ;;
  all)
    python -m pytest -q "${focused_tests[@]}" "${stress_tests[@]}"
    ;;
  *)
    echo "Usage: $0 [focused|stress|all]" >&2
    exit 2
    ;;
esac

python -m py_compile \
  core/reconstruction/mvs_environment.py \
  core/reconstruction/mvs_camera_bridge.py \
  core/reconstruction/mvs_execution.py \
  core/reconstruction/mvs_visual_ab.py \
  core/reconstruction/mvs_visual_ab_closure.py \
  scripts/reconstruction_pycolmap_mvs_bridge_worker.py \
  scripts/reconstruction_pycolmap_mvs_worker.py \
  scripts/run_i4_1_9a_a2_house_mvs.py \
  scripts/run_i4_1_9a_a3_visual_ab.py \
  scripts/run_i4_1_9a_a3_1_finalize_manual_verdict.py
