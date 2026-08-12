#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

quick=(
  tests/unit/test_phase8_9d_gpu_production_profile.py
  tests/unit/test_phase8_9d_hardware_agnostic_contract.py
  tests/integration/test_phase8_9d_gpu_profile_contract.py
  tests/unit/test_gpu_profiles.py
  tests/unit/test_reconstruction_profiles_backends.py
  tests/unit/test_hardware_tuning_matrix.py
)

affected=(
  "${quick[@]}"
  tests/unit/test_gpu_info.py
  tests/unit/test_runtime_telemetry.py
  tests/unit/test_performance_formatting.py
  tests/ui/test_hardware_profile_ui.py
  tests/ui/test_performance_bar.py
  tests/unit/test_phase8_8h2_contract.py
  tests/unit/test_phase8_8h3_contract.py
)

case "$MODE" in
  quick) pytest -q "${quick[@]}" ;;
  affected) pytest -q "${affected[@]}" ;;
  *) echo "usage: $0 [quick|affected]" >&2; exit 2 ;;
esac
