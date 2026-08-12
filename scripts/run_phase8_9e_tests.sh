#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

quick=(
  tests/unit/test_phase8_9e_endurance_contract.py
  tests/integration/test_phase8_9e_endurance_recovery.py
  tests/stress/test_phase8_9e_endurance_resource_recovery_stress.py
  tests/unit/test_reconstruction_engine_endurance_g3.py
  tests/integration/test_reconstruction_engine_endurance_g3.py
  tests/stress/test_reconstruction_engine_endurance_g3_stress.py
)

affected=(
  "${quick[@]}"
  tests/integration/test_phase8_8h2_resource_coordination.py
  tests/integration/test_phase8_8h3_failure_cancellation_recovery.py
  tests/ui/test_phase8_8h2_application_resource_coordination.py
  tests/ui/test_phase8_8h3_failure_recovery_integration.py
  tests/ui/test_phase8_9e_endurance_resource_recovery.py
  tests/ui/test_shutdown_lifecycle.py
  tests/stress/test_phase8_9c_persistence_reproducibility_stress.py
  tests/integration/test_phase8_9a1_reconstruction_export.py
  tests/integration/test_phase8_9b_qc_report_export.py
)

case "$MODE" in
  quick) pytest -q "${quick[@]}" ;;
  affected) pytest -q "${affected[@]}" ;;
  *) echo "usage: $0 [quick|affected]" >&2; exit 2 ;;
esac
