#!/usr/bin/env bash
set -u
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"

run_core() {
  pytest -q \
    tests/unit/test_i4_3f3_adaptive_selection_closure.py \
    tests/integration/test_i4_3f3_production_validation_contract.py \
    tests/stress/test_i4_3f3_production_validation_stress.py || return $?
  python - <<'PY'
from core.reconstruction.adaptive_selection_closure import build_i4_3f3_closure_snapshot
from core.reconstruction.keyframes import KeyframeSelectionPolicy
assert KeyframeSelectionPolicy().maximum_keyframes is None
assert callable(build_i4_3f3_closure_snapshot)
print("I4.3F3 production-validation closure/import probe: PASS")
PY
}

run_affected() {
  pytest -q \
    tests/unit/test_i4_3f1_adaptive_view_selection.py \
    tests/integration/test_i4_3f1_adaptive_view_selection_contract.py \
    tests/stress/test_i4_3f1_adaptive_view_selection_stress.py \
    tests/unit/test_i4_3f2_adaptive_production_planning.py \
    tests/unit/test_i4_3f2_production_source_contract.py \
    tests/integration/test_i4_3f2_production_wiring_contract.py \
    tests/stress/test_i4_3f2_production_wiring_stress.py \
    tests/unit/test_reconstruction_ingestion_quality_sampling.py \
    tests/unit/test_reconstruction_ingestion_pipeline.py \
    tests/unit/test_reconstruction_sfm_workspace.py \
    tests/unit/test_i4_3b_mvs_provider.py \
    tests/integration/test_i4_3b_mvs_provider_contract.py \
    tests/stress/test_i4_3b_mvs_provider_stress.py
}

run_qt() {
  if python - <<'PY' >/dev/null 2>&1
import PyQt6
PY
  then
    pytest -q \
      tests/unit/test_phase8_8a3_reconstruction_controller.py \
      tests/unit/test_phase8_8b3_reconstruction_runtime_controls.py
  else
    echo "I4.3F3 Qt/controller backward sanity: SKIP (PyQt6 unavailable in this environment)"
  fi
}

case "$MODE" in
  core) run_core ;;
  affected) run_affected ;;
  qt) run_qt ;;
  all)
    run_core || exit $?
    run_affected || exit $?
    run_qt || exit $?
    ;;
  *) echo "usage: $0 [core|affected|qt|all]" >&2; exit 2 ;;
esac
