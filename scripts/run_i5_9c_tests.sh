#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOCUSED=(
  tests/unit/test_i5_9c_remaining_depth_provider_robustness_closure.py
  tests/integration/test_i5_9c_remaining_depth_provider_robustness_contract.py
  tests/stress/test_i5_9c_remaining_depth_provider_robustness_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i5_9b2_relative_depth_provider_robustness_replay.py
  tests/unit/test_i5_9b1_metric_like_depth_provider_robustness_replay.py
  tests/unit/test_i5_9a_remaining_depth_provider_robustness_admission.py
  tests/unit/test_i5_8c_video_depth_anything_robustness_closure.py
  tests/unit/test_i4_3g4c_single_image_matched_view_review.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i5_9b2_relative_depth_provider_robustness_contract.py
  tests/integration/test_i5_9b1_metric_like_depth_provider_robustness_contract.py
  tests/integration/test_i5_9a_remaining_depth_provider_robustness_contract.py
  tests/integration/test_i5_8c_video_depth_anything_robustness_contract.py
  tests/integration/test_i4_3g4c_single_image_review_contract.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i5_9b2_relative_depth_provider_robustness_stress.py
  tests/stress/test_i5_9b1_metric_like_depth_provider_robustness_stress.py
  tests/stress/test_i5_9a_remaining_depth_provider_robustness_stress.py
  tests/stress/test_i5_8c_video_depth_anything_robustness_stress.py
  tests/stress/test_i4_3g4c_single_image_review_stress.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
PUNIT_RC=0; PUNIT_COUNT=0
PINT_RC=0; PINT_COUNT=0
PSTRESS_RC=0; PSTRESS_COUNT=0
IMPORT_RC=0; HELP_RC=0

run_pytest_count() {
  local label="$1"; shift
  local tmp; tmp="$(mktemp)"
  pytest -q "$@" >"$tmp" 2>&1
  local rc=$?
  cat "$tmp"
  local count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
print(sum(int(x) for x in re.findall(r'(\d+) passed',text)))
PY
)"
  fi
  rm -f "$tmp"
  printf '%s_RC=%s\n' "$label" "$rc"
  printf '%s_COUNT=%s\n' "$label" "$count"
  eval "${label}_RC=$rc"
  eval "${label}_COUNT=$count"
}

probe_import() {
  python - <<'PY'
from core.reconstruction.i5_remaining_depth_provider_robustness_qualification import (
    I59C_PHASE,
    I59C_I59A_FROZEN_SHA256,
    I59C_I59B1_FROZEN_SHA256,
    I59C_B1_STATUS_COUNTS,
    I59C_B2_STATUS_COUNTS,
    I59C_COMBINED_STATUS_COUNTS,
    build_i5_9c_remaining_depth_provider_robustness_closure,
    validate_i5_9b2_for_i5_9c,
)
assert I59C_PHASE == 'I5.9C'
assert len(I59C_I59A_FROZEN_SHA256) == 64
assert len(I59C_I59B1_FROZEN_SHA256) == 64
assert I59C_B1_STATUS_COUNTS == {'pass': 24, 'degraded': 0, 'rejected': 0}
assert I59C_B2_STATUS_COUNTS == {'pass': 11, 'degraded': 1, 'rejected': 0}
assert I59C_COMBINED_STATUS_COUNTS == {'pass': 35, 'degraded': 1, 'rejected': 0}
assert callable(build_i5_9c_remaining_depth_provider_robustness_closure)
assert callable(validate_i5_9b2_for_i5_9c)
print('I5.9C remaining depth-provider robustness closure contract/import probe: PASS')
PY
  IMPORT_RC=$?
  echo "import_probe_rc=$IMPORT_RC"
}

probe_help() {
  python scripts/run_i5_9c_remaining_depth_provider_robustness_closure.py --help >/dev/null
  HELP_RC=$?
  echo "closure_runner_help_rc=$HELP_RC"
}

write_receipt() {
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]).expanduser().resolve()
focused,unit,integ,stress=map(int,sys.argv[2:6])
rcs=list(map(int,sys.argv[6:12]))
pred=unit+integ+stress
passed=all(v==0 for v in rcs)
payload={
  'schema_version':1,
  'phase':'I5.9C_QUALIFICATION_RECEIPT',
  'status':'pass' if passed else 'fail',
  'pass':passed,
  'scope':{
    'focused_i5_9c_tests':True,
    'i5_9a_admission_regression':True,
    'i5_9b1_replay_regression':True,
    'i5_9b2_replay_regression':True,
    'historical_g4_preservation_regression':True,
    'mixed_terminal_quality_regression':True,
    'authority_guard_regression':True,
    'roadmap_guard_regression':True,
    'import_probe':True,
    'runner_help_probe':True,
    'provider_inference':False,
    'sfm_rerun':False,
    'mvs_rerun':False,
    'provider_comparison_started':False,
    'thresholds_modified':False,
    'fusion_capacity_increased':False,
    'production_default_modified':False,
    'result_driven_retuning':False,
    'retry_until_pass':False,
    'mp_sfm_started':False,
    'auto_policy_implemented':False,
    'hybrid_authorized':False,
  },
  'counts':{
    'focused_tests':focused,
    'predecessor_unit_tests':unit,
    'predecessor_integration_tests':integ,
    'predecessor_stress_tests':stress,
    'predecessor_total_tests':pred,
    'combined_exercised_tests':focused+pred,
  },
  'return_codes':{
    'focused':rcs[0],
    'predecessor_unit':rcs[1],
    'predecessor_integration':rcs[2],
    'predecessor_stress':rcs[3],
    'import_probe':rcs[4],
    'runner_help_probe':rcs[5],
  },
}
p.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
print(f'closure_receipt={p}')
PY
}

case "$MODE" in
  focused)
    run_pytest_count FOCUSED "${FOCUSED[@]}"
    probe_import
    probe_help
    ;;
  predecessors)
    run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
    run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
    run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
    probe_import
    probe_help
    ;;
  all)
    run_pytest_count FOCUSED "${FOCUSED[@]}"
    probe_import
    probe_help
    run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
    run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
    run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
    ;;
  *)
    echo "usage: $0 [focused|predecessors|all] [receipt.json]" >&2
    exit 2
    ;;
esac

write_receipt
RC=0
for v in "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC"; do
  [[ "$v" -eq 0 ]] || RC=1
done
echo "I5.9C_TEST_RC=$RC"
exit "$RC"
