#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_8b_video_depth_anything_robustness_replay.py
  tests/integration/test_i5_8b_video_depth_anything_robustness_contract.py
  tests/stress/test_i5_8b_video_depth_anything_robustness_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i4_3d1_video_depth_anything_dense_provider.py
  tests/unit/test_i4_3g3a_temporal_provider_qualification.py
  tests/unit/test_i4_3g3b_temporal_production_qualification.py
  tests/unit/test_i4_3g3c_temporal_matched_view_review.py
  tests/unit/test_i5_6a_mvs_robustness_admission.py
  tests/unit/test_i5_7c_rolling_depth_robustness_closure.py
  tests/unit/test_i5_8a_video_depth_anything_robustness_admission.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i4_3d1_video_depth_anything_provider_contract.py
  tests/integration/test_i4_3g3a_temporal_preflight_contract.py
  tests/integration/test_i4_3g3b_temporal_production_contract.py
  tests/integration/test_i4_3g3c_temporal_review_contract.py
  tests/integration/test_i5_6a_mvs_robustness_contract.py
  tests/integration/test_i5_7c_rolling_depth_robustness_contract.py
  tests/integration/test_i5_8a_video_depth_anything_robustness_contract.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i4_3d1_video_depth_anything_provider_stress.py
  tests/stress/test_i4_3g3a_temporal_preflight_stress.py
  tests/stress/test_i4_3g3b_temporal_production_stress.py
  tests/stress/test_i4_3g3c_temporal_review_stress.py
  tests/stress/test_i5_6a_mvs_robustness_stress.py
  tests/stress/test_i5_7c_rolling_depth_robustness_stress.py
  tests/stress/test_i5_8a_video_depth_anything_robustness_stress.py
)

run_pytest_count(){
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

probe_import(){
  python - <<'PY'
from core.reconstruction.i5_video_depth_anything_robustness_replay import (
    I58B_PHASE, I58B_SCENARIO_IDS, I58B_I58A_FROZEN_SHA256,
    I58BReplayPolicy, build_i5_8b_video_depth_anything_robustness_replay,
)
assert I58B_PHASE == 'I5.8B'
assert len(I58B_SCENARIO_IDS) == 6
assert len(I58B_I58A_FROZEN_SHA256) == 64
p=I58BReplayPolicy().as_dict()
assert p['temporal_mode'] == 'official_streaming_cache'
assert p['per_frame_alignment_performed'] is True
assert p['sequence_global_alignment_performed'] is False
assert p['retry_until_pass'] is False
assert callable(build_i5_8b_video_depth_anything_robustness_replay)
print('I5.8B Video Depth Anything robustness replay contract/import probe: PASS')
PY
  IMPORT_RC=$?; echo "import_probe_rc=$IMPORT_RC"
}

probe_help(){
  python scripts/run_i5_8b_video_depth_anything_robustness_replay.py --help >/dev/null
  HELP_RC=$?; echo "replay_runner_help_rc=$HELP_RC"
}

write_receipt(){
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
p=sys.argv[1]
focused,unit,integ,stress=map(int,sys.argv[2:6])
rcs=list(map(int,sys.argv[6:12]))
pred=unit+integ+stress
passed=all(v==0 for v in rcs)
payload={
  "schema_version":1,
  "phase":"I5.8B_REPLAY_RECEIPT",
  "status":"pass" if passed else "fail",
  "pass":passed,
  "scope":{
    "focused_i5_8b_tests":True,
    "video_depth_anything_provider_regression":True,
    "i5_8a_predecessor_regression":True,
    "authority_guard_regression":True,
    "runtime_runner_help_probe":True,
    "video_depth_anything_inference":False,
    "sfm_rerun":False,
    "mvs_rerun":False,
    "provider_comparison_started":False,
    "thresholds_modified":False,
    "production_default_modified":False,
    "result_driven_retuning":False,
    "retry_until_pass":False,
    "mp_sfm_started":False,
    "auto_policy_implemented":False,
    "hybrid_authorized":False,
  },
  "counts":{
    "focused_tests":focused,
    "predecessor_unit_tests":unit,
    "predecessor_integration_tests":integ,
    "predecessor_stress_tests":stress,
    "predecessor_total_tests":pred,
    "combined_exercised_tests":focused+pred,
  },
  "return_codes":{
    "focused":rcs[0],
    "predecessor_unit":rcs[1],
    "predecessor_integration":rcs[2],
    "predecessor_stress":rcs[3],
    "import_probe":rcs[4],
    "runner_help_probe":rcs[5],
  },
}
with open(p,'w',encoding='utf-8') as f:
    json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
print(f'replay_receipt={p}')
PY
}

FOCUSED_RC=0; FOCUSED_COUNT=0; PUNIT_RC=0; PUNIT_COUNT=0; PINT_RC=0; PINT_COUNT=0; PSTRESS_RC=0; PSTRESS_COUNT=0; IMPORT_RC=0; HELP_RC=0
case "$MODE" in
  focused) run_pytest_count FOCUSED "${FOCUSED[@]}"; probe_import; probe_help ;;
  predecessor)
    run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
    run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
    run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
    probe_import; probe_help
    ;;
  all)
    run_pytest_count FOCUSED "${FOCUSED[@]}"
    probe_import
    probe_help
    run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
    run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
    run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
    ;;
  *) echo "usage: $0 [focused|predecessor|all] [receipt.json]" >&2; exit 2 ;;
esac
write_receipt
RC=0
for v in "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC"; do [[ "$v" -eq 0 ]] || RC=1; done
echo "I5.8B_TEST_RC=$RC"
exit "$RC"
