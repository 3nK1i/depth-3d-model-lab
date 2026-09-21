#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
MODE="${1:-all}"; RECEIPT="${2:-}"
FOCUSED=(tests/unit/test_i5_11d_failure_robustness_closure.py tests/integration/test_i5_11d_failure_robustness_closure_contract.py tests/stress/test_i5_11d_failure_robustness_closure_stress.py)
PUNIT=(tests/unit/test_i5_11c_failure_recovery_campaign.py tests/unit/test_i5_11b_failure_preflight.py tests/unit/test_i5_11a_failure_mode_admission.py)
PINT=(tests/integration/test_i5_11c_failure_recovery_campaign_contract.py tests/integration/test_i5_11b_failure_preflight_contract.py tests/integration/test_i5_11a_failure_mode_admission_contract.py)
PSTRESS=(tests/stress/test_i5_11c_failure_recovery_campaign_stress.py tests/stress/test_i5_11b_failure_preflight_stress.py tests/stress/test_i5_11a_failure_mode_admission_stress.py)
run(){ local label="$1"; shift; local t; t="$(mktemp)"; pytest -q "$@" | tee "$t"; local rc=${PIPESTATUS[0]}; local count=0; [[ $rc -ne 0 ]] || count="$(python - "$t" <<'PY'
import re,sys
print(sum(map(int,re.findall(r'(\d+) passed',open(sys.argv[1],encoding='utf-8',errors='replace').read()))))
PY
)"; rm -f "$t"; echo "${label}_RC=$rc"; echo "${label}_COUNT=$count"; eval "${label}_RC=$rc"; eval "${label}_COUNT=$count"; }
probe_import(){ python - <<'PY'
from core.reconstruction.i5_failure_robustness_qualification import I511D_PHASE, build_i5_11d_failure_robustness_closure
assert I511D_PHASE == 'I5.11D'; assert callable(build_i5_11d_failure_robustness_closure)
print('I5.11D failure-robustness closure contract/import probe: PASS')
PY
IMPORT_RC=$?; echo "import_probe_rc=$IMPORT_RC"; }
probe_help(){ python scripts/run_i5_11d_failure_robustness_closure.py --help >/dev/null; HELP_RC=$?; echo "closure_runner_help_rc=$HELP_RC"; }
write_receipt(){ [[ -n "$RECEIPT" ]] || return 0; mkdir -p "$(dirname "$RECEIPT")"; python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
p=sys.argv[1]; f,u,i,s=map(int,sys.argv[2:6]); r=list(map(int,sys.argv[6:12])); pred=u+i+s; ok=all(x==0 for x in r)
x={'schema_version':1,'phase':'I5.11D_CLOSURE_RECEIPT','status':'pass' if ok else 'fail','pass':ok,'scope':{'focused_i5_11d_tests':True,'predecessor_i5_11c_regression':True,'evidence_only':True,'fault_injection':False,'supervised_fault_worker_execution':False,'real_gpu_oom':False,'real_disk_fill':False,'sfm_execution':False,'mvs_execution':False,'provider_inference':False},'counts':{'focused_tests':f,'predecessor_unit_tests':u,'predecessor_integration_tests':i,'predecessor_stress_tests':s,'predecessor_total_tests':pred,'combined_exercised_tests':f+pred},'return_codes':{'focused':r[0],'predecessor_unit':r[1],'predecessor_integration':r[2],'predecessor_stress':r[3],'import_probe':r[4],'runner_help_probe':r[5]}}
open(p,'w').write(json.dumps(x,indent=2,sort_keys=True)+'\n'); print('closure_receipt='+p)
PY
}
case "$MODE" in
 focused) pytest -q "${FOCUSED[@]}";;
 predecessor-unit) pytest -q "${PUNIT[@]}";;
 predecessor-integration) pytest -q "${PINT[@]}";;
 predecessor-stress) pytest -q "${PSTRESS[@]}";;
 all) run FOCUSED "${FOCUSED[@]}"; probe_import; probe_help; run PUNIT "${PUNIT[@]}"; run PINT "${PINT[@]}"; run PSTRESS "${PSTRESS[@]}"; write_receipt; FINAL_RC=0; for v in "$FOCUSED_RC" "$IMPORT_RC" "$HELP_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC"; do [[ "$v" -eq 0 ]] || FINAL_RC=1; done; echo "I5.11D_TEST_RC=$FINAL_RC"; exit "$FINAL_RC";;
 *) echo "usage: $0 [focused|predecessor-unit|predecessor-integration|predecessor-stress|all] [receipt.json]" >&2; exit 2;;
esac
