#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
MODE="${1:-all}"; RECEIPT="${2:-}"
FOCUSED=(tests/unit/test_i5_10d_mp_sfm_closure.py tests/integration/test_i5_10d_mp_sfm_closure_contract.py tests/stress/test_i5_10d_mp_sfm_closure_stress.py)
PUNIT=(tests/unit/test_i5_10c_mp_sfm_real_evaluation.py tests/unit/test_i5_10b_mp_sfm_preflight.py tests/unit/test_i5_10a_mp_sfm_admission.py)
PINT=(tests/integration/test_i5_10c_mp_sfm_real_evaluation_contract.py tests/integration/test_i5_10b_mp_sfm_preflight_contract.py tests/integration/test_i5_10a_mp_sfm_contract.py)
PSTRESS=(tests/stress/test_i5_10c_mp_sfm_real_evaluation_stress.py tests/stress/test_i5_10b_mp_sfm_preflight_stress.py tests/stress/test_i5_10a_mp_sfm_stress.py)
run(){ local label="$1"; shift; local t; t="$(mktemp)"; pytest -q "$@" | tee "$t"; local rc=${PIPESTATUS[0]}; local count=0; [[ $rc -ne 0 ]] || count="$(python - "$t" <<'PY'
import re,sys
print(sum(map(int,re.findall(r'(\d+) passed',open(sys.argv[1],encoding='utf-8',errors='replace').read()))))
PY
)"; rm -f "$t"; echo "${label}_RC=$rc"; echo "${label}_COUNT=$count"; eval "${label}_RC=$rc"; eval "${label}_COUNT=$count"; }
probe_import(){ python - <<'PY'
from core.reconstruction.i5_mp_sfm_qualification import I510D_PHASE, build_i5_10d_mp_sfm_closure
assert I510D_PHASE == 'I5.10D'; assert callable(build_i5_10d_mp_sfm_closure)
print('I5.10D MP-SfM closure contract/import probe: PASS')
PY
IMPORT_RC=$?; echo "import_probe_rc=$IMPORT_RC"; }
probe_help(){ python scripts/run_i5_10d_mp_sfm_closure.py --help >/dev/null; HELP_RC=$?; echo "closure_runner_help_rc=$HELP_RC"; }
write_receipt(){ [[ -n "$RECEIPT" ]] || return 0; mkdir -p "$(dirname "$RECEIPT")"; python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
p=sys.argv[1]; f,u,i,s=map(int,sys.argv[2:6]); r=list(map(int,sys.argv[6:12])); pred=u+i+s; ok=all(x==0 for x in r)
x={'schema_version':1,'phase':'I5.10D_CLOSURE_RECEIPT','status':'pass' if ok else 'fail','pass':ok,'scope':{'focused_i5_10d_tests':True,'predecessor_i5_10c_regression':True,'evidence_only':True,'sfm_execution':False,'mvs_execution':False,'provider_inference':False},'counts':{'focused_tests':f,'predecessor_unit_tests':u,'predecessor_integration_tests':i,'predecessor_stress_tests':s,'predecessor_total_tests':pred,'combined_exercised_tests':f+pred},'return_codes':{'focused':r[0],'predecessor_unit':r[1],'predecessor_integration':r[2],'predecessor_stress':r[3],'import_probe':r[4],'runner_help_probe':r[5]}}
open(p,'w').write(json.dumps(x,indent=2,sort_keys=True)+'\n'); print('closure_receipt='+p)
PY
}
case "$MODE" in
 focused) pytest -q "${FOCUSED[@]}";;
 predecessor-unit) pytest -q "${PUNIT[@]}";;
 predecessor-integration) pytest -q "${PINT[@]}";;
 predecessor-stress) pytest -q "${PSTRESS[@]}";;
 all) run FOCUSED "${FOCUSED[@]}"; probe_import; probe_help; run PUNIT "${PUNIT[@]}"; run PINT "${PINT[@]}"; run PSTRESS "${PSTRESS[@]}"; write_receipt; FINAL_RC=0; for v in "$FOCUSED_RC" "$IMPORT_RC" "$HELP_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC"; do [[ "$v" -eq 0 ]] || FINAL_RC=1; done; echo "I5.10D_TEST_RC=$FINAL_RC"; exit "$FINAL_RC";;
 *) echo "usage: $0 [focused|predecessor-unit|predecessor-integration|predecessor-stress|all] [receipt.json]" >&2; exit 2;;
esac
