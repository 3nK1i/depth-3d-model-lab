#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"
TMPDIR_I44D="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_I44D"' EXIT

FOCUSED=(
  tests/unit/test_i4_4d_final_i4_closure.py
  tests/integration/test_i4_4d_final_i4_closure_contract.py
  tests/stress/test_i4_4d_final_i4_closure_stress.py
)

PREDECESSOR_UNIT=(
  tests/unit/test_i4_4a_offline_dense_foundation_regression.py
  tests/unit/test_i4_4b_cross_provider_visual_verification.py
  tests/unit/test_i4_4c_provenance_authority_audit.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i4_4a_offline_dense_foundation_contract.py
  tests/integration/test_i4_4b_cross_provider_visual_verification_contract.py
  tests/integration/test_i4_4c_provenance_authority_audit_contract.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i4_4a_offline_dense_foundation_stress.py
  tests/stress/test_i4_4b_cross_provider_visual_verification_stress.py
  tests/stress/test_i4_4c_provenance_authority_audit_stress.py
)

run_pytest_xml() {
  local xml="$1"
  shift
  python -m pytest -q --junitxml="$xml" "$@"
  local rc=$?
  echo "pytest_rc=$rc"
  return "$rc"
}

focused() { run_pytest_xml "$TMPDIR_I44D/focused.xml" "${FOCUSED[@]}"; }
predecessor_unit() { run_pytest_xml "$TMPDIR_I44D/unit.xml" "${PREDECESSOR_UNIT[@]}"; }
predecessor_integration() { run_pytest_xml "$TMPDIR_I44D/integration.xml" "${PREDECESSOR_INTEGRATION[@]}"; }
predecessor_stress() { run_pytest_xml "$TMPDIR_I44D/stress.xml" "${PREDECESSOR_STRESS[@]}"; }

probe() {
  python - <<'PY'
from core.reconstruction.offline_dense_final_i4_closure import I44D_PHASE, build_i4_4d_final_i4_closure
from tests.i4_4d_helpers import build_passing_i4_4d
assert I44D_PHASE == "I4.4D"
assert callable(build_i4_4d_final_i4_closure)
result = build_passing_i4_4d()
assert result["decision"]["i4_closed"] is True
assert result["decision"]["i5_authorized"] is True
assert result["decision"]["i5_started"] is False
assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
assert result["authority_audit"]["default_reconstruction_scale_status"] == "arbitrary"
print("I4.4D final I4 closure contract/import probe: PASS")
PY
  local rc=$?
  echo "import_probe_rc=$rc"
  [ "$rc" -eq 0 ] || return "$rc"
  python scripts/run_i4_4d_final_i4_closure.py --help >/dev/null
  rc=$?
  echo "closure_runner_help_rc=$rc"
  return "$rc"
}

xml_count() {
  python - "$1" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
root = ET.parse(Path(sys.argv[1])).getroot()
if root.tag == "testsuite":
    print(int(root.attrib.get("tests", 0)))
else:
    print(sum(int(item.attrib.get("tests", 0)) for item in root.findall("testsuite")))
PY
}

write_receipt() {
  local target="$1"
  local focused_count unit_count integration_count stress_count file_count
  focused_count="$(xml_count "$TMPDIR_I44D/focused.xml")"
  unit_count="$(xml_count "$TMPDIR_I44D/unit.xml")"
  integration_count="$(xml_count "$TMPDIR_I44D/integration.xml")"
  stress_count="$(xml_count "$TMPDIR_I44D/stress.xml")"
  file_count=$(( ${#PREDECESSOR_UNIT[@]} + ${#PREDECESSOR_INTEGRATION[@]} + ${#PREDECESSOR_STRESS[@]} ))
  python - "$target" "$focused_count" "$unit_count" "$integration_count" "$stress_count" "$file_count" <<'PY'
from pathlib import Path
import json
import sys
path = Path(sys.argv[1]).expanduser().resolve()
focused, unit, integration, stress, file_count = map(int, sys.argv[2:])
predecessor_total = unit + integration + stress
payload = {
    "schema_version": 1,
    "phase": "I4.4D_FINAL_CLOSURE_RECEIPT",
    "status": "pass",
    "pass": True,
    "scope": {
        "focused_i4_4d_tests": True,
        "predecessor_i4_4_unit_regression": True,
        "predecessor_i4_4_integration_regression": True,
        "predecessor_i4_4_stress_regression": True,
        "import_probe": True,
        "runner_help_probe": True,
        "provider_execution_started": False,
        "provider_runtime_probe_started": False,
        "sfm_rerun": False,
        "matched_view_rerun": False,
        "manual_review_rerun": False,
        "new_provider_comparison_started": False,
        "i5_work_started": False,
    },
    "counts": {
        "focused_tests": focused,
        "predecessor_test_files": file_count,
        "predecessor_unit_tests": unit,
        "predecessor_integration_tests": integration,
        "predecessor_stress_tests": stress,
        "predecessor_total_tests": predecessor_total,
        "combined_exercised_tests": focused + predecessor_total,
    },
    "return_codes": {
        "focused": 0,
        "predecessor_unit": 0,
        "predecessor_integration": 0,
        "predecessor_stress": 0,
        "import_probe": 0,
        "runner_help_probe": 0,
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"closure_receipt={path}")
PY
}

all_suites() {
  focused || return $?
  probe || return $?
  predecessor_unit || return $?
  predecessor_integration || return $?
  predecessor_stress || return $?
  if [ -n "$RECEIPT" ]; then
    write_receipt "$RECEIPT" || return $?
  fi
  echo "I4.4D_TEST_RC=0"
}

case "$MODE" in
  focused) focused ;;
  probe) probe ;;
  unit) predecessor_unit ;;
  integration) predecessor_integration ;;
  stress) predecessor_stress ;;
  regression) predecessor_unit && predecessor_integration && predecessor_stress ;;
  all) all_suites ;;
  *)
    echo "usage: $0 [focused|probe|unit|integration|stress|regression|all] [receipt.json]" >&2
    exit 2
    ;;
esac
