#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"
TMPDIR_I44C="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_I44C"' EXIT

FOCUSED=(
  tests/unit/test_i4_4c_provenance_authority_audit.py
  tests/integration/test_i4_4c_provenance_authority_audit_contract.py
  tests/stress/test_i4_4c_provenance_authority_audit_stress.py
)

AFFECTED_UNIT=(
  tests/unit/test_i4_4a_offline_dense_foundation_regression.py
  tests/unit/test_i4_4b_cross_provider_visual_verification.py
  tests/unit/test_i4_3h1_final_qualification.py
  tests/unit/test_i4_3h2_final_closure.py
  tests/unit/test_i4_3g5b_cross_provider_qualification_closure.py
  tests/unit/test_i4_3a_dense_provider_architecture.py
  tests/unit/test_reconstruction_sfm_contracts.py
  tests/unit/test_i4_2e_2_native_visibility_provenance.py
  tests/unit/test_phase9_1h_persistent_authority.py
  tests/unit/test_phase9_1g_inspection_calibration_provenance.py
)

AFFECTED_INTEGRATION=(
  tests/integration/test_i4_4a_offline_dense_foundation_contract.py
  tests/integration/test_i4_4b_cross_provider_visual_verification_contract.py
  tests/integration/test_i4_3h1_final_qualification_contract.py
  tests/integration/test_i4_3h2_final_closure_contract.py
  tests/integration/test_i4_3g5b_final_g_closure_contract.py
  tests/integration/test_i4_2e_2_native_visibility_provenance_contract.py
  tests/integration/test_reconstruction_persistent_scene_f5.py
)

AFFECTED_STRESS=(
  tests/stress/test_i4_4a_offline_dense_foundation_stress.py
  tests/stress/test_i4_4b_cross_provider_visual_verification_stress.py
  tests/stress/test_i4_3h1_final_qualification_stress.py
  tests/stress/test_i4_3h2_final_closure_stress.py
  tests/stress/test_i4_3g5b_final_g_closure_stress.py
  tests/stress/test_i4_2e_2_native_visibility_provenance_stress.py
  tests/stress/test_i4_3a_dense_provider_architecture_stress.py
  tests/stress/test_reconstruction_persistent_scene_f5_stress.py
)

run_pytest_xml() {
  local xml="$1"
  shift
  python -m pytest -q --junitxml="$xml" "$@"
  local rc=$?
  echo "pytest_rc=$rc"
  return "$rc"
}

focused() { run_pytest_xml "$TMPDIR_I44C/focused.xml" "${FOCUSED[@]}"; }
affected_unit() { run_pytest_xml "$TMPDIR_I44C/unit.xml" "${AFFECTED_UNIT[@]}"; }
affected_integration() { run_pytest_xml "$TMPDIR_I44C/integration.xml" "${AFFECTED_INTEGRATION[@]}"; }
affected_stress() { run_pytest_xml "$TMPDIR_I44C/stress.xml" "${AFFECTED_STRESS[@]}"; }

probe() {
  python - <<'PY'
from core.reconstruction.offline_dense_provenance_authority_audit import (
    I44C_PHASE,
    audit_current_offline_dense_authority_contracts,
    build_i4_4c_provenance_authority_audit,
)
from tests.i4_4c_helpers import build_passing_i4_4c
assert I44C_PHASE == "I4.4C"
assert callable(build_i4_4c_provenance_authority_audit)
audit = audit_current_offline_dense_authority_contracts()
assert audit["camera_pose_authority_owner"] == "pycolmap_sfm"
assert audit["default_reconstruction_scale_status"] == "arbitrary"
assert audit["permanent_global_view_cap"] is None
result = build_passing_i4_4c()
assert result["decision"]["i4_4d_authorized"] is True
assert result["provenance_chain_audit"]["raw_nested_provider_artifact_tree_recursively_rehashed_by_i4_4c"] is False
print("I4.4C provenance/authority contract/import probe: PASS")
PY
  local rc=$?
  echo "import_probe_rc=$rc"
  [ "$rc" -eq 0 ] || return "$rc"
  python scripts/run_i4_4c_provenance_authority_audit.py --help >/dev/null
  rc=$?
  echo "audit_runner_help_rc=$rc"
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
  focused_count="$(xml_count "$TMPDIR_I44C/focused.xml")"
  unit_count="$(xml_count "$TMPDIR_I44C/unit.xml")"
  integration_count="$(xml_count "$TMPDIR_I44C/integration.xml")"
  stress_count="$(xml_count "$TMPDIR_I44C/stress.xml")"
  file_count=$(( ${#AFFECTED_UNIT[@]} + ${#AFFECTED_INTEGRATION[@]} + ${#AFFECTED_STRESS[@]} ))
  python - "$target" "$focused_count" "$unit_count" "$integration_count" "$stress_count" "$file_count" <<'PY'
from pathlib import Path
import json
import sys
path = Path(sys.argv[1]).expanduser().resolve()
focused, unit, integration, stress, file_count = map(int, sys.argv[2:])
payload = {
    "schema_version": 1,
    "phase": "I4.4C_PROVENANCE_AUTHORITY_AUDIT_RECEIPT",
    "status": "pass",
    "pass": True,
    "scope": {
        "focused_i4_4c_tests": True,
        "affected_provenance_unit_regression": True,
        "affected_provenance_integration_regression": True,
        "affected_provenance_stress_regression": True,
        "import_probe": True,
        "runner_help_probe": True,
        "provider_execution_started": False,
        "provider_runtime_probe_started": False,
        "sfm_rerun": False,
        "matched_view_rerun": False,
        "manual_review_rerun": False,
        "final_i4_closure_started": False,
    },
    "counts": {
        "focused_tests": focused,
        "affected_test_files": file_count,
        "affected_unit_tests": unit,
        "affected_integration_tests": integration,
        "affected_stress_tests": stress,
        "affected_total_tests": unit + integration + stress,
    },
    "return_codes": {
        "focused": 0,
        "affected_unit": 0,
        "affected_integration": 0,
        "affected_stress": 0,
        "import_probe": 0,
        "runner_help_probe": 0,
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"audit_receipt={path}")
PY
}

all_suites() {
  focused || return $?
  probe || return $?
  affected_unit || return $?
  affected_integration || return $?
  affected_stress || return $?
  if [ -n "$RECEIPT" ]; then
    write_receipt "$RECEIPT" || return $?
  fi
  echo "I4.4C_TEST_RC=0"
}

case "$MODE" in
  focused) focused ;;
  probe) probe ;;
  unit) affected_unit ;;
  integration) affected_integration ;;
  stress) affected_stress ;;
  regression) affected_unit && affected_integration && affected_stress ;;
  all) all_suites ;;
  *)
    echo "usage: $0 [focused|probe|unit|integration|stress|regression|all] [receipt.json]" >&2
    exit 2
    ;;
esac
