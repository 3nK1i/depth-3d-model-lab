#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"
TMPDIR_H1="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_H1"' EXIT

FOCUSED=(
  tests/unit/test_i4_3h1_final_qualification.py
  tests/integration/test_i4_3h1_final_qualification_contract.py
  tests/stress/test_i4_3h1_final_qualification_stress.py
)

mapfile -t PRE_UNIT < <(find tests/unit -maxdepth 1 -type f -name 'test_i4_3*.py' ! -name 'test_i4_3h1*.py' | sort)
mapfile -t PRE_INTEGRATION < <(find tests/integration -maxdepth 1 -type f -name 'test_i4_3*.py' ! -name 'test_i4_3h1*.py' | sort)
mapfile -t PRE_STRESS < <(find tests/stress -maxdepth 1 -type f -name 'test_i4_3*.py' ! -name 'test_i4_3h1*.py' | sort)

run_pytest_xml() {
  local xml="$1"
  shift
  python -m pytest -q --junitxml="$xml" "$@"
  local rc=$?
  echo "pytest_rc=$rc"
  return "$rc"
}

focused() {
  run_pytest_xml "$TMPDIR_H1/focused.xml" "${FOCUSED[@]}"
}

predecessor_unit() {
  run_pytest_xml "$TMPDIR_H1/unit.xml" "${PRE_UNIT[@]}"
}

predecessor_integration() {
  run_pytest_xml "$TMPDIR_H1/integration.xml" "${PRE_INTEGRATION[@]}"
}

predecessor_stress() {
  run_pytest_xml "$TMPDIR_H1/stress.xml" "${PRE_STRESS[@]}"
}

probe() {
  python - <<'PY'
from core.reconstruction.i4_3_final_qualification import (
    I43H1_PHASE,
    audit_i4_3_provider_architecture,
    build_i4_3h1_final_qualification_audit,
)
assert I43H1_PHASE == "I4.3H1"
assert callable(build_i4_3h1_final_qualification_audit)
audit = audit_i4_3_provider_architecture()
assert audit["provider_implementation_count"] == 10
print("I4.3H1 final qualification contract/import audit probe: PASS")
PY
  local rc=$?
  echo "import_probe_rc=$rc"
  [ "$rc" -eq 0 ] || return "$rc"
  python scripts/run_i4_3h1_final_qualification.py --help >/dev/null
  rc=$?
  echo "qualification_runner_help_rc=$rc"
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
  focused_count="$(xml_count "$TMPDIR_H1/focused.xml")"
  unit_count="$(xml_count "$TMPDIR_H1/unit.xml")"
  integration_count="$(xml_count "$TMPDIR_H1/integration.xml")"
  stress_count="$(xml_count "$TMPDIR_H1/stress.xml")"
  file_count=$(( ${#PRE_UNIT[@]} + ${#PRE_INTEGRATION[@]} + ${#PRE_STRESS[@]} ))
  python - "$target" "$focused_count" "$unit_count" "$integration_count" "$stress_count" "$file_count" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1]).expanduser().resolve()
focused, unit, integration, stress, file_count = map(int, sys.argv[2:])
payload = {
    "schema_version": 1,
    "phase": "I4.3H1_REGRESSION_AUDIT",
    "status": "pass",
    "pass": True,
    "scope": {
        "focused_h1_contract_tests": True,
        "predecessor_i4_3_unit_tests": True,
        "predecessor_i4_3_integration_tests": True,
        "predecessor_i4_3_stress_tests": True,
        "import_probe": True,
        "runner_help_probe": True,
    },
    "counts": {
        "focused_h1_tests": focused,
        "predecessor_test_file_count": file_count,
        "predecessor_unit_tests": unit,
        "predecessor_integration_tests": integration,
        "predecessor_stress_tests": stress,
        "predecessor_total_tests": unit + integration + stress,
    },
    "return_codes": {
        "focused_h1": 0,
        "predecessor_unit": 0,
        "predecessor_integration": 0,
        "predecessor_stress": 0,
        "import_probe": 0,
        "runner_help_probe": 0,
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"regression_receipt={path}")
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
  echo "I4.3H1_TEST_RC=0"
}

case "$MODE" in
  focused) focused ;;
  probe) probe ;;
  regression)
    predecessor_unit && predecessor_integration && predecessor_stress
    ;;
  all) all_suites ;;
  *)
    echo "usage: $0 [focused|probe|regression|all] [receipt.json]" >&2
    exit 2
    ;;
esac
