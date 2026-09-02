#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"
TMPDIR_I44B="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_I44B"' EXIT

FOCUSED=(
  tests/unit/test_i4_4b_cross_provider_visual_verification.py
  tests/integration/test_i4_4b_cross_provider_visual_verification_contract.py
  tests/stress/test_i4_4b_cross_provider_visual_verification_stress.py
)

AFFECTED_UNIT=(
  tests/unit/test_i4_4a_offline_dense_foundation_regression.py
  tests/unit/test_i4_3g2c_matched_view_review.py
  tests/unit/test_i4_3g2c_remaining_scenario_review.py
  tests/unit/test_i4_3g3c_temporal_matched_view_review.py
  tests/unit/test_i4_3g4c_single_image_matched_view_review.py
  tests/unit/test_i4_3g5a_cross_provider_evidence_synthesis.py
  tests/unit/test_i4_3g5b_cross_provider_qualification_closure.py
  tests/unit/test_i4_3h1_final_qualification.py
  tests/unit/test_i4_3h2_final_closure.py
)

AFFECTED_INTEGRATION=(
  tests/integration/test_i4_4a_offline_dense_foundation_contract.py
  tests/integration/test_i4_3g2c_entry_contract.py
  tests/integration/test_i4_3g2c_remaining_scenario_contract.py
  tests/integration/test_i4_3g3c_temporal_review_contract.py
  tests/integration/test_i4_3g4c_single_image_review_contract.py
  tests/integration/test_i4_3g5a_cross_provider_synthesis_contract.py
  tests/integration/test_i4_3g5b_final_g_closure_contract.py
  tests/integration/test_i4_3h1_final_qualification_contract.py
  tests/integration/test_i4_3h2_final_closure_contract.py
)

AFFECTED_STRESS=(
  tests/stress/test_i4_4a_offline_dense_foundation_stress.py
  tests/stress/test_i4_3g2c_review_stress.py
  tests/stress/test_i4_3g2c_remaining_scenario_stress.py
  tests/stress/test_i4_3g3c_temporal_review_stress.py
  tests/stress/test_i4_3g4c_single_image_review_stress.py
  tests/stress/test_i4_3g5a_cross_provider_synthesis_stress.py
  tests/stress/test_i4_3g5b_final_g_closure_stress.py
  tests/stress/test_i4_3h1_final_qualification_stress.py
  tests/stress/test_i4_3h2_final_closure_stress.py
)

run_pytest_xml() {
  local xml="$1"
  shift
  python -m pytest -q --junitxml="$xml" "$@"
  local rc=$?
  echo "pytest_rc=$rc"
  return "$rc"
}

focused() { run_pytest_xml "$TMPDIR_I44B/focused.xml" "${FOCUSED[@]}"; }
affected_unit() { run_pytest_xml "$TMPDIR_I44B/unit.xml" "${AFFECTED_UNIT[@]}"; }
affected_integration() { run_pytest_xml "$TMPDIR_I44B/integration.xml" "${AFFECTED_INTEGRATION[@]}"; }
affected_stress() { run_pytest_xml "$TMPDIR_I44B/stress.xml" "${AFFECTED_STRESS[@]}"; }

probe() {
  python - <<'PY'
from core.reconstruction.offline_dense_cross_provider_visual_verification import (
    I44B_PHASE,
    build_i4_4b_cross_provider_visual_verification,
)
from tests.i4_4b_helpers import build_passing_i4_4b
assert I44B_PHASE == "I4.4B"
assert callable(build_i4_4b_cross_provider_visual_verification)
result = build_passing_i4_4b()
assert result["coverage"]["provider_implementation_count"] == 10
assert result["cross_family_visual_verification"]["quality_event_counts_are_not_provider_scores"] is True
assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
assert result["decision"]["i4_4c_authorized"] is True
print("I4.4B cross-provider visual verification contract/import probe: PASS")
PY
  local rc=$?
  echo "import_probe_rc=$rc"
  [ "$rc" -eq 0 ] || return "$rc"
  python scripts/run_i4_4b_cross_provider_visual_verification.py --help >/dev/null
  rc=$?
  echo "visual_runner_help_rc=$rc"
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
  focused_count="$(xml_count "$TMPDIR_I44B/focused.xml")"
  unit_count="$(xml_count "$TMPDIR_I44B/unit.xml")"
  integration_count="$(xml_count "$TMPDIR_I44B/integration.xml")"
  stress_count="$(xml_count "$TMPDIR_I44B/stress.xml")"
  file_count=$(( ${#AFFECTED_UNIT[@]} + ${#AFFECTED_INTEGRATION[@]} + ${#AFFECTED_STRESS[@]} ))
  python - "$target" "$focused_count" "$unit_count" "$integration_count" "$stress_count" "$file_count" <<'PY'
from pathlib import Path
import json
import sys
path = Path(sys.argv[1]).expanduser().resolve()
focused, unit, integration, stress, file_count = map(int, sys.argv[2:])
payload = {
    "schema_version": 1,
    "phase": "I4.4B_VISUAL_VERIFICATION_RECEIPT",
    "status": "pass",
    "pass": True,
    "scope": {
        "focused_i4_4b_tests": True,
        "affected_visual_unit_regression": True,
        "affected_visual_integration_regression": True,
        "affected_visual_stress_regression": True,
        "import_probe": True,
        "runner_help_probe": True,
        "provider_execution_started": False,
        "sfm_rerun": False,
        "visual_review_rerun": False,
        "manual_review_rerun": False,
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
print(f"visual_receipt={path}")
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
  echo "I4.4B_TEST_RC=0"
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
