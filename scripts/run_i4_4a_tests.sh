#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"
TMPDIR_I44A="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_I44A"' EXIT

FOCUSED=(
  tests/unit/test_i4_4a_offline_dense_foundation_regression.py
  tests/integration/test_i4_4a_offline_dense_foundation_contract.py
  tests/stress/test_i4_4a_offline_dense_foundation_stress.py
)

AFFECTED_UNIT=(
  tests/unit/test_reconstruction_sfm_contracts.py
  tests/unit/test_reconstruction_sfm_dense_alignment.py
  tests/unit/test_reconstruction_sfm_dense_projection.py
  tests/unit/test_reconstruction_dense_fusion.py
  tests/unit/test_g_persistent_dense_rgb_geometry.py
  tests/unit/test_reconstruction_persistent_scene_f5.py
  tests/unit/test_i4_3a_dense_provider_architecture.py
  tests/unit/test_i4_3f1_adaptive_view_selection.py
  tests/unit/test_i4_3f2_adaptive_production_planning.py
  tests/unit/test_i4_3f2_production_source_contract.py
  tests/unit/test_i4_3h2_final_closure.py
  tests/unit/test_i4_3b_mvs_provider.py
  tests/unit/test_i4_3c_da3_dense_provider.py
  tests/unit/test_i4_3d1_video_depth_anything_dense_provider.py
  tests/unit/test_i4_3d2_rolling_depth_dense_provider.py
  tests/unit/test_i4_3e1_metric3d_v2_dense_provider.py
  tests/unit/test_i4_3e2_unidepth_v2_dense_provider.py
  tests/unit/test_i4_3e3_moge2_dense_provider.py
  tests/unit/test_i4_3e4_depth_pro_dense_provider.py
  tests/unit/test_i4_3e5_depth_anything_v2_dense_provider.py
  tests/unit/test_i4_3e6_marigold_dense_provider.py
)

AFFECTED_INTEGRATION=(
  tests/integration/test_reconstruction_dense_fusion_pipeline.py
  tests/integration/test_reconstruction_persistent_scene_f5.py
  tests/integration/test_phase9_3a_canonical_scene_compatibility.py
  tests/integration/test_i4_3a_dense_provider_architecture_contract.py
  tests/integration/test_i4_3f1_adaptive_view_selection_contract.py
  tests/integration/test_i4_3f2_production_wiring_contract.py
  tests/integration/test_i4_3h2_final_closure_contract.py
  tests/integration/test_i4_3b_mvs_provider_contract.py
  tests/integration/test_i4_3c_da3_dense_provider_contract.py
  tests/integration/test_i4_3d1_video_depth_anything_provider_contract.py
  tests/integration/test_i4_3d2_rolling_depth_provider_contract.py
  tests/integration/test_i4_3e1_metric3d_v2_provider_contract.py
  tests/integration/test_i4_3e2_unidepth_v2_provider_contract.py
  tests/integration/test_i4_3e3_moge2_provider_contract.py
  tests/integration/test_i4_3e4_depth_pro_provider_contract.py
  tests/integration/test_i4_3e5_depth_anything_v2_provider_contract.py
  tests/integration/test_i4_3e6_marigold_provider_contract.py
)

AFFECTED_STRESS=(
  tests/stress/test_reconstruction_dense_fusion_stress.py
  tests/stress/test_reconstruction_persistent_scene_f5_stress.py
  tests/stress/test_i4_3a_dense_provider_architecture_stress.py
  tests/stress/test_i4_3f1_adaptive_view_selection_stress.py
  tests/stress/test_i4_3f2_production_wiring_stress.py
  tests/stress/test_i4_3h2_final_closure_stress.py
  tests/stress/test_i4_3b_mvs_provider_stress.py
  tests/stress/test_i4_3c_da3_dense_provider_stress.py
  tests/stress/test_i4_3d1_video_depth_anything_provider_stress.py
  tests/stress/test_i4_3d2_rolling_depth_provider_stress.py
  tests/stress/test_i4_3e1_metric3d_v2_provider_stress.py
  tests/stress/test_i4_3e2_unidepth_v2_provider_stress.py
  tests/stress/test_i4_3e3_moge2_provider_stress.py
  tests/stress/test_i4_3e4_depth_pro_provider_stress.py
  tests/stress/test_i4_3e5_depth_anything_v2_provider_stress.py
  tests/stress/test_i4_3e6_marigold_provider_stress.py
)

run_pytest_xml() {
  local xml="$1"
  shift
  python -m pytest -q --junitxml="$xml" "$@"
  local rc=$?
  echo "pytest_rc=$rc"
  return "$rc"
}

focused() { run_pytest_xml "$TMPDIR_I44A/focused.xml" "${FOCUSED[@]}"; }
affected_unit() { run_pytest_xml "$TMPDIR_I44A/unit.xml" "${AFFECTED_UNIT[@]}"; }
affected_integration() { run_pytest_xml "$TMPDIR_I44A/integration.xml" "${AFFECTED_INTEGRATION[@]}"; }
affected_stress() { run_pytest_xml "$TMPDIR_I44A/stress.xml" "${AFFECTED_STRESS[@]}"; }

probe() {
  python - <<'PY'
from core.reconstruction.offline_dense_foundation_regression import (
    I44A_PHASE,
    audit_offline_dense_foundation_contracts,
    build_i4_4a_focused_regression,
)
assert I44A_PHASE == "I4.4A"
assert callable(build_i4_4a_focused_regression)
audit = audit_offline_dense_foundation_contracts()
assert audit["provider_architecture"]["provider_implementation_count"] == 10
assert audit["sfm_authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
assert audit["adaptive_planning"]["permanent_global_view_cap"] is None
assert audit["heavy_execution_started"] is False
print("I4.4A offline dense foundation contract/import probe: PASS")
PY
  local rc=$?
  echo "import_probe_rc=$rc"
  [ "$rc" -eq 0 ] || return "$rc"
  python scripts/run_i4_4a_focused_regression.py --help >/dev/null
  rc=$?
  echo "regression_runner_help_rc=$rc"
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
  focused_count="$(xml_count "$TMPDIR_I44A/focused.xml")"
  unit_count="$(xml_count "$TMPDIR_I44A/unit.xml")"
  integration_count="$(xml_count "$TMPDIR_I44A/integration.xml")"
  stress_count="$(xml_count "$TMPDIR_I44A/stress.xml")"
  file_count=$(( ${#AFFECTED_UNIT[@]} + ${#AFFECTED_INTEGRATION[@]} + ${#AFFECTED_STRESS[@]} ))
  python - "$target" "$focused_count" "$unit_count" "$integration_count" "$stress_count" "$file_count" <<'PY'
from pathlib import Path
import json
import sys
path = Path(sys.argv[1]).expanduser().resolve()
focused, unit, integration, stress, file_count = map(int, sys.argv[2:])
payload = {
    "schema_version": 1,
    "phase": "I4.4A_REGRESSION_RECEIPT",
    "status": "pass",
    "pass": True,
    "scope": {
        "focused_i4_4a_tests": True,
        "affected_unit_regression": True,
        "affected_integration_regression": True,
        "affected_stress_regression": True,
        "import_probe": True,
        "runner_help_probe": True,
        "provider_execution_started": False,
        "sfm_rerun": False,
        "visual_verification_started": False,
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
print(f"regression_receipt={path}")
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
  echo "I4.4A_TEST_RC=0"
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
