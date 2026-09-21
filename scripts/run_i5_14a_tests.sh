#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

focused=(
  tests/unit/test_i5_14a_final_qualification_readiness.py
  tests/integration/test_i5_14a_final_qualification_readiness_contract.py
  tests/stress/test_i5_14a_final_qualification_readiness_stress.py
)
pre_unit=(
  tests/unit/test_i5_1c_difficult_handheld_closure.py
  tests/unit/test_i5_2c_low_overlap_weak_parallax_closure.py
  tests/unit/test_i5_3c_blur_exposure_clutter_symmetry_closure.py
  tests/unit/test_i5_4c_forward_motion_kitti_closure.py
  tests/unit/test_i5_5c_dynamic_scene_closure.py
  tests/unit/test_i5_6c_mvs_robustness_closure.py
  tests/unit/test_i5_7c_rolling_depth_robustness_closure.py
  tests/unit/test_i5_8c_video_depth_anything_robustness_closure.py
  tests/unit/test_i5_9c_remaining_depth_provider_robustness_closure.py
  tests/unit/test_i5_10d_mp_sfm_closure.py
  tests/unit/test_i5_11d_failure_robustness_closure.py
  tests/unit/test_i5_12a_hybrid_admission.py
  tests/unit/test_i5_13d_auto_production_integration_decision.py
)
pre_int=(
  tests/integration/test_i5_1c_difficult_handheld_contract.py
  tests/integration/test_i5_2c_low_overlap_weak_parallax_contract.py
  tests/integration/test_i5_3c_blur_exposure_clutter_symmetry_contract.py
  tests/integration/test_i5_4c_forward_motion_kitti_contract.py
  tests/integration/test_i5_5c_dynamic_scene_contract.py
  tests/integration/test_i5_6c_mvs_robustness_contract.py
  tests/integration/test_i5_7c_rolling_depth_robustness_contract.py
  tests/integration/test_i5_8c_video_depth_anything_robustness_contract.py
  tests/integration/test_i5_9c_remaining_depth_provider_robustness_contract.py
  tests/integration/test_i5_10d_mp_sfm_closure_contract.py
  tests/integration/test_i5_11d_failure_robustness_closure_contract.py
  tests/integration/test_i5_12a_hybrid_admission_contract.py
  tests/integration/test_i5_13d_auto_production_integration_decision_contract.py
)
pre_stress=(
  tests/stress/test_i5_1c_difficult_handheld_stress.py
  tests/stress/test_i5_2c_low_overlap_weak_parallax_stress.py
  tests/stress/test_i5_3c_blur_exposure_clutter_symmetry_stress.py
  tests/stress/test_i5_4c_forward_motion_kitti_stress.py
  tests/stress/test_i5_5c_dynamic_scene_stress.py
  tests/stress/test_i5_6c_mvs_robustness_stress.py
  tests/stress/test_i5_7c_rolling_depth_robustness_stress.py
  tests/stress/test_i5_8c_video_depth_anything_robustness_stress.py
  tests/stress/test_i5_9c_remaining_depth_provider_robustness_stress.py
  tests/stress/test_i5_10d_mp_sfm_closure_stress.py
  tests/stress/test_i5_11d_failure_robustness_closure_stress.py
  tests/stress/test_i5_12a_hybrid_admission_stress.py
  tests/stress/test_i5_13d_auto_production_integration_decision_stress.py
)

case "$MODE" in
  focused)
    pytest -q "${focused[@]}"; exit $?
    ;;
  all)
    TMPDIR_I514A="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_I514A"' EXIT
    set +e
    pytest -q "${focused[@]}" | tee "$TMPDIR_I514A/focused.log"; FOCUSED_RC=${PIPESTATUS[0]}
    pytest -q "${pre_unit[@]}" | tee "$TMPDIR_I514A/punit.log"; PUNIT_RC=${PIPESTATUS[0]}
    pytest -q "${pre_int[@]}" | tee "$TMPDIR_I514A/pint.log"; PINT_RC=${PIPESTATUS[0]}
    pytest -q "${pre_stress[@]}" | tee "$TMPDIR_I514A/pstress.log"; PSTRESS_RC=${PIPESTATUS[0]}
    python - <<'PY2'
from core.reconstruction.i5_final_qualification_readiness import I514A_CLOSURE_SPECS
assert len(I514A_CLOSURE_SPECS) == 13
assert I514A_CLOSURE_SPECS["i5_13d"]["quality"] == "offline_auto_qualified_production_integration_deferred"
print("I5.14A final qualification/readiness contract/import probe: PASS")
PY2
    IMPORT_RC=$?
    python scripts/run_i5_14a_final_qualification_readiness.py --help >/dev/null; HELP_RC=$?
    set -e

    echo "FOCUSED_RC=$FOCUSED_RC"
    echo "PUNIT_RC=$PUNIT_RC"
    echo "PINT_RC=$PINT_RC"
    echo "PSTRESS_RC=$PSTRESS_RC"
    echo "import_probe_rc=$IMPORT_RC"
    echo "readiness_runner_help_rc=$HELP_RC"

    extract_count() { local value; value="$(grep -Eo '[0-9]+ passed' "$1" | tail -n 1 | awk '{print $1}')"; [[ "$value" =~ ^[0-9]+$ ]] || value=0; printf '%s' "$value"; }
    FOCUSED_COUNT="$(extract_count "$TMPDIR_I514A/focused.log")"
    PUNIT_COUNT="$(extract_count "$TMPDIR_I514A/punit.log")"
    PINT_COUNT="$(extract_count "$TMPDIR_I514A/pint.log")"
    PSTRESS_COUNT="$(extract_count "$TMPDIR_I514A/pstress.log")"
    PRE=$((PUNIT_COUNT + PINT_COUNT + PSTRESS_COUNT))
    COMBINED=$((FOCUSED_COUNT + PRE))
    RC=0
    for value in "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC"; do [[ "$value" -eq 0 ]] || RC=1; done

    if [[ -n "$RECEIPT" ]]; then
      mkdir -p "$(dirname "$RECEIPT")"
      python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$PRE" "$COMBINED" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY2'
import json, sys
(path, focused, pu, pi, ps, pred, combined, frc, urc, irc, src, imprc, helprc) = sys.argv[1:]
all_zero = all(int(x) == 0 for x in (frc, urc, irc, src, imprc, helprc))
payload = {
  "schema_version": 1,
  "phase": "I5.14A_FINAL_QUALIFICATION_READINESS_RECEIPT",
  "status": "pass" if all_zero else "failed",
  "pass": all_zero,
  "scope": {
    "focused_i5_14a_tests": True, "i5_closure_regression": True,
    "evidence_only_readiness_review": True, "import_probe": True, "runner_help_probe": True,
    "provider_execution": False, "sfm_execution": False, "mvs_execution": False,
    "scene_classifier_execution": False, "production_auto_integration": False,
    "production_default_modified": False, "provider_promoted": False,
  },
  "counts": {
    "focused_tests": int(focused), "predecessor_unit_tests": int(pu),
    "predecessor_integration_tests": int(pi), "predecessor_stress_tests": int(ps),
    "predecessor_total_tests": int(pred), "combined_exercised_tests": int(combined),
  },
  "return_codes": {
    "focused": int(frc), "predecessor_unit": int(urc), "predecessor_integration": int(irc),
    "predecessor_stress": int(src), "import_probe": int(imprc), "runner_help_probe": int(helprc),
  },
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True); f.write("\n")
print(f"readiness_receipt={path}")
PY2
    fi
    echo "FOCUSED_COUNT=$FOCUSED_COUNT"
    echo "PUNIT_COUNT=$PUNIT_COUNT"
    echo "PINT_COUNT=$PINT_COUNT"
    echo "PSTRESS_COUNT=$PSTRESS_COUNT"
    echo "I5.14A_TEST_RC=$RC"
    exit "$RC"
    ;;
  *) echo "usage: $0 [focused|all] [receipt.json]" >&2; exit 2 ;;
esac
