#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOCUSED=(
  tests/unit/test_i6_1b_controller_workspace_contract_static.py
  tests/integration/test_i6_1b_controller_workspace_plumbing.py
)
I6_1A_REGRESSION=(
  tests/unit/test_i6_1a_offline_reconstruction_workspace.py
  tests/integration/test_i6_1a_offline_workspace_contract.py
)
PURE_PREDECESSORS=(
  tests/unit/test_phase8_8g1_reconstruction_session_model.py
  tests/unit/test_phase8_8g2_reconstruction_session_interaction.py
  tests/unit/test_phase8_8g3_reconstruction_checkpoint_resume.py
  tests/unit/test_phase8_8g4_reconstruction_failure_recovery.py
)
CONTROLLER_PREDECESSORS=(
  tests/unit/test_phase8_8a1_reconstruction_controller.py
  tests/unit/test_phase8_8a2_reconstruction_controller.py
  tests/unit/test_phase8_8a3_reconstruction_controller.py
  tests/unit/test_phase8_8a5_controller_hardening.py
  tests/unit/test_phase8_8b2_reconstruction_source_selection.py
  tests/unit/test_phase8_8b3_reconstruction_runtime_controls.py
  tests/integration/test_phase8_8a3_offline_controller_process.py
  tests/integration/test_phase8_8a4_scene_checkpoint_bridge.py
  tests/integration/test_phase8_8a5_controller_hardening.py
  tests/ui/test_phase8_8g1_reconstruction_session_model.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
I6_1A_RC=0; I6_1A_COUNT=0
PURE_RC=0; PURE_COUNT=0
CONTROLLER_RC=0; CONTROLLER_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0
COMPILE_RC=0

run_pytest_count() {
  local rc_var="$1" count_var="$2" label="$3"; shift 3
  local tmp
  tmp="$(mktemp)"
  pytest -q "$@" >"$tmp" 2>&1
  local rc=$?
  cat "$tmp"
  local count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
print(sum(int(value) for value in re.findall(r"(\d+) passed", text)))
PY
)"
  fi
  rm -f "$tmp"
  printf '%s_RC=%s\n' "$label" "$rc"
  printf '%s_COUNT=%s\n' "$label" "$count"
  printf -v "$rc_var" '%s' "$rc"
  printf -v "$count_var" '%s' "$count"
}

probe_import() {
  python - <<'PY'
from app.application_controller import ApplicationController
from app.reconstruction_controller import ReconstructionController
from app.reconstruction_offline_workspace import OfflineReconstructionWorkspaceSnapshot

assert hasattr(ReconstructionController, "offline_workspace_changed")
assert isinstance(ReconstructionController.offline_workspace_snapshot, property)
assert OfflineReconstructionWorkspaceSnapshot.__dataclass_params__.frozen is True
print("I6.1B controller workspace/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_controller.py \
    app/reconstruction_offline_workspace.py \
    tests/unit/test_i6_1b_controller_workspace_contract_static.py \
    tests/integration/test_i6_1b_controller_workspace_plumbing.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
}

run_i6_1a() {
  run_pytest_count I6_1A_RC I6_1A_COUNT I6_1A_REGRESSION "${I6_1A_REGRESSION[@]}"
}

run_pure() {
  run_pytest_count PURE_RC PURE_COUNT PURE_PREDECESSOR "${PURE_PREDECESSORS[@]}"
}

run_controller() {
  run_pytest_count CONTROLLER_RC CONTROLLER_COUNT CONTROLLER_PREDECESSOR "${CONTROLLER_PREDECESSORS[@]}"
}

run_full_unit() {
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused)
    run_focused
    ;;
  i6-1a)
    run_i6_1a
    ;;
  predecessors-pure)
    run_pure
    ;;
  predecessors-controller)
    run_controller
    ;;
  predecessors)
    run_i6_1a
    run_pure
    run_controller
    ;;
  full-unit)
    run_full_unit
    ;;
  all)
    run_focused
    run_i6_1a
    run_pure
    run_controller
    ;;
  closure)
    run_focused
    run_i6_1a
    run_pure
    run_controller
    run_full_unit
    ;;
  *)
    echo "usage: $0 [focused|i6-1a|predecessors-pure|predecessors-controller|predecessors|full-unit|all|closure] [receipt.json]" >&2
    exit 2
    ;;
esac

RC=0
case "$MODE" in
  focused)
    for value in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  i6-1a)
    [[ "$I6_1A_RC" -eq 0 ]] || RC=1
    ;;
  predecessors-pure)
    [[ "$PURE_RC" -eq 0 ]] || RC=1
    ;;
  predecessors-controller)
    [[ "$CONTROLLER_RC" -eq 0 ]] || RC=1
    ;;
  predecessors)
    for value in "$I6_1A_RC" "$PURE_RC" "$CONTROLLER_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  full-unit)
    [[ "$FULL_UNIT_RC" -eq 0 ]] || RC=1
    ;;
  all)
    for value in "$FOCUSED_RC" "$I6_1A_RC" "$PURE_RC" "$CONTROLLER_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  closure)
    for value in "$FOCUSED_RC" "$I6_1A_RC" "$PURE_RC" "$CONTROLLER_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$RC" "$FOCUSED_RC" "$FOCUSED_COUNT" "$I6_1A_RC" "$I6_1A_COUNT" "$PURE_RC" "$PURE_COUNT" "$CONTROLLER_RC" "$CONTROLLER_COUNT" "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2]
(
    rc,
    focused_rc,
    focused_count,
    i6_1a_rc,
    i6_1a_count,
    pure_rc,
    pure_count,
    controller_rc,
    controller_count,
    full_unit_rc,
    full_unit_count,
    import_rc,
    compile_rc,
) = map(int, sys.argv[3:])

payload = {
    "schema_version": 1,
    "phase": "I6.1B_CONTROLLER_COMPOSITION_EVENT_PLUMBING",
    "mode": mode,
    "status": "pass" if rc == 0 else "fail",
    "pass": rc == 0,
    "scope": {
        "i6_1a_workspace_contract_reused": True,
        "workspace_composed_from_existing_authoritative_state": True,
        "workspace_signal_deduplicated": True,
        "offline_event_forwarding_preserved": True,
        "stale_progress_cleared_on_new_start": True,
        "identity_conflicts_preserved": True,
        "controller_modified": True,
        "ui_modified": False,
        "persistence_format_modified": False,
        "reconstruction_algorithm_modified": False,
        "provider_selection_modified": False,
        "production_auto_activated": False,
        "production_auto_qualified": False,
        "production_auto_authorized": False,
        "pycolmap_camera_authority_preserved": True,
        "certified_metrology_claimed": False,
        "production_default_modified": False,
        "threshold_modified": False,
    },
    "counts": {
        "focused_tests": focused_count,
        "i6_1a_regression_tests": i6_1a_count,
        "pure_predecessor_tests": pure_count,
        "controller_predecessor_tests": controller_count,
        "full_unit_tests": full_unit_count,
        "combined_selected_tests": focused_count + i6_1a_count + pure_count + controller_count,
    },
    "return_codes": {
        "focused": focused_rc,
        "i6_1a_regression": i6_1a_rc,
        "pure_predecessor": pure_rc,
        "controller_predecessor": controller_rc,
        "full_unit": full_unit_rc,
        "import_probe": import_rc,
        "compile_probe": compile_rc,
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"I6.1B receipt={path}")
PY
fi

echo "I6.1B_TEST_RC=$RC"
exit "$RC"
