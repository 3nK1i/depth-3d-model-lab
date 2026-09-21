#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOCUSED=(
  tests/unit/test_i6_1a_offline_reconstruction_workspace.py
  tests/integration/test_i6_1a_offline_workspace_contract.py
)
PURE_PREDECESSORS=(
  tests/unit/test_phase8_8g1_reconstruction_session_model.py
  tests/unit/test_phase8_8g2_reconstruction_session_interaction.py
  tests/unit/test_phase8_8g3_reconstruction_checkpoint_resume.py
  tests/unit/test_phase8_8g4_reconstruction_failure_recovery.py
)
QT_PREDECESSORS=(
  tests/unit/test_phase8_8a4_reconstruction_persistence_controller.py
  tests/integration/test_phase8_8a4_scene_checkpoint_bridge.py
  tests/ui/test_phase8_8g1_reconstruction_session_model.py
  tests/ui/test_phase8_8g2_session_save_reload_interaction.py
  tests/ui/test_phase8_8g3_checkpoint_resume_ux.py
  tests/ui/test_phase8_8g4_failure_recovery_ux.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
PURE_RC=0; PURE_COUNT=0
QT_RC=0; QT_COUNT=0
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
from app.reconstruction_offline_workspace import (
    CANONICAL_CAMERA_AUTHORITY_OWNER,
    PRODUCTION_AUTO_POLICY_SCOPE,
    OfflineReconstructionWorkspaceSnapshot,
    OfflineWorkspaceEvidenceIdentityStatus,
    OfflineWorkspaceScaleStatus,
    build_offline_reconstruction_workspace_snapshot,
)
assert CANONICAL_CAMERA_AUTHORITY_OWNER == "pycolmap_sfm"
assert PRODUCTION_AUTO_POLICY_SCOPE == "offline_shadow_only"
assert OfflineReconstructionWorkspaceSnapshot.__dataclass_params__.frozen is True
assert OfflineWorkspaceEvidenceIdentityStatus.CONFLICT.value == "conflict"
assert OfflineWorkspaceScaleStatus.ARBITRARY.value == "arbitrary"
assert callable(build_offline_reconstruction_workspace_snapshot)
print("I6.1A offline workspace contract/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_offline_workspace.py \
    tests/unit/test_i6_1a_offline_reconstruction_workspace.py \
    tests/integration/test_i6_1a_offline_workspace_contract.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
}

run_pure() {
  run_pytest_count PURE_RC PURE_COUNT PURE_PREDECESSOR "${PURE_PREDECESSORS[@]}"
}

run_qt() {
  run_pytest_count QT_RC QT_COUNT QT_PREDECESSOR "${QT_PREDECESSORS[@]}"
}

case "$MODE" in
  focused)
    run_focused
    ;;
  predecessors-pure)
    run_pure
    ;;
  predecessors-qt)
    run_qt
    ;;
  predecessors)
    run_pure
    run_qt
    ;;
  all)
    run_focused
    run_pure
    run_qt
    ;;
  *)
    echo "usage: $0 [focused|predecessors-pure|predecessors-qt|predecessors|all] [receipt.json]" >&2
    exit 2
    ;;
esac

RC=0
case "$MODE" in
  focused)
    for value in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  predecessors-pure)
    [[ "$PURE_RC" -eq 0 ]] || RC=1
    ;;
  predecessors-qt)
    [[ "$QT_RC" -eq 0 ]] || RC=1
    ;;
  predecessors)
    for value in "$PURE_RC" "$QT_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  all)
    for value in "$FOCUSED_RC" "$PURE_RC" "$QT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$RC" "$FOCUSED_RC" "$FOCUSED_COUNT" "$PURE_RC" "$PURE_COUNT" "$QT_RC" "$QT_COUNT" "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2]
rc, focused_rc, focused_count, pure_rc, pure_count, qt_rc, qt_count, import_rc, compile_rc = map(int, sys.argv[3:])
payload = {
    "schema_version": 1,
    "phase": "I6.1A_OFFLINE_WORKSPACE_CONTRACT",
    "mode": mode,
    "status": "pass" if rc == 0 else "fail",
    "pass": rc == 0,
    "scope": {
        "dependency_light_workspace_contract": True,
        "existing_session_model_composed_not_replaced": True,
        "controller_modified": False,
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
        "pure_predecessor_tests": pure_count,
        "qt_predecessor_tests": qt_count,
        "combined_tests": focused_count + pure_count + qt_count,
    },
    "return_codes": {
        "focused": focused_rc,
        "pure_predecessor": pure_rc,
        "qt_predecessor": qt_rc,
        "import_probe": import_rc,
        "compile_probe": compile_rc,
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"I6.1A receipt={path}")
PY
fi

echo "I6.1A_TEST_RC=$RC"
exit "$RC"
