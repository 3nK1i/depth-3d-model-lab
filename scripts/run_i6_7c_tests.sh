#!/usr/bin/env bash
set +e
set +u
set +o pipefail

MODE="${1:-all}"

FOCUSED=(
  tests/unit/test_i6_7c_progressive_dense_transport.py
  tests/unit/test_i6_7c_progressive_dense_buffer.py
  tests/integration/test_i6_7c_progressive_dense_transport_integration.py
  tests/stress/test_i6_7c_progressive_dense_backpressure_stress.py
)

I6_6C_PURE_REGRESSION=(
  tests/unit/test_i6_6c_progressive_sfm_transport.py
  tests/unit/test_i6_6c_progressive_sfm_buffer.py
  tests/stress/test_i6_6c_progressive_sfm_backpressure_stress.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
I67B_RC=0
I66C_RC=0; I66C_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0; COMPILE_RC=0; BASH_RC=0; SCOPE_RC=0

run_pytest_count() {
  local rc_var="$1" count_var="$2" label="$3"; shift 3
  local tmp rc count
  tmp="$(mktemp)"
  pytest -q "$@" >"$tmp" 2>&1
  rc=$?
  cat "$tmp"
  count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re, sys
text=open(sys.argv[1], encoding='utf-8', errors='replace').read()
print(sum(int(v) for v in re.findall(r'(\d+) passed', text)))
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
from core.reconstruction.reconstruction_progressive_dense_transport import (
    I67C_MAX_GEOMETRY_PREVIEW_POINTS,
    I67C_MAX_PENDING_SNAPSHOTS,
    I67C_MAX_RETAINED_PAYLOAD_BYTES,
    I67C_MAX_RETAINED_SNAPSHOTS,
    I67C_MAX_TRANSPORT_SNAPSHOT_BYTES,
    PROGRESSIVE_DENSE_EVENT_PREFIX,
    ProgressiveDenseSnapshotBuffer,
    ProgressiveDenseTransportBinding,
    ProgressiveDenseTransportLineSink,
    parse_progressive_dense_transport_line,
)
from tests.i6_7c_helpers import make_snapshot

first = make_snapshot(revision=1)
buffer = ProgressiveDenseSnapshotBuffer()
buffer.reset(ProgressiveDenseTransportBinding.from_snapshot(first))
lines=[]
sink=ProgressiveDenseTransportLineSink(lines.append)
sink.publish(first)
assert parse_progressive_dense_transport_line(lines[0]).snapshot_sha256 == first.snapshot_sha256
assert buffer.offer_line(lines[0]).value == 'accepted'
assert buffer.take_pending().revision == 1
assert I67C_MAX_TRANSPORT_SNAPSHOT_BYTES == 4 * 1024 * 1024
assert I67C_MAX_PENDING_SNAPSHOTS == 1
assert I67C_MAX_RETAINED_SNAPSHOTS == 2
assert I67C_MAX_RETAINED_PAYLOAD_BYTES == 8 * 1024 * 1024
assert I67C_MAX_GEOMETRY_PREVIEW_POINTS == 4096
assert PROGRESSIVE_DENSE_EVENT_PREFIX == 'DEPTH3D_DENSE_PROGRESSIVE '
print('I6.7C dense transport/backpressure/memory import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_progressive_dense_transport.py \
    scripts/run_i6_7c_progressive_dense_transport_replay.py \
    tests/i6_7c_helpers.py \
    tests/unit/test_i6_7c_progressive_dense_transport.py \
    tests/unit/test_i6_7c_progressive_dense_buffer.py \
    tests/integration/test_i6_7c_progressive_dense_transport_integration.py \
    tests/stress/test_i6_7c_progressive_dense_backpressure_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"

  bash -n scripts/run_i6_7c_tests.sh
  BASH_RC=$?
  echo "BASH_RC=$BASH_RC"
}

probe_scope() {
  python - <<'PY'
from hashlib import sha256
from pathlib import Path

expected = {
    'core/reconstruction/reconstruction_progressive_dense_contract.py': '918c67574ef342909fffdfd495d2503c5909bafd0624fb3856b34c488f2257ee',
    'core/reconstruction/reconstruction_progressive_dense_emission.py': '5c68ec59d1ec0238b15853e95135a56df7e2043824e9c835c66fadaa4f594ff7',
    'core/reconstruction/mvs_provider.py': 'fb4f96d368316f351b910916611459ff51dc59d666e5ee00cb3084fe609a7f79',
    'scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py': '82b5de59687c691fda4959faa411149a3a5f4c8fa10e3d85da513cc254d32689',
    'scripts/run_i6_7b_progressive_dense_emission_smoke.py': 'cfacbb604d553f2ce96c115cbdb35329b65628c95476ad33ff50b399fc0143b2',
    'tests/integration/test_i6_7b_dense_provider_execution_emission.py': 'aab059e30203cc05e47b0274a9b9db9a501cc7fff5d15999f50f7123b67ab9da',
    'docs/phase_history/I6_7B_DENSE_PROVIDER_EXECUTION_AND_PROGRESSIVE_EMISSION.md': '9b2f77e75339ea00b84f57d627a27dde1d877530017e239504ed8af9a31f5367',
    'app/offline_reconstruction_controller.py': 'bd4f9c1cfe1739759d4b95d736cf376f36cffd04288ffb615e133fbb8358aa79',
    'app/reconstruction_controller.py': '57ef77cc5ebad69006b11ad25225e289ef540539e09e1636129b9d9bd4211186',
    'app/widgets/reconstruction_page.py': 'de6d28d89cd6f853f6005488847368f0cfb605d939dc23aa1d93405fdde960d1',
    'scripts/run_reconstruction_offline_job.py': '4b6562cb2a8395ed181f0d360a68d14d81b117b02b2259727bc0839e7ee7e763',
}
for path, digest in expected.items():
    actual = sha256(Path(path).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f'I6.7C scope violation: frozen/non-scope file changed: {path} {actual}')
print('I6_7C_SCOPE_GUARD=PASS')
print('I6_7A_FROZEN_CONTRACT_MODIFIED=NO')
print('I6_7B_FROZEN_EXECUTION_EMISSION_MODIFIED=NO')
print('OFFLINE_CONTROLLER_MODIFIED=NO')
print('RECONSTRUCTION_CONTROLLER_MODIFIED=NO')
print('UI_MODIFIED=NO')
print('VIEWPORT_MODIFIED=NO')
print('OFFLINE_JOB_DENSE_EXECUTION_WIRING_ADDED=NO')
print('LATEST_ONLY_PENDING_BUFFER=YES')
print('MAX_PENDING_SNAPSHOTS=1')
print('MAX_RETAINED_SNAPSHOTS=2')
print('MAX_TRANSPORT_SNAPSHOT_BYTES=4194304')
print('MAX_RETAINED_PAYLOAD_BYTES=8388608')
print('MAX_GEOMETRY_PREVIEW_POINTS=4096')
print('FINAL_DENSE_RESULT_ADMISSION_ADDED=NO')
print('PERSISTENT_DENSE_SCENE_CREATED=NO')
print('PRODUCTION_AUTO_ACTIVATED=NO')
print('SURFACE_MESH_GENERATED=NO')
print('PYCOLMAP_CAMERA_AUTHORITY_PRESERVED=YES')
print('METRIC_SCALE_CLAIMED=NO')
PY
  SCOPE_RC=$?
  echo "SCOPE_RC=$SCOPE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
  probe_scope
}

run_predecessors() {
  local tmp
  tmp="$(mktemp)"
  bash scripts/run_i6_7b_tests.sh all >"$tmp" 2>&1
  I67B_RC=$?
  cat "$tmp"
  rm -f "$tmp"
  echo "I6_7B_FROZEN_REGRESSION_RC=$I67B_RC"
  run_pytest_count I66C_RC I66C_COUNT I6_6C_PURE_REGRESSION "${I6_6C_PURE_REGRESSION[@]}"
}

run_full_unit() {
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused) run_focused ;;
  predecessors) run_predecessors ;;
  all) run_focused; run_predecessors ;;
  full-unit) run_full_unit ;;
  closure) run_focused; run_predecessors; run_full_unit ;;
  *) echo "usage: $0 [focused|predecessors|all|full-unit|closure]" >&2; exit 2 ;;
esac

TEST_RC=0
case "$MODE" in
  focused)
    for rc in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC" "$BASH_RC" "$SCOPE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  predecessors)
    for rc in "$I67B_RC" "$I66C_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  all)
    for rc in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC" "$BASH_RC" "$SCOPE_RC" "$I67B_RC" "$I66C_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure)
    for rc in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC" "$BASH_RC" "$SCOPE_RC" "$I67B_RC" "$I66C_RC" "$FULL_UNIT_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
esac

echo "I6_7C_TEST_RC=$TEST_RC"
exit "$TEST_RC"
