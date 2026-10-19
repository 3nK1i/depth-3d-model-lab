#!/usr/bin/env bash
# I6.7D bounded incremental viewer qualification; no provider execution.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
MODE="${1:-focused}"
if [[ "$MODE" != "focused" && "$MODE" != "closure" ]]; then
  echo 'Usage: bash scripts/run_i6_7d_tests.sh focused|closure' >&2
  exit 2
fi

# Immutable prior source checks: this phase must not edit frozen I6.7A/B/C.
python - <<'PY'
from pathlib import Path
import hashlib
expected = {
 'core/reconstruction/reconstruction_progressive_dense_contract.py': '918c67574ef342909fffdfd495d2503c5909bafd0624fb3856b34c488f2257ee',
 'core/reconstruction/reconstruction_progressive_dense_emission.py': '5c68ec59d1ec0238b15853e95135a56df7e2043824e9c835c66fadaa4f594ff7',
 'core/reconstruction/reconstruction_progressive_dense_transport.py': '5be5d9bca0b564c2454deccff5530b3f483ca7ae45425a4394c05a0efa968074',
 'app/reconstruction_progressive_sfm_viewer.py': 'cda2e1da490a83a9af0875a95d4ac3672f9637b3dd616a7e6738911702864d76',
}
for name, sha in expected.items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != sha:
        raise SystemExit('FROZEN_PREDECESSOR_MISMATCH=' + name)
    print('FROZEN_SOURCE=PASS ' + name)
print('I6_7D_PREDECESSOR_SCOPE=PASS')
PY
SCOPE_RC=$?
echo "SCOPE_RC=$SCOPE_RC"
if (( SCOPE_RC != 0 )); then exit "$SCOPE_RC"; fi

python -m py_compile \
 app/reconstruction_progressive_dense_viewer.py \
 app/widgets/reconstruction_page.py \
 app/widgets/viewport/vispy_canvas.py \
 tests/unit/test_i6_7d_progressive_dense_viewer_projection.py \
 tests/stress/test_i6_7d_progressive_dense_viewer_projection_stress.py \
 tests/ui/test_i6_7d_incremental_dense_viewer_ui.py
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"
if (( COMPILE_RC != 0 )); then exit "$COMPILE_RC"; fi

python -m pytest -q \
 tests/unit/test_i6_7d_progressive_dense_viewer_projection.py \
 tests/stress/test_i6_7d_progressive_dense_viewer_projection_stress.py \
 tests/ui/test_i6_7d_incremental_dense_viewer_ui.py
FOCUSED_RC=$?
echo "I6_7D_FOCUSED_RC=$FOCUSED_RC"
if (( FOCUSED_RC != 0 )); then exit "$FOCUSED_RC"; fi
if [[ "$MODE" == "focused" ]]; then echo I6_7D_TEST_RC=0; exit 0; fi

# Do not call I6.7C's archived closure wrapper: it intentionally guards against
# viewport changes, whereas I6.7D is precisely the authorized viewport step.
python -m pytest -q \
 tests/unit/test_i6_7c_progressive_dense_transport.py \
 tests/unit/test_i6_7c_progressive_dense_buffer.py \
 tests/integration/test_i6_7c_progressive_dense_transport_integration.py \
 tests/stress/test_i6_7c_progressive_dense_backpressure_stress.py \
 tests/unit/test_i6_7b_progressive_dense_emission.py \
 tests/integration/test_i6_7b_dense_provider_execution_emission.py \
 tests/stress/test_i6_7b_progressive_dense_emission_stress.py \
 tests/unit/test_i6_7a_progressive_dense_snapshot_contract.py \
 tests/integration/test_i6_7a_progressive_dense_contract_integration.py \
 tests/stress/test_i6_7a_progressive_dense_contract_stress.py \
 tests/ui/test_i6_6d_incremental_sparse_viewer_ui.py \
 tests/ui/test_i6_6e_progressive_sfm_ux_ui.py
PREDECESSOR_RC=$?
echo "I6_7D_PREDECESSOR_RC=$PREDECESSOR_RC"
if (( PREDECESSOR_RC != 0 )); then exit "$PREDECESSOR_RC"; fi

python - <<'PY'
import shutil
free=shutil.disk_usage('/tmp').free
print(f'FREE_DISK_GIB={free / 1024**3:.3f}')
if free < 8*1024**3:
    raise SystemExit('STORAGE_PREFLIGHT=BLOCKED: restore disk safety floor before full tests')
print('STORAGE_PREFLIGHT=PASS')
PY
STORAGE_RC=$?
if (( STORAGE_RC != 0 )); then exit "$STORAGE_RC"; fi

python -m pytest -q tests/unit
FULL_UNIT_RC=$?
echo "FULL_UNIT_RC=$FULL_UNIT_RC"
if (( FULL_UNIT_RC != 0 )); then exit "$FULL_UNIT_RC"; fi

echo I6_7D_TEST_RC=0
