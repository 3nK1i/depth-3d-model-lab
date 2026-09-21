#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED=(
  tests/unit/test_i6_7a_progressive_dense_snapshot_contract.py
  tests/integration/test_i6_7a_progressive_dense_contract_integration.py
  tests/stress/test_i6_7a_progressive_dense_contract_stress.py
)

DENSE_PREDECESSOR_REGRESSION=(
  tests/unit/test_i4_3a_dense_provider_architecture.py
  tests/integration/test_i4_3a_dense_provider_architecture_contract.py
  tests/stress/test_i4_3a_dense_provider_architecture_stress.py
  tests/unit/test_g_persistent_dense_rgb_geometry.py
  tests/unit/test_h_dense_rgb_reconstruction_ui.py
)

I6_4_PROVIDER_BOUNDARY_REGRESSION=(
  tests/unit/test_i6_4a_provider_availability_selection_contract.py
  tests/integration/test_i6_4a_provider_availability_selection_contract.py
  tests/stress/test_i6_4a_provider_availability_selection_contract_stress.py
)

I6_6_AUTHORITY_BOUNDARY_REGRESSION=(
  tests/unit/test_i6_6a_progressive_sfm_snapshot_contract.py
  tests/integration/test_i6_6a_progressive_sfm_snapshot_transition.py
  tests/stress/test_i6_6a_progressive_sfm_snapshot_stress.py
  tests/unit/test_i6_6e_progressive_sfm_ux.py
  tests/unit/test_i6_6e_progressive_sfm_ui_static.py
  tests/stress/test_i6_6e_progressive_sfm_performance_stress.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
DENSE_RC=0; DENSE_COUNT=0
I6_4_RC=0; I6_4_COUNT=0
I6_6_RC=0; I6_6_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0; COMPILE_RC=0; SCOPE_RC=0

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
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.reconstruction_progressive_dense_contract import (
    ProgressiveDenseSnapshot,
    ProgressiveDenseStage,
    ProgressiveDenseStatistics,
    validate_progressive_dense_snapshot_transition,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION,
)

implementation = DenseProviderImplementation.COLMAP_MVS
first = ProgressiveDenseSnapshot(
    source_sfm_session_id='11111111-2222-4333-8444-555555555555',
    dense_run_id='aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee',
    revision=1,
    timestamp_utc='2026-09-18T09:00:00+00:00',
    elapsed_ns=1,
    provider_id=PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[implementation],
    implementation=implementation,
    stage=ProgressiveDenseStage.PROVIDER_EXECUTION,
    stage_detail='probe',
    request_fingerprint_sha256='a' * 64,
    source_sparse_result_sha256='b' * 64,
    reconstruction_frame_id='pycolmap-reconstruction-frame',
    statistics=ProgressiveDenseStatistics(
        input_views=24,
        selected_views=18,
        processed_views=2,
        current_dense_points=1200,
    ),
    progress_fraction=0.4,
)
second_payload = first.as_dict()
second_payload.pop('snapshot_sha256')
second_payload.update({
    'revision': 4,
    'timestamp_utc': '2026-09-18T09:00:01+00:00',
    'elapsed_ns': 2,
    'stage': ProgressiveDenseStage.FUSION_FILTERING.value,
    'progress_fraction': 0.8,
})
second_payload['statistics']['current_dense_points'] = 900
second = ProgressiveDenseSnapshot.from_payload(second_payload)
validate_progressive_dense_snapshot_transition(first, second)
assert first.provisional is True
assert first.final_dense_result_admitted is False
assert first.persistent_dense_scene is False
assert first.canonical_geometry_authoritative is False
assert first.metric_scale_claimed is False
assert first.surface_mesh_generated is False
assert len(first.snapshot_sha256) == 64
print('I6.7A progressive dense contract/authority import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_progressive_dense_contract.py \
    tests/unit/test_i6_7a_progressive_dense_snapshot_contract.py \
    tests/integration/test_i6_7a_progressive_dense_contract_integration.py \
    tests/stress/test_i6_7a_progressive_dense_contract_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

probe_scope() {
  python - <<'PY'
from pathlib import Path

text = Path('core/reconstruction/reconstruction_progressive_dense_contract.py').read_text(encoding='utf-8')
for forbidden in (
    'from app.', 'import app.', 'PyQt6', 'vispy', 'subprocess',
    'import pycolmap', 'start_offline', 'select_dense_provider',
    'i5_auto_policy_engine',
):
    assert forbidden not in text, forbidden
for required in (
    'ProgressiveDensePointSample',
    'ProgressiveDenseGeometryPreview',
    'ProgressiveDenseStatistics',
    'ProgressiveDenseSnapshot',
    'validate_progressive_dense_snapshot_transition',
    'source_sparse_result_sha256',
    'reconstruction_frame_id',
    'provisional_dense_visualization_only',
    'final_dense_result_admitted',
    'persistent_dense_scene',
    'surface_mesh_generated',
):
    assert required in text, required
print('I6_7A_SCOPE_GUARD=PASS')
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
  run_pytest_count DENSE_RC DENSE_COUNT DENSE_PREDECESSOR_REGRESSION "${DENSE_PREDECESSOR_REGRESSION[@]}"
  run_pytest_count I6_4_RC I6_4_COUNT I6_4_PROVIDER_BOUNDARY_REGRESSION "${I6_4_PROVIDER_BOUNDARY_REGRESSION[@]}"
  run_pytest_count I6_6_RC I6_6_COUNT I6_6_AUTHORITY_BOUNDARY_REGRESSION "${I6_6_AUTHORITY_BOUNDARY_REGRESSION[@]}"
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
  *) echo "usage: $0 [focused|predecessors|all|full-unit|closure] [receipt.json]" >&2; exit 2 ;;
esac

TEST_RC=0
case "$MODE" in
  focused)
    for rc in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC" "$SCOPE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  predecessors)
    for rc in "$DENSE_RC" "$I6_4_RC" "$I6_6_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  all)
    for rc in "$FOCUSED_RC" "$DENSE_RC" "$I6_4_RC" "$I6_6_RC" "$IMPORT_RC" "$COMPILE_RC" "$SCOPE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure)
    for rc in "$FOCUSED_RC" "$DENSE_RC" "$I6_4_RC" "$I6_6_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC" "$SCOPE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
    "$DENSE_RC" "$DENSE_COUNT" \
    "$I6_4_RC" "$I6_4_COUNT" \
    "$I6_6_RC" "$I6_6_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" "$SCOPE_RC" <<'PY'
import json, sys
(
    path, mode, test_rc,
    focused_rc, focused_count,
    dense_rc, dense_count,
    i64_rc, i64_count,
    i66_rc, i66_count,
    full_rc, full_count,
    import_rc, compile_rc, scope_rc,
) = sys.argv[1:]
payload = {
    'schema_version': 1,
    'phase': 'I6.7A_PROGRESSIVE_DENSE_CONTRACT_AND_AUTHORITY_BOUNDARY',
    'mode': mode,
    'status': 'pass' if int(test_rc) == 0 else 'fail',
    'pass': int(test_rc) == 0,
    'manual_ui_validation_required': False,
    'counts': {
        'focused_tests': int(focused_count),
        'dense_predecessor_regression_tests': int(dense_count),
        'i6_4_provider_boundary_regression_tests': int(i64_count),
        'i6_6_authority_boundary_regression_tests': int(i66_count),
        'full_unit_tests': int(full_count),
    },
    'return_codes': {
        'focused': int(focused_rc),
        'dense_predecessor_regression': int(dense_rc),
        'i6_4_provider_boundary_regression': int(i64_rc),
        'i6_6_authority_boundary_regression': int(i66_rc),
        'full_unit': int(full_rc),
        'import_probe': int(import_rc),
        'compile_probe': int(compile_rc),
        'scope_guard': int(scope_rc),
    },
    'scope': {
        'progressive_dense_snapshot_contract_added': True,
        'bounded_dense_point_preview_contract_added': True,
        'snapshot_digest_added': True,
        'bound_to_admitted_sparse_result_digest': True,
        'bound_to_authoritative_reconstruction_frame': True,
        'all_canonical_concrete_providers_supported': True,
        'auto_is_executable_provider': False,
        'revision_gaps_allowed_for_future_coalescing': True,
        'stale_or_duplicate_revision_rejected': True,
        'cross_run_or_provider_transition_rejected': True,
        'dense_point_counts_required_monotonic': False,
        'snapshot_provisional': True,
        'final_dense_result_admitted': False,
        'persistent_dense_scene_created': False,
        'canonical_geometry_authoritative': False,
        'provider_execution_added': False,
        'provider_selection_modified': False,
        'production_auto_activated': False,
        'worker_modified': False,
        'controller_modified': False,
        'ui_modified': False,
        'viewport_modified': False,
        'progressive_dense_visualization_added': False,
        'surface_mesh_generated': False,
        'pycolmap_camera_authority_preserved': True,
        'metric_scale_claimed': False,
        'measurement_authoritative': False,
        'certified_metrology_claimed': False,
    },
}
open(path, 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'I6.7A receipt={path}')
PY
fi

echo "I6_7A_TEST_RC=$TEST_RC"
exit "$TEST_RC"
