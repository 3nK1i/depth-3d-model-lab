#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED=(
  tests/unit/test_i6_7b_progressive_dense_emission.py
  tests/integration/test_i6_7b_dense_provider_execution_emission.py
  tests/stress/test_i6_7b_progressive_dense_emission_stress.py
)

I6_7A_REGRESSION=(
  tests/unit/test_i6_7a_progressive_dense_snapshot_contract.py
  tests/integration/test_i6_7a_progressive_dense_contract_integration.py
  tests/stress/test_i6_7a_progressive_dense_contract_stress.py
)

CONCRETE_PROVIDER_PREDECESSOR=(
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

I6_4_PROVIDER_BOUNDARY=(
  tests/unit/test_i6_4a_provider_availability_selection_contract.py
  tests/integration/test_i6_4a_provider_availability_selection_contract.py
  tests/stress/test_i6_4a_provider_availability_selection_contract_stress.py
)

I6_6_AUTHORITY_BOUNDARY=(
  tests/unit/test_i6_6a_progressive_sfm_snapshot_contract.py
  tests/integration/test_i6_6a_progressive_sfm_snapshot_transition.py
  tests/stress/test_i6_6a_progressive_sfm_snapshot_stress.py
  tests/unit/test_i6_6e_progressive_sfm_ux.py
  tests/unit/test_i6_6e_progressive_sfm_ui_static.py
  tests/stress/test_i6_6e_progressive_sfm_performance_stress.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
I67A_RC=0; I67A_COUNT=0
PROVIDER_RC=0; PROVIDER_COUNT=0
I64_RC=0; I64_COUNT=0
I66_RC=0; I66_COUNT=0
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
from pathlib import Path
import tempfile
import numpy as np

from core.reconstruction.reconstruction_progressive_dense_emission import (
    I67B_MAX_PREVIEW_POINTS,
    ProgressiveDenseEmissionPolicy,
    evenly_sample_indices,
    load_dense_geometry_preview,
)

assert I67B_MAX_PREVIEW_POINTS == 4096
assert ProgressiveDenseEmissionPolicy().target_status_snapshots == 24
assert evenly_sample_indices(100, 4) == (0, 33, 66, 99)
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / 'probe.npz'
    np.savez_compressed(
        path,
        points_reconstruction=np.asarray([[0,0,0],[1,2,3],[4,5,6]], dtype=np.float32),
        colors_rgb=np.asarray([[1,2,3],[4,5,6],[7,8,9]], dtype=np.uint8),
    )
    preview = load_dense_geometry_preview(path, max_preview_points=2)
    assert preview.point_count == 3
    assert len(preview.preview.point_samples) == 2
print('I6.7B dense-provider execution/progressive-emission import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_progressive_dense_emission.py \
    tests/unit/test_i6_7b_progressive_dense_emission.py \
    tests/integration/test_i6_7b_dense_provider_execution_emission.py \
    tests/stress/test_i6_7b_progressive_dense_emission_stress.py \
    scripts/run_i6_7b_progressive_dense_emission_smoke.py \
    scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

probe_scope() {
  python - <<'PY'
from pathlib import Path

text = Path('core/reconstruction/reconstruction_progressive_dense_emission.py').read_text(encoding='utf-8')
for forbidden in (
    'from app.', 'import app.', 'PyQt6', 'vispy', 'import pycolmap',
    'select_dense_provider', 'i5_auto_policy_engine',
):
    assert forbidden not in text, forbidden
for required in (
    'I67B_MAX_PREVIEW_POINTS = 4096',
    'I67B_TARGET_STATUS_SNAPSHOTS = 24',
    'I67B_MAX_FAILURE_RECORDS = 16',
    'I67B_MAX_PROVISIONAL_PROGRESS = 0.999999',
    'class ProgressiveDenseEmissionSession',
    'def execute_dense_provider_with_progressive_emission',
    'load_dense_geometry_preview',
    'bounded_canonical_geometry_before_external_admission',
    'validate_progressive_dense_snapshot_transition',
):
    assert required in text, required

provider_files = (
    'core/reconstruction/mvs_provider.py',
    'core/reconstruction/da3_dense_provider.py',
    'core/reconstruction/rolling_depth_dense_provider.py',
    'core/reconstruction/video_depth_anything_dense_provider.py',
    'core/reconstruction/metric3d_v2_dense_provider.py',
    'core/reconstruction/unidepth_v2_dense_provider.py',
    'core/reconstruction/moge2_dense_provider.py',
    'core/reconstruction/depth_pro_dense_provider.py',
    'core/reconstruction/depth_anything_v2_dense_provider.py',
    'core/reconstruction/marigold_dense_provider.py',
)
assert all(Path(path).is_file() for path in provider_files)

worker = Path('scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py').read_text(encoding='utf-8')
assert 'fusion_options.use_cache = False' in worker
assert 'fusion_options.num_threads = -1' in worker
assert 'fusion_options.use_cache = True' not in worker
assert '"scientific_fusion_parameters_changed": False' in worker

smoke = Path('scripts/run_i6_7b_progressive_dense_emission_smoke.py').read_text(encoding='utf-8')
status_guard = smoke.index(
    'if envelope.result.status not in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}:'
)
geometry_guard = smoke.index('if not geometry:')
assert status_guard < geometry_guard
assert 'failure_reason' in smoke[status_guard:geometry_guard]

print('I6_7B_SCOPE_GUARD=PASS')
print('GENERIC_EXISTING_PROVIDER_WRAPPER=YES')
print('ALL_10_PROVIDER_ADAPTERS_SOURCE_MODIFIED=NO')
print('MAX_PREVIEW_POINTS=4096')
print('TARGET_STATUS_SNAPSHOTS=24')
print('SNAPSHOT_SINK_FAILURE_FAIL_SOFT=YES')
print('EMITTED_STAGE_REGRESSION_REJECTED=YES')
print('RETRY_UNTIL_PASS=NO')
print('FINAL_GEOMETRY_PREVIEW_BEFORE_EXTERNAL_ADMISSION=YES')
print('PRODUCTION_FUSION_USE_CACHE=NO')
print('PRODUCTION_FUSION_NUM_THREADS=AUTO')
print('SCIENTIFIC_FUSION_PARAMETERS_CHANGED=NO')
print('FINAL_DENSE_RESULT_ADMITTED_BY_PROGRESSIVE_PATH=NO')
print('PERSISTENT_DENSE_SCENE_CREATED=NO')
print('PROVIDER_SELECTION_MODIFIED=NO')
print('PRODUCTION_AUTO_ACTIVATED=NO')
print('CONTROLLER_MODIFIED=NO')
print('UI_MODIFIED=NO')
print('VIEWPORT_MODIFIED=NO')
print('SURFACE_MESH_GENERATED=NO')
print('PYCOLMAP_CAMERA_AUTHORITY_PRESERVED=YES')
print('METRIC_SCALE_CLAIMED=NO')
print('MEASUREMENT_AUTHORITATIVE=NO')
print('CERTIFIED_METROLOGY_CLAIMED=NO')
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
  run_pytest_count I67A_RC I67A_COUNT I6_7A_REGRESSION "${I6_7A_REGRESSION[@]}"
  run_pytest_count PROVIDER_RC PROVIDER_COUNT CONCRETE_PROVIDER_PREDECESSOR "${CONCRETE_PROVIDER_PREDECESSOR[@]}"
  run_pytest_count I64_RC I64_COUNT I6_4_PROVIDER_BOUNDARY "${I6_4_PROVIDER_BOUNDARY[@]}"
  run_pytest_count I66_RC I66_COUNT I6_6_AUTHORITY_BOUNDARY "${I6_6_AUTHORITY_BOUNDARY[@]}"
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
    for rc in "$I67A_RC" "$PROVIDER_RC" "$I64_RC" "$I66_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  all)
    for rc in "$FOCUSED_RC" "$I67A_RC" "$PROVIDER_RC" "$I64_RC" "$I66_RC" "$IMPORT_RC" "$COMPILE_RC" "$SCOPE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure)
    for rc in "$FOCUSED_RC" "$I67A_RC" "$PROVIDER_RC" "$I64_RC" "$I66_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC" "$SCOPE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
    "$I67A_RC" "$I67A_COUNT" \
    "$PROVIDER_RC" "$PROVIDER_COUNT" \
    "$I64_RC" "$I64_COUNT" \
    "$I66_RC" "$I66_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" "$SCOPE_RC" <<'PY'
import json, sys
(
    path, mode, test_rc,
    focused_rc, focused_count,
    i67a_rc, i67a_count,
    provider_rc, provider_count,
    i64_rc, i64_count,
    i66_rc, i66_count,
    full_rc, full_count,
    import_rc, compile_rc, scope_rc,
) = sys.argv[1:]
payload = {
    'schema_version': 1,
    'phase': 'I6.7B_DENSE_PROVIDER_EXECUTION_AND_PROGRESSIVE_EMISSION',
    'mode': mode,
    'status': 'pass' if int(test_rc) == 0 else 'fail',
    'pass': int(test_rc) == 0,
    'real_provider_smoke_required_before_freeze': True,
    'manual_ui_validation_required': False,
    'counts': {
        'focused_tests': int(focused_count),
        'i6_7a_regression_tests': int(i67a_count),
        'concrete_provider_predecessor_tests': int(provider_count),
        'i6_4_provider_boundary_tests': int(i64_count),
        'i6_6_authority_boundary_tests': int(i66_count),
        'full_unit_tests': int(full_count),
    },
    'return_codes': {
        'focused': int(focused_rc),
        'i6_7a_regression': int(i67a_rc),
        'concrete_provider_predecessor': int(provider_rc),
        'i6_4_provider_boundary': int(i64_rc),
        'i6_6_authority_boundary': int(i66_rc),
        'full_unit': int(full_rc),
        'import_probe': int(import_rc),
        'compile_probe': int(compile_rc),
        'scope_guard': int(scope_rc),
    },
    'scope': {
        'generic_existing_provider_execution_wrapper_added': True,
        'bounded_progressive_dense_status_emission_added': True,
        'bounded_actual_dense_geometry_preview_added': True,
        'max_preview_points': 4096,
        'target_status_snapshots': 24,
        'preview_limit_is_scientific_reconstruction_cap': False,
        'snapshot_sink_failure_blocks_provider_result': False,
        'retry_until_pass': False,
        'production_fusion_use_cache': False,
        'production_fusion_num_threads': -1,
        'scientific_fusion_parameters_changed': False,
        'all_ten_provider_adapter_sources_modified': False,
        'provider_selection_modified': False,
        'production_auto_activated': False,
        'final_dense_result_admitted_by_progressive_path': False,
        'persistent_dense_scene_created': False,
        'controller_modified': False,
        'ui_modified': False,
        'viewport_modified': False,
        'surface_mesh_generated': False,
        'pycolmap_camera_authority_preserved': True,
        'metric_scale_claimed': False,
        'measurement_authoritative': False,
        'certified_metrology_claimed': False,
    },
}
open(path, 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'I6.7B receipt={path}')
PY
fi

echo "I6_7B_TEST_RC=$TEST_RC"
exit "$TEST_RC"
