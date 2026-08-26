#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# I4.3D1 extends the validated I4.3C gate with Video Depth Anything as the
# first temporal-depth production provider. RollingDepth remains I4.3D2.
i42_focused=(
  tests/unit/test_i4_2a_temple_ring_preflight.py
  tests/unit/test_i4_2a_1_middlebury_header.py
  tests/integration/test_i4_2a_temple_ring_preflight_contract.py
  tests/integration/test_i4_2a_1_middlebury_header_contract.py
  tests/unit/test_i4_2b_temple_ring_sfm.py
  tests/integration/test_i4_2b_temple_ring_sfm_contract.py
  tests/unit/test_i4_2c_temple_ring_trajectory_validation.py
  tests/integration/test_i4_2c_temple_ring_trajectory_validation_contract.py
  tests/unit/test_i4_2d_temple_ring_mvs.py
  tests/integration/test_i4_2d_temple_ring_mvs_contract.py
  tests/unit/test_i4_2e_temple_ring_detail_qualification.py
  tests/integration/test_i4_2e_temple_ring_detail_qualification_contract.py
  tests/integration/test_i4_2e_temple_ring_detail_preview_contract.py
  tests/unit/test_i4_2e_1_worker_dependency_isolation.py
  tests/integration/test_i4_2e_1_worker_dependency_isolation_contract.py
  tests/unit/test_i4_2e_2_native_visibility_provenance.py
  tests/integration/test_i4_2e_2_native_visibility_provenance_contract.py
  tests/unit/test_i4_2e_2a_native_visibility_coverage_contract.py
  tests/integration/test_i4_2e_2a_native_visibility_coverage_contract.py
  tests/unit/test_i4_2f_temple_ring_closure.py
  tests/integration/test_i4_2f_temple_ring_closure_contract.py
)
i42_stress=(
  tests/stress/test_i4_2a_temple_ring_preflight_stress.py
  tests/stress/test_i4_2a_1_middlebury_header_stress.py
  tests/stress/test_i4_2b_temple_ring_sfm_stress.py
  tests/stress/test_i4_2c_temple_ring_trajectory_validation_stress.py
  tests/stress/test_i4_2d_temple_ring_mvs_stress.py
  tests/stress/test_i4_2e_temple_ring_detail_qualification_stress.py
  tests/stress/test_i4_2e_1_worker_dependency_isolation_stress.py
  tests/stress/test_i4_2e_2_native_visibility_provenance_stress.py
  tests/stress/test_i4_2e_2a_native_visibility_coverage_stress.py
  tests/stress/test_i4_2f_temple_ring_closure_stress.py
)
i43a_focused=(tests/unit/test_i4_3a_dense_provider_architecture.py tests/integration/test_i4_3a_dense_provider_architecture_contract.py)
i43a_stress=(tests/stress/test_i4_3a_dense_provider_architecture_stress.py)
i43b_focused=(tests/unit/test_i4_3b_mvs_provider.py tests/integration/test_i4_3b_mvs_provider_contract.py)
i43b_stress=(tests/stress/test_i4_3b_mvs_provider_stress.py)
da3_backward_focused=(tests/unit/test_reconstruction_da3.py tests/unit/test_reconstruction_da3_provider.py tests/integration/test_reconstruction_da3_provider.py)
da3_backward_stress=(tests/stress/test_reconstruction_da3_provider_stress.py)
i43c_focused=(tests/unit/test_i4_3c_da3_dense_provider.py tests/integration/test_i4_3c_da3_dense_provider_contract.py)
i43c_stress=(tests/stress/test_i4_3c_da3_dense_provider_stress.py)
vda_backward_focused=(
  tests/unit/test_video_depth_anything_adapter.py
  tests/unit/test_model_output_contract.py
  tests/unit/test_model_registry.py
)
vda_backward_stress=(tests/stress/test_depth_anything_adapters_stress.py)
i43d1_focused=(
  tests/unit/test_i4_3d1_video_depth_anything_dense_provider.py
  tests/integration/test_i4_3d1_video_depth_anything_provider_contract.py
)
i43d1_stress=(tests/stress/test_i4_3d1_video_depth_anything_provider_stress.py)

case "${1:-all}" in
  focused) python -m pytest -q "${i42_focused[@]}" "${i43a_focused[@]}" "${i43b_focused[@]}" "${da3_backward_focused[@]}" "${i43c_focused[@]}" "${vda_backward_focused[@]}" "${i43d1_focused[@]}" ;;
  stress) python -m pytest -q "${i42_stress[@]}" "${i43a_stress[@]}" "${i43b_stress[@]}" "${da3_backward_stress[@]}" "${i43c_stress[@]}" "${vda_backward_stress[@]}" "${i43d1_stress[@]}" ;;
  all) python -m pytest -q "${i42_focused[@]}" "${i43a_focused[@]}" "${i43b_focused[@]}" "${da3_backward_focused[@]}" "${i43c_focused[@]}" "${vda_backward_focused[@]}" "${i43d1_focused[@]}" "${i42_stress[@]}" "${i43a_stress[@]}" "${i43b_stress[@]}" "${da3_backward_stress[@]}" "${i43c_stress[@]}" "${vda_backward_stress[@]}" "${i43d1_stress[@]}" ;;
  *) echo "usage: $0 [focused|stress|all]" >&2; exit 2 ;;
esac

python - <<'PYPROBE'
from pathlib import Path
from core.models.adapters.video_depth_anything_adapter import VideoDepthAnythingGeometryModel
from core.models.catalog import VIDEO_DEPTH_ANYTHING_MODEL_ID
root = Path.cwd()
source = root / "third_party/Video-Depth-Anything/video_depth_anything/video_depth_stream.py"
assert VideoDepthAnythingGeometryModel is not None
assert VIDEO_DEPTH_ANYTHING_MODEL_ID == "video-depth-anything"
assert source.is_file()
print("I4.3D1 Video Depth Anything adapter/source import probe: PASS")
PYPROBE
