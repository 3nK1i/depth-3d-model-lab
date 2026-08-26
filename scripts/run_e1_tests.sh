#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_e1_qualified_ten_view_voxel_fusion.py \
      tests/unit/test_reconstruction_dense_fusion.py \
      tests/unit/test_d4_3_repaired_ten_view_chain.py
    ;;
  affected)
    pytest -q \
      tests/unit/test_e1_qualified_ten_view_voxel_fusion.py \
      tests/unit/test_d4_3_repaired_ten_view_chain.py \
      tests/unit/test_d4_2_weak_link_bridge_diagnostics.py \
      tests/unit/test_d4_1_ten_view_chain_diagnostics.py \
      tests/unit/test_d4_ten_view_canonical_accumulation.py \
      tests/unit/test_d3_1_five_view_chain_diagnostics.py \
      tests/unit/test_d3_five_view_canonical_accumulation.py \
      tests/unit/test_d2_three_view_overlap_diagnostics.py \
      tests/unit/test_d1_three_view_canonical_accumulation.py \
      tests/unit/test_b2_sfm_dense_real_alignment.py \
      tests/unit/test_c_single_frame_canonical_projection.py \
      tests/unit/test_reconstruction_sfm_dense_alignment.py \
      tests/unit/test_reconstruction_sfm_dense_projection.py \
      tests/unit/test_reconstruction_dense_fusion.py \
      tests/unit/test_reconstruction_sfm_contracts.py \
      tests/unit/test_reconstruction_sfm_coordinates.py \
      tests/unit/test_reconstruction_sfm_pycolmap.py \
      tests/unit/test_reconstruction_pipeline_geometry_f3.py \
      tests/integration/test_reconstruction_dense_fusion_pipeline.py
    ;;
  *) echo "usage: $0 [quick|affected]" >&2; exit 2;;
esac
