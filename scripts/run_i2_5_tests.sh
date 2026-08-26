#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
case "$MODE" in
  focused)
    python -m pytest -q tests/unit/test_i2_5_conservative_diversity_filter_counterfactual.py
    ;;
  quick)
    python -m pytest -q \
      tests/unit/test_i2_5_conservative_diversity_filter_counterfactual.py \
      tests/unit/test_i2_4_support_diversity_parallax_diagnosis.py \
      tests/unit/test_i2_3_balanced_cross_segment_counterfactuals.py \
      tests/unit/test_i2_2_narrow_exact_counterfactual_refusion.py \
      tests/unit/test_i2_1_targeted_peripheral_diagnosis.py \
      tests/unit/test_i2_full_video_quality_qualification.py \
      tests/unit/test_i1_offline_dense_campaign.py \
      tests/unit/test_h_dense_rgb_reconstruction_ui.py \
      tests/unit/test_g_persistent_dense_rgb_geometry.py \
      tests/unit/test_f_confidence_artifact_filtering.py
    ;;
  affected)
    python -m pytest -q \
      tests/unit/test_i2_5_conservative_diversity_filter_counterfactual.py \
      tests/unit/test_i2_4_support_diversity_parallax_diagnosis.py \
      tests/unit/test_i2_3_balanced_cross_segment_counterfactuals.py \
      tests/unit/test_i2_2_narrow_exact_counterfactual_refusion.py \
      tests/unit/test_i2_1_targeted_peripheral_diagnosis.py \
      tests/unit/test_i2_full_video_quality_qualification.py \
      tests/unit/test_i1_offline_dense_campaign.py \
      tests/unit/test_e3_qualified_scene_fusion_expansion.py \
      tests/unit/test_h_dense_rgb_reconstruction_ui.py \
      tests/ui/test_h_dense_rgb_reconstruction_ui.py \
      tests/unit/test_g_persistent_dense_rgb_geometry.py \
      tests/unit/test_f_confidence_artifact_filtering.py \
      tests/ui/test_phase9_3b_canonical_geometry_viewer.py \
      tests/ui/test_phase9_3f_scene_layers.py \
      tests/ui/test_phase9_4f_gaussian_controls_layers_ui.py \
      tests/ui/test_phase9_5f_nerf_controls_layers_ui.py \
      tests/ui/test_phase9_6b_unified_scene_composition_ui.py \
      tests/ui/test_phase9_6d_unified_interaction_ui.py \
      tests/ui/test_phase9_6e4_workspace_ux.py \
      tests/ui/test_viewport_controls.py \
      tests/ui/test_viewport_layout.py
    ;;
  *)
    echo "usage: $0 [focused|quick|affected]" >&2
    exit 2
    ;;
esac
