#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
case "$MODE" in
  focused)
    python -m pytest -q \
      tests/unit/test_i4_1_7_clean_static_orbit_qualification.py \
      tests/unit/test_i4_1_7a_final_geometry_diagnosis.py \
      tests/unit/test_i4_1_7b_bounded_target_correction.py \
      tests/integration/test_i4_1_7_clean_static_orbit_contract.py \
      tests/integration/test_i4_1_7a_final_geometry_diagnosis_contract.py \
      tests/integration/test_i4_1_7b_bounded_target_correction_contract.py
    ;;
  quick)
    python -m pytest -q \
      tests/unit/test_i4_1_7_clean_static_orbit_qualification.py \
      tests/unit/test_i4_1_7a_final_geometry_diagnosis.py \
      tests/unit/test_i4_1_7b_bounded_target_correction.py \
      tests/integration/test_i4_1_7_clean_static_orbit_contract.py \
      tests/integration/test_i4_1_7a_final_geometry_diagnosis_contract.py \
      tests/integration/test_i4_1_7b_bounded_target_correction_contract.py \
      tests/unit/test_i4_1_6b_decoupled_dense_ab.py \
      tests/integration/test_i4_1_6b_dense_ab_contract.py \
      tests/unit/test_i4_1_6_decoupled_sampling_integration.py \
      tests/integration/test_i4_1_6_decoupled_sampling_contract.py \
      tests/unit/test_i4_1_5_decoupled_sampling_counterfactual.py \
      tests/integration/test_i4_1_5_decoupled_sampling_contract.py \
      tests/unit/test_i4_1_4_decoupled_sampling_density_diagnosis.py \
      tests/integration/test_i4_1_4_sampling_density_contract.py \
      tests/unit/test_i4_1_3_full_clip_uniform_sampling.py \
      tests/integration/test_i4_1_3_full_clip_sampling_pipeline.py \
      tests/unit/test_i4_1_2_orbit_geometry_sampling_diagnosis.py \
      tests/integration/test_i4_1_2_orbit_geometry_sampling_contract.py \
      tests/unit/test_i4_1_1_orbit_alignment_coverage_diagnosis.py \
      tests/integration/test_i4_1_1_orbit_diagnosis_contract.py \
      tests/integration/test_i4_1_1_preview_canvas_contract.py \
      tests/unit/test_i4_1_object_orbit_qualification.py \
      tests/integration/test_i4_1_i3_closure_contract.py \
      tests/unit/test_i3_2_second_scene_production_promotion.py \
      tests/unit/test_i3_1_second_scene_transfer_campaign.py \
      tests/unit/test_i2_7_2_baseline_aware_component_gate.py \
      tests/unit/test_i2_7_1_detached_component_qualification.py \
      tests/unit/test_i2_7_final_requalification.py \
      tests/unit/test_i2_6_manifold_filter_promotion.py \
      tests/unit/test_i2_5_3_conservative_manifold_filter_counterfactual.py \
      tests/unit/test_i2_5_2_extremum_local_geometry_manifold_diagnosis.py \
      tests/unit/test_i2_5_1_spatial_extremum_survivor_diagnosis.py \
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
  integration)
    python -m pytest -q \
      tests/integration/test_i4_1_7_clean_static_orbit_contract.py \
      tests/integration/test_i4_1_7a_final_geometry_diagnosis_contract.py \
      tests/integration/test_i4_1_7b_bounded_target_correction_contract.py \
      tests/integration/test_i4_1_6b_dense_ab_contract.py \
      tests/integration/test_i4_1_6_decoupled_sampling_contract.py \
      tests/integration/test_i4_1_5_decoupled_sampling_contract.py \
      tests/integration/test_i4_1_4_sampling_density_contract.py \
      tests/integration/test_i4_1_3_full_clip_sampling_pipeline.py \
      tests/integration/test_i4_1_2_orbit_geometry_sampling_contract.py \
      tests/integration/test_i4_1_1_orbit_diagnosis_contract.py \
      tests/integration/test_i4_1_1_preview_canvas_contract.py \
      tests/integration/test_i4_1_i3_closure_contract.py
    ;;
  stress)
    python -m pytest -q \
      tests/stress/test_i4_1_7_clean_static_orbit_stress.py \
      tests/stress/test_i4_1_7a_final_geometry_diagnosis_stress.py \
      tests/stress/test_i4_1_7b_bounded_target_correction_stress.py \
      tests/stress/test_i4_1_6b_dense_ab_stress.py \
      tests/stress/test_i4_1_6_decoupled_sampling_stress.py \
      tests/stress/test_i4_1_5_decoupled_sampling_stress.py \
      tests/stress/test_i4_1_4_sampling_density_stress.py \
      tests/stress/test_i4_1_3_full_clip_sampling_stress.py \
      tests/stress/test_i4_1_2_orbit_geometry_sampling_stress.py \
      tests/stress/test_i4_1_1_orbit_diagnosis_stress.py \
      tests/stress/test_i4_1_closure_contract_stress.py
    ;;
  affected)
    python -m pytest -q \
      tests/unit/test_i4_1_7_clean_static_orbit_qualification.py \
      tests/unit/test_i4_1_7a_final_geometry_diagnosis.py \
      tests/unit/test_i4_1_7b_bounded_target_correction.py \
      tests/integration/test_i4_1_7_clean_static_orbit_contract.py \
      tests/integration/test_i4_1_7a_final_geometry_diagnosis_contract.py \
      tests/integration/test_i4_1_7b_bounded_target_correction_contract.py \
      tests/stress/test_i4_1_7_clean_static_orbit_stress.py \
      tests/stress/test_i4_1_7a_final_geometry_diagnosis_stress.py \
      tests/stress/test_i4_1_7b_bounded_target_correction_stress.py \
      tests/unit/test_i4_1_6b_decoupled_dense_ab.py \
      tests/integration/test_i4_1_6b_dense_ab_contract.py \
      tests/stress/test_i4_1_6b_dense_ab_stress.py \
      tests/unit/test_i4_1_6_decoupled_sampling_integration.py \
      tests/integration/test_i4_1_6_decoupled_sampling_contract.py \
      tests/stress/test_i4_1_6_decoupled_sampling_stress.py \
      tests/unit/test_i4_1_5_decoupled_sampling_counterfactual.py \
      tests/integration/test_i4_1_5_decoupled_sampling_contract.py \
      tests/stress/test_i4_1_5_decoupled_sampling_stress.py \
      tests/unit/test_i4_1_4_decoupled_sampling_density_diagnosis.py \
      tests/integration/test_i4_1_4_sampling_density_contract.py \
      tests/stress/test_i4_1_4_sampling_density_stress.py \
      tests/unit/test_i4_1_3_full_clip_uniform_sampling.py \
      tests/integration/test_i4_1_3_full_clip_sampling_pipeline.py \
      tests/stress/test_i4_1_3_full_clip_sampling_stress.py \
      tests/unit/test_reconstruction_ingestion_quality_sampling.py \
      tests/unit/test_reconstruction_ingestion_pipeline.py \
      tests/integration/test_reconstruction_ingestion_lifecycle.py \
      tests/stress/test_reconstruction_ingestion_stress.py \
      tests/unit/test_i4_1_2_orbit_geometry_sampling_diagnosis.py \
      tests/integration/test_i4_1_2_orbit_geometry_sampling_contract.py \
      tests/stress/test_i4_1_2_orbit_geometry_sampling_stress.py \
      tests/unit/test_i4_1_1_orbit_alignment_coverage_diagnosis.py \
      tests/integration/test_i4_1_1_orbit_diagnosis_contract.py \
      tests/integration/test_i4_1_1_preview_canvas_contract.py \
      tests/stress/test_i4_1_1_orbit_diagnosis_stress.py \
      tests/unit/test_i4_1_object_orbit_qualification.py \
      tests/integration/test_i4_1_i3_closure_contract.py \
      tests/stress/test_i4_1_closure_contract_stress.py \
      tests/unit/test_i3_2_second_scene_production_promotion.py \
      tests/unit/test_i3_1_second_scene_transfer_campaign.py \
      tests/unit/test_i2_7_2_baseline_aware_component_gate.py \
      tests/unit/test_i2_7_1_detached_component_qualification.py \
      tests/unit/test_i2_7_final_requalification.py \
      tests/unit/test_i2_6_manifold_filter_promotion.py \
      tests/unit/test_i2_5_3_conservative_manifold_filter_counterfactual.py \
      tests/unit/test_i2_5_2_extremum_local_geometry_manifold_diagnosis.py \
      tests/unit/test_i2_5_1_spatial_extremum_survivor_diagnosis.py \
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
  *) echo "usage: $0 [focused|quick|integration|stress|affected]" >&2; exit 2;;
esac
