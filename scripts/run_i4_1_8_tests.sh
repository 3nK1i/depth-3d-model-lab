#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
case "$MODE" in
  focused)
    python -m pytest -q \
      tests/unit/test_i4_1_8c_final_object_orbit_causality.py \
      tests/integration/test_i4_1_8c_final_object_orbit_causality_contract.py \
      tests/unit/test_i4_1_8b_retained_core_object_support.py \
      tests/integration/test_i4_1_8b_retained_core_object_support_contract.py \
      tests/unit/test_i4_1_8a_retained_core_target_diagnosis.py \
      tests/integration/test_i4_1_8a_retained_core_target_diagnosis_contract.py \
      tests/unit/test_i4_1_8_production_binding.py \
      tests/integration/test_i4_1_8_production_binding_contract.py \
      tests/integration/test_i4_1_8_preview_canvas_contract.py
    ;;
  quick)
    python -m pytest -q \
      tests/unit/test_i4_1_8c_final_object_orbit_causality.py \
      tests/integration/test_i4_1_8c_final_object_orbit_causality_contract.py \
      tests/unit/test_i4_1_8b_retained_core_object_support.py \
      tests/integration/test_i4_1_8b_retained_core_object_support_contract.py \
      tests/unit/test_i4_1_8a_retained_core_target_diagnosis.py \
      tests/integration/test_i4_1_8a_retained_core_target_diagnosis_contract.py \
      tests/unit/test_i4_1_8_production_binding.py \
      tests/integration/test_i4_1_8_production_binding_contract.py \
      tests/integration/test_i4_1_8_preview_canvas_contract.py \
      tests/unit/test_i4_1_7b_bounded_target_correction.py \
      tests/integration/test_i4_1_7b_bounded_target_correction_contract.py \
      tests/unit/test_i4_1_7a_final_geometry_diagnosis.py \
      tests/integration/test_i4_1_7a_final_geometry_diagnosis_contract.py \
      tests/unit/test_i4_1_7_clean_static_orbit_qualification.py \
      tests/integration/test_i4_1_7_clean_static_orbit_contract.py \
      tests/unit/test_i4_1_6b_decoupled_dense_ab.py \
      tests/integration/test_i4_1_6b_dense_ab_contract.py \
      tests/unit/test_i4_1_6_decoupled_sampling_integration.py \
      tests/integration/test_i4_1_6_decoupled_sampling_contract.py \
      tests/unit/test_i4_1_object_orbit_qualification.py \
      tests/unit/test_i2_full_video_quality_qualification.py \
      tests/unit/test_i1_offline_dense_campaign.py \
      tests/unit/test_e3_qualified_scene_fusion_expansion.py \
      tests/unit/test_f_confidence_artifact_filtering.py \
      tests/unit/test_g_persistent_dense_rgb_geometry.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_i4_1_8c_final_object_orbit_causality_contract.py \
      tests/integration/test_i4_1_8b_retained_core_object_support_contract.py \
      tests/integration/test_i4_1_8a_retained_core_target_diagnosis_contract.py \
      tests/integration/test_i4_1_8_production_binding_contract.py \
      tests/integration/test_i4_1_8_preview_canvas_contract.py \
      tests/integration/test_i4_1_7b_bounded_target_correction_contract.py \
      tests/integration/test_i4_1_7a_final_geometry_diagnosis_contract.py \
      tests/integration/test_i4_1_7_clean_static_orbit_contract.py \
      tests/integration/test_i4_1_6b_dense_ab_contract.py \
      tests/integration/test_i4_1_6_decoupled_sampling_contract.py \
      tests/integration/test_i4_1_i3_closure_contract.py
    ;;
  stress)
    python -m pytest -q \
      tests/stress/test_i4_1_8c_final_object_orbit_causality_stress.py \
      tests/stress/test_i4_1_8b_retained_core_object_support_stress.py \
      tests/stress/test_i4_1_8a_retained_core_target_diagnosis_stress.py \
      tests/stress/test_i4_1_8_production_binding_stress.py \
      tests/stress/test_i4_1_7b_bounded_target_correction_stress.py \
      tests/stress/test_i4_1_7a_final_geometry_diagnosis_stress.py \
      tests/stress/test_i4_1_7_clean_static_orbit_stress.py \
      tests/stress/test_i4_1_6b_dense_ab_stress.py \
      tests/stress/test_i4_1_6_decoupled_sampling_stress.py \
      tests/stress/test_i4_1_closure_contract_stress.py
    ;;
  affected)
    python -m pytest -q \
      tests/unit/test_i4_1_8c_final_object_orbit_causality.py \
      tests/integration/test_i4_1_8c_final_object_orbit_causality_contract.py \
      tests/stress/test_i4_1_8c_final_object_orbit_causality_stress.py \
      tests/unit/test_i4_1_8b_retained_core_object_support.py \
      tests/integration/test_i4_1_8b_retained_core_object_support_contract.py \
      tests/stress/test_i4_1_8b_retained_core_object_support_stress.py \
      tests/unit/test_i4_1_8a_retained_core_target_diagnosis.py \
      tests/integration/test_i4_1_8a_retained_core_target_diagnosis_contract.py \
      tests/stress/test_i4_1_8a_retained_core_target_diagnosis_stress.py \
      tests/unit/test_i4_1_8_production_binding.py \
      tests/integration/test_i4_1_8_production_binding_contract.py \
      tests/integration/test_i4_1_8_preview_canvas_contract.py \
      tests/stress/test_i4_1_8_production_binding_stress.py \
      tests/unit/test_i4_1_7b_bounded_target_correction.py \
      tests/integration/test_i4_1_7b_bounded_target_correction_contract.py \
      tests/stress/test_i4_1_7b_bounded_target_correction_stress.py \
      tests/unit/test_i4_1_7a_final_geometry_diagnosis.py \
      tests/integration/test_i4_1_7a_final_geometry_diagnosis_contract.py \
      tests/stress/test_i4_1_7a_final_geometry_diagnosis_stress.py \
      tests/unit/test_i4_1_7_clean_static_orbit_qualification.py \
      tests/integration/test_i4_1_7_clean_static_orbit_contract.py \
      tests/stress/test_i4_1_7_clean_static_orbit_stress.py \
      tests/unit/test_i4_1_6b_decoupled_dense_ab.py \
      tests/integration/test_i4_1_6b_dense_ab_contract.py \
      tests/stress/test_i4_1_6b_dense_ab_stress.py \
      tests/unit/test_i4_1_6_decoupled_sampling_integration.py \
      tests/integration/test_i4_1_6_decoupled_sampling_contract.py \
      tests/stress/test_i4_1_6_decoupled_sampling_stress.py \
      tests/unit/test_i4_1_object_orbit_qualification.py \
      tests/integration/test_i4_1_i3_closure_contract.py \
      tests/stress/test_i4_1_closure_contract_stress.py \
      tests/unit/test_i2_full_video_quality_qualification.py \
      tests/unit/test_i1_offline_dense_campaign.py \
      tests/unit/test_e3_qualified_scene_fusion_expansion.py \
      tests/unit/test_f_confidence_artifact_filtering.py \
      tests/unit/test_g_persistent_dense_rgb_geometry.py \
      tests/unit/test_h_dense_rgb_reconstruction_ui.py \
      tests/ui/test_h_dense_rgb_reconstruction_ui.py \
      tests/ui/test_phase9_3b_canonical_geometry_viewer.py \
      tests/ui/test_phase9_3f_scene_layers.py \
      tests/ui/test_viewport_controls.py \
      tests/ui/test_viewport_layout.py
    ;;
  *) echo "usage: $0 [focused|quick|integration|stress|affected]" >&2; exit 2;;
esac
