from __future__ import annotations

import json
from pathlib import Path

from core.reconstruction.i5_dynamic_scene_handling import build_i5_5a_dynamic_scene_admission, build_i5_5a_source_identity
from tests.i5_5a_helpers import I54C_SHA, RECEIPT_55A_SHA, binding, castle_source_dir, passing_i5_4c, passing_i5_5a_receipt


def test_i55a_contract_is_json_serializable_and_freezes_b_revalidation(tmp_path: Path) -> None:
    result = build_i5_5a_dynamic_scene_admission(
        i5_4c_result=passing_i5_4c(),
        i5_4c_binding=binding("/evidence/i5_4c.json", I54C_SHA),
        source_identity=build_i5_5a_source_identity(castle_source_dir(tmp_path)),
        source_label="Castle-P19 static multiview control",
        static_control_confirmation=True,
        admission_receipt=passing_i5_5a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_5a_receipt.json", RECEIPT_55A_SHA),
    )
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "dynamic_region_feature_bearing_revalidation" in encoded
    assert result["stress_plan"]["required_i5_5b_revalidation"]["result_driven_recipe_tuning"] is False
    assert result["stress_plan"]["required_i5_5b_revalidation"]["same_production_sfm_configuration_for_control_and_stress"] is True


def test_i55a_control_and_stress_share_identical_source_asset_binding(tmp_path: Path) -> None:
    result = build_i5_5a_dynamic_scene_admission(
        i5_4c_result=passing_i5_4c(),
        i5_4c_binding=binding("/evidence/i5_4c.json", I54C_SHA),
        source_identity=build_i5_5a_source_identity(castle_source_dir(tmp_path)),
        source_label="Castle-P19 static multiview control",
        static_control_confirmation=True,
        admission_receipt=passing_i5_5a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_5a_receipt.json", RECEIPT_55A_SHA),
    )
    plan = result["stress_plan"]
    assert plan["static_control"]["source_asset_set_sha256"] == plan["dynamic_foreground_proxy"]["source_asset_set_sha256"]
