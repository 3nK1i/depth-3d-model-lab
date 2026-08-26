from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction.offline_dense_i4_clean_static_orbit_diagnosis import (
    build_i4_1_7a_orbit_geometry_diagnosis,
)
from core.reconstruction.offline_dense_i4_orbit_qualification import orbit_coverage_metrics
from tests.unit.test_i4_1_7_clean_static_orbit_qualification import _arc


def test_i4_1_7a_contract_is_diagnosis_only_and_keeps_authority_fail_closed() -> None:
    matrices = _arc(40, 220.0)
    provisional = np.asarray([6.0, 0.0, 0.0])
    source = {
        "phase": "I4.1.7",
        "status": "degraded",
        "assessment": {"selected_orbit": orbit_coverage_metrics(matrices, provisional)},
    }
    core = np.asarray([[x, y, z] for x in (-0.2, 0.0, 0.2, 0.4) for y in (-0.2, 0.0, 0.2, 0.4) for z in (-0.2, 0.0, 0.2, 0.4)])
    diagnosis = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=source,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        provisional_target_reconstruction=provisional,
        retained_core_points=core,
    )
    assert diagnosis["status"] == "diagnosis_ready"
    assert diagnosis["authority"]["sfm_rerun"] is False
    assert diagnosis["authority"]["da3_rerun"] is False
    assert diagnosis["authority"]["production_binding_executed"] is False
    assert diagnosis["decision"]["authorizes_i4_1_8_production_binding"] is False
    assert diagnosis["authority"]["metric_scale_claimed"] is False


def test_i4_1_7a_runner_exists_as_additive_tool() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "scripts/run_i4_1_7a_final_geometry_diagnosis.py").is_file()
    assert (root / "core/reconstruction/offline_dense_i4_clean_static_orbit_diagnosis.py").is_file()
