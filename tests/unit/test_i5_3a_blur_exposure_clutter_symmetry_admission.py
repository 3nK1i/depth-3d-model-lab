from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_blur_exposure_clutter_symmetry import (
    I53A_PHASE,
    I53AStressDesignPolicy,
    build_i5_3a_blur_exposure_clutter_symmetry_admission,
    build_i5_3a_stress_plan,
    validate_i5_2c_for_i5_3,
    validate_i5_3a_receipt,
)
from core.reconstruction.i5_low_overlap_weak_parallax import build_ordered_source_identity
from tests.i5_3a_helpers import I52C_SHA, RECEIPT_53A_SHA, clone, passing_i5_2c, passing_i5_3a_receipt, source_dir


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def _build(tmp_path: Path, *, confirmed: bool = True) -> dict:
    closure = passing_i5_2c(tmp_path)
    source = build_ordered_source_identity(source_dir(tmp_path))
    return build_i5_3a_blur_exposure_clutter_symmetry_admission(
        i5_2c_result=closure,
        i5_2c_binding=_binding("/evidence/i5_2c.json", I52C_SHA),
        source_identity=source,
        controlled_source_confirmation=confirmed,
        admission_receipt=passing_i5_3a_receipt(),
        admission_receipt_binding=_binding("/evidence/i5_3a_receipt.json", RECEIPT_53A_SHA),
    )


def test_i53a_accepts_frozen_i52c_handoff(tmp_path: Path) -> None:
    result = validate_i5_2c_for_i5_3(passing_i5_2c(tmp_path))
    assert result["decision"]["i5_2_complete"] is True
    assert result["decision"]["i5_3_authorized"] is True
    assert result["decision"]["i5_3_started"] is False


def test_i53a_rejects_invalid_i52c_handoff(tmp_path: Path) -> None:
    result = clone(passing_i5_2c(tmp_path))
    result["decision"]["i5_3_authorized"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2c_for_i5_3(result)


def test_i53a_plan_is_deterministic_and_like_for_like(tmp_path: Path) -> None:
    identity = build_ordered_source_identity(source_dir(tmp_path))
    first = build_i5_3a_stress_plan(identity)
    second = build_i5_3a_stress_plan(identity)
    assert first == second
    assert set(first) == {
        "control",
        "blur_proxy",
        "underexposure_proxy",
        "overexposure_proxy",
        "clutter_proxy",
        "symmetry_proxy",
    }
    names = first["control"]["names"]
    assert len(names) == 19
    for scenario in first.values():
        assert scenario["names"] == names
        assert scenario["image_count"] == 19
    assert first["blur_proxy"]["transform"]["kernel_size"] == 13
    assert first["underexposure_proxy"]["transform"]["gain"] == pytest.approx(0.35)
    assert first["overexposure_proxy"]["transform"]["gain"] == pytest.approx(1.55)
    assert first["clutter_proxy"]["transform"]["source_offset"] == 7
    assert first["symmetry_proxy"]["transform"]["band_fraction"] == pytest.approx(0.18)


def test_i53a_stress_labels_remain_proxies_until_i53b(tmp_path: Path) -> None:
    plan = build_i5_3a_stress_plan(build_ordered_source_identity(source_dir(tmp_path)))
    for key in ("blur_proxy", "underexposure_proxy", "overexposure_proxy", "clutter_proxy", "symmetry_proxy"):
        assert plan[key]["stress_proxy"] is True
        assert plan[key]["semantic_stress_authoritative"] is False
        assert plan[key]["i5_3b_revalidation_required"] is True


def test_i53a_binds_same_source_as_i52c(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["source_identity"]["asset_set_sha256"] == result["predecessor"]["i5_2_source_asset_set_sha256"]
    assert result["scope"]["same_control_source_as_i5_2"] is True


def test_i53a_rejects_source_identity_drift(tmp_path: Path) -> None:
    closure = passing_i5_2c(tmp_path)
    other = source_dir(tmp_path / "other")
    changed = other / "castle00.png"
    changed.write_bytes(changed.read_bytes() + b"drift")
    identity = build_ordered_source_identity(other)
    with pytest.raises(ReconstructionValidationError):
        build_i5_3a_blur_exposure_clutter_symmetry_admission(
            i5_2c_result=closure,
            i5_2c_binding=_binding("/evidence/i5_2c.json", I52C_SHA),
            source_identity=identity,
            controlled_source_confirmation=True,
            admission_receipt=passing_i5_3a_receipt(),
            admission_receipt_binding=_binding("/evidence/i5_3a_receipt.json", RECEIPT_53A_SHA),
        )


def test_i53a_requires_explicit_controlled_source_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError):
        _build(tmp_path, confirmed=False)


def test_i53a_policy_rejects_invalid_transform_parameters() -> None:
    with pytest.raises(ReconstructionValidationError):
        I53AStressDesignPolicy(blur_kernel_size=12)
    with pytest.raises(ReconstructionValidationError):
        I53AStressDesignPolicy(underexposure_gain=1.0)
    with pytest.raises(ReconstructionValidationError):
        I53AStressDesignPolicy(clutter_band_fraction=0.75)


def test_i53a_receipt_requires_execution_free_green_gate() -> None:
    receipt = validate_i5_3a_receipt(passing_i5_3a_receipt())
    assert receipt["scope"]["sfm_rerun"] is False
    bad = clone(passing_i5_3a_receipt())
    bad["scope"]["stress_images_materialized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3a_receipt(bad)


def test_i53a_builds_admission_and_authorizes_only_i53b(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["phase"] == I53A_PHASE == "I5.3A"
    assert result["pass"] is True
    assert result["decision"]["i5_3_started"] is True
    assert result["decision"]["i5_3a_complete"] is True
    assert result["decision"]["i5_3b_real_replay_authorized"] is True
    assert result["decision"]["i5_3_complete"] is False
    assert result["decision"]["i5_4_authorized"] is False
    assert result["next"] == "I5.3B current frozen-foundation blur / exposure / clutter / symmetry replay."
