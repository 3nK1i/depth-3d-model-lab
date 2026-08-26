from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_visual_ab_closure import (
    MVS_VISUAL_AB_REQUIRED_VIEWS,
    MVS_VISUAL_AB_REVIEW_DIMENSIONS,
    build_a3_1_manual_closure,
    require_absent_output,
    sha256_file,
    validate_a3_manual_review_evidence,
)

UNIT_PATH = Path(__file__).resolve().parents[1] / "unit/test_i4_1_9a_mvs_visual_ab_closure.py"
_spec = importlib.util.spec_from_file_location("a31_unit_helpers", UNIT_PATH)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_fixture = _helpers._fixture


def test_a31_detects_mutated_review_contact_sheet(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    path = Path(summary["artifacts"]["renders"]["oblique"]["ab"])
    path.write_bytes(b"mutated-review-image")
    with pytest.raises(ReconstructionValidationError, match="render hash mismatch"):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_detects_missing_review_render(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    path = Path(summary["artifacts"]["renders"]["top"]["mvs"])
    path.unlink()
    with pytest.raises(ReconstructionValidationError, match="render is missing"):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_detects_mutated_mvs_fused_cloud_after_review(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    path = Path(summary["source_artifacts"]["mvs_fused_ply"])
    path.write_bytes(path.read_bytes() + b"mutated")
    with pytest.raises(ReconstructionValidationError, match="mvs_fused_ply"):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_detects_mutated_frozen_a1_authority_manifest(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    a2 = Path(summary["source_artifacts"]["mvs_a2_dir"])
    path = a2 / "evidence/frozen_authority/a1_camera_authority_manifest.json"
    path.write_text('{"authority":"changed"}\n', encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="a1_camera_authority_manifest"):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_rejects_incomplete_manual_decision_matrix(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    del summary["decision"]["decision_matrix"]["mvs_better_but_imperfect"]
    with pytest.raises(ReconstructionValidationError, match="decision matrix"):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_rejects_any_preselected_automatic_winner(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    summary["decision"]["automatic_winner_selected"] = True
    with pytest.raises(ReconstructionValidationError, match="manual-verdict"):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_closure_is_deterministic_for_identical_reviewed_evidence(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    evidence = validate_a3_manual_review_evidence(summary, a3_dir=a3)
    kwargs = dict(
        evidence_validation=evidence,
        verdict="mvs_better_but_imperfect",
        confirmed=True,
        a3_summary_sha256="c" * 64,
    )
    assert build_a3_1_manual_closure(summary, **kwargs) == build_a3_1_manual_closure(copy.deepcopy(summary), **kwargs)


def test_a31_immutable_closure_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "i4_1_9a_a3_1_closure.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite"):
        require_absent_output(output)
