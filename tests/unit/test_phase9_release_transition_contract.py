from __future__ import annotations

from pathlib import Path

from core.runtime.release import project_root, version_consistency_check
from core.version import RELEASE_NAME, RELEASE_PHASE, __version__


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v090_release_identity_is_exact_and_consistent() -> None:
    assert __version__ == "0.9.0"
    assert RELEASE_NAME == "Phase 9 Digital Twin Stable"
    assert RELEASE_PHASE == 9.0
    check = version_consistency_check(project_root())
    assert check.passed, check.detail


def test_release_facing_docs_describe_phase9_without_overclaiming_authority() -> None:
    readme = _read("README.md")
    notes = _read("RELEASE_NOTES_v0.9.0.md")
    assert "Version `0.9.0` is the **Phase 9 Digital Twin Stable** release" in readme
    assert "0.9.0` / `Phase 9 Digital Twin Stable" in notes
    for phrase in (
        "Canonical geometry",
        "appearance only",
        "certified_metrology=false",
        "intrinsics alone",
        "global loop closure",
        "photorealistic",
    ):
        assert phrase.lower() in (readme + notes).lower()


def test_phase8_release_evidence_remains_historical_and_present() -> None:
    notes = _read("RELEASE_NOTES_v0.8.0.md")
    transition = _read("docs/phase_history/PHASE8_9I2_RELEASE_TRANSITION.md")
    assert "v0.8.0 Release Notes" in notes
    assert "Phase 8.9I2 — v0.8.0 Release Transition" in transition
    assert 'RELEASE_NAME  = "Phase 8 Reconstruction Stable"' in transition


def test_transition_is_metadata_documentation_and_test_alignment_only() -> None:
    doc = _read("docs/phase_history/PHASE9_RELEASE_TRANSITION.md")
    runner = _read("scripts/run_phase9_release_transition_tests.sh")
    assert "does not reopen Phase 9.6 implementation" in doc
    assert "must never create a Git tag" in doc
    for forbidden in ("git tag -a", "git tag v0.9.0", "git push origin v0.9.0"):
        assert forbidden not in runner
