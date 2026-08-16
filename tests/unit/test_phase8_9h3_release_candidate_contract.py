from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_h3_candidate_state_remains_historical_after_release_transition() -> None:
    h3 = _read("docs/phase_history/PHASE8_9H3_RELEASE_CANDIDATE_QUALIFICATION.md")
    readme = _read("README.md")
    notes = _read("RELEASE_NOTES_v0.8.0.md")
    assert "release candidate" in h3.lower()
    assert "local and remote `v0.8.0` tags remain absent" in h3
    assert "Version `0.9.0` is the **Phase 9 Digital Twin Stable** release" in readme
    assert "v0.8.0 Release Notes" in notes
    assert "Release status" in notes


def test_reconstruction_authority_and_metric_limits_are_explicit() -> None:
    readme = _read("README.md")
    notes = _read("RELEASE_NOTES_v0.8.0.md")
    readme_lower = readme.lower()
    notes_lower = notes.lower()
    phrases = (
        "canonical geometry",
        "semantic geometry",
        "appearance only",
        "`certified_metrology=false`",
        "camera intrinsics alone do not establish metric scale",
    )
    for phrase in phrases:
        assert phrase.lower() in readme_lower
        assert phrase.lower() in notes_lower


def test_future_work_is_not_claimed_as_v080_functionality() -> None:
    readme = _read("README.md")
    notes = _read("RELEASE_NOTES_v0.8.0.md")
    assert "Known limitations and deliberate exclusions" in notes
    assert "post-`v0.8.0`" in notes
    assert "fully interactive neural free-view" in notes
    assert "unified digital-twin workspace" in notes
    assert "future work" in notes


def test_backend_isolation_and_mesh_boundary_are_documented() -> None:
    readme = _read("README.md")
    notes = _read("RELEASE_NOTES_v0.8.0.md")
    assert "heavy reconstruction, Gaussian and NeRF runtimes remain outside the production `depth` environment" in readme
    assert "Heavy/native/neural reconstruction runtimes remain outside the production `depth` environment" in notes
    assert "shared Live 3D render card" in readme
    assert "shared Live 3D `Mesh (Poisson)` render-card action remains disabled" in notes
    assert "dedicated Reconstruction workflow" in readme


def test_current_release_metadata_has_advanced_beyond_v080_identity() -> None:
    version_py = _read("core/version.py")
    pyproject = _read("pyproject.toml")
    app_yaml = _read("config/app.yaml")
    assert '__version__ = "0.9.0"' in version_py
    assert 'RELEASE_NAME = "Phase 9 Digital Twin Stable"' in version_py
    assert "RELEASE_PHASE = 9.0" in version_py
    assert 'version = "0.9.0"' in pyproject
    assert 'version: "0.9.0"' in app_yaml
    assert 'release: "Phase 9 Digital Twin Stable"' in app_yaml


def test_h3_record_reserves_release_transition_for_i() -> None:
    h3 = _read("docs/phase_history/PHASE8_9H3_RELEASE_CANDIDATE_QUALIFICATION.md")
    assert "qualification-only" in h3
    assert "dff62357874e28de8d0ceadb9a222dd0cd17d999" in h3
    assert "local and remote `v0.8.0` tags remain absent" in h3
    assert "reserve" in h3.lower() and "8.9I" in h3
    assert "Do not create the `v0.8.0` tag in H3" in h3
