from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_h2_candidate_state_is_preserved_as_historical_evidence() -> None:
    h2 = _read("docs/phase_history/PHASE8_9H2_RELEASE_PREPARATION.md")
    assert "Prepare truthful, release-facing Phase 8 documentation" in h2
    assert "0.7.5 / Phase 7.5 Stable / 7.5" in h2
    assert "v0.8.0 tag        -> absent" in h2
    assert "reserved for 8.9I" in h2


def test_readme_preserves_reconstruction_authority_boundaries() -> None:
    readme = _read("README.md")
    required = (
        "canonical geometry is the authoritative scene geometry",
        "semantic geometry supplies semantic/object context",
        "Gaussian and NeRF/Nerfacto outputs are **appearance only**",
        "`certified_metrology=false`",
        "camera intrinsics alone do not establish metric scale",
        "shared Live 3D render card",
        "dedicated Reconstruction workflow",
    )
    for phrase in required:
        assert phrase in readme


def test_v080_release_notes_are_finalized_after_h2_preparation() -> None:
    notes = _read("RELEASE_NOTES_v0.8.0.md")
    assert "Release status" in notes
    assert "Release-preparation status" not in notes
    assert "0.8.0` / `Phase 8 Reconstruction Stable" in notes
    assert "annotated `v0.8.0` tag" in notes
    assert "Gaussian and NeRF/Nerfacto representations are appearance only" in notes
    assert "Known limitations and deliberate exclusions" in notes
    assert "8.9I2" in notes and "8.9I3" in notes


def test_current_metadata_has_transitioned_after_h2_history() -> None:
    version_py = _read("core/version.py")
    pyproject = _read("pyproject.toml")
    app_yaml = _read("config/app.yaml")

    assert '__version__ = "0.8.0"' in version_py
    assert 'RELEASE_NAME = "Phase 8 Reconstruction Stable"' in version_py
    assert "RELEASE_PHASE = 8.0" in version_py
    assert 'version = "0.8.0"' in pyproject
    assert 'version: "0.8.0"' in app_yaml
    assert 'release: "Phase 8 Reconstruction Stable"' in app_yaml


def test_h1_h2_phase_history_records_exist_and_define_release_boundary() -> None:
    h1 = _read("docs/phase_history/PHASE8_9H1_RELEASE_INVENTORY_AUDIT.md")
    h2 = _read("docs/phase_history/PHASE8_9H2_RELEASE_PREPARATION.md")
    assert "read-only" in h1.lower()
    assert "dff62357874e28de8d0ceadb9a222dd0cd17d999" in h1
    assert "H2 must not modify production code" in h1
    assert "The coordinated version transition and annotated tag are reserved for 8.9I" in h2
    assert "No `app/`, `core/`, `native/`" in h2
