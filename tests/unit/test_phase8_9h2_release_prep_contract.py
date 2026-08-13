from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_describes_phase8_release_candidate_without_claiming_release() -> None:
    readme = _read("README.md")
    assert "target `v0.8.0` (Phase 8 Reconstruction)" in readme
    assert "runtime/package metadata intentionally remains `0.7.5`" in readme
    assert "no `v0.8.0` tag is created" in readme
    assert "## Phase 8 Reconstruction" in readme
    assert "Version `0.7.5` is the stable Phase 7.5 measurement and inspection release" not in readme
    assert "planned after Phase 8" not in readme
    assert "post-`v0.8.0`" in readme


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


def test_v080_release_notes_are_prepared_not_released() -> None:
    notes = _read("RELEASE_NOTES_v0.8.0.md")
    assert "Release-preparation status" in notes
    assert "the `v0.8.0` tag must remain absent" in notes
    assert "annotated `v0.8.0` tag" in notes
    assert "Gaussian and NeRF/Nerfacto representations are appearance only" in notes
    assert "Known limitations and deliberate exclusions" in notes
    assert "8.9I" in notes


def test_h2_does_not_bump_canonical_release_metadata() -> None:
    version_py = _read("core/version.py")
    pyproject = _read("pyproject.toml")
    app_yaml = _read("config/app.yaml")

    assert '__version__ = "0.7.5"' in version_py
    assert 'RELEASE_NAME = "Phase 7.5 Stable"' in version_py
    assert "RELEASE_PHASE = 7.5" in version_py
    assert 'version = "0.7.5"' in pyproject
    assert 'version: "0.7.5"' in app_yaml
    assert 'release: "Phase 7.5 Stable"' in app_yaml

    assert '__version__ = "0.8.0"' not in version_py
    assert 'version = "0.8.0"' not in pyproject
    assert 'version: "0.8.0"' not in app_yaml


def test_h1_h2_phase_history_records_exist_and_define_release_boundary() -> None:
    h1 = _read("docs/phase_history/PHASE8_9H1_RELEASE_INVENTORY_AUDIT.md")
    h2 = _read("docs/phase_history/PHASE8_9H2_RELEASE_PREPARATION.md")
    assert "read-only" in h1.lower()
    assert "dff62357874e28de8d0ceadb9a222dd0cd17d999" in h1
    assert "H2 must not modify production code" in h1
    assert "The coordinated version transition and annotated tag are reserved for 8.9I" in h2
    assert "No `app/`, `core/`, `native/`" in h2
