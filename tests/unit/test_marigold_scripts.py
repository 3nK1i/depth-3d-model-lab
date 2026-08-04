"""Marigold installer and pinned snapshot downloader tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.download_marigold_model import download_snapshot
from scripts.install_marigold import install


class _Api:
    def model_info(self, repository, revision):
        assert repository == "prs-eth/marigold-depth-v1-1"
        assert revision == "main"
        return SimpleNamespace(sha="abc123")


def test_downloader_pins_resolved_revision_and_writes_atomic_marker(tmp_path) -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model_index.json").write_text("{}\n", encoding="utf-8")
        return str(destination)

    destination = tmp_path / "snapshot"
    payload = download_snapshot(
        destination=destination,
        api_factory=_Api,
        snapshot_downloader=fake_download,
    )
    assert calls[-1]["revision"] == "abc123"
    assert payload["resolved_revision"] == "abc123"
    marker = json.loads((destination / ".ready").read_text(encoding="utf-8"))
    assert marker["repository"] == "prs-eth/marigold-depth-v1-1"
    assert not (destination / ".tmp").exists()


def test_downloader_reuses_complete_snapshot_without_network(tmp_path) -> None:
    destination = tmp_path / "snapshot"
    destination.mkdir()
    (destination / "model_index.json").write_text("{}\n", encoding="utf-8")
    expected = {"resolved_revision": "cached"}
    (destination / ".ready").write_text(json.dumps(expected), encoding="utf-8")
    payload = download_snapshot(
        destination=destination,
        api_factory=lambda: (_ for _ in ()).throw(AssertionError("network")),
    )
    assert payload == expected


def test_installer_dry_run_does_not_execute_pip(capsys) -> None:
    install(dry_run=True)
    assert "pip install" in capsys.readouterr().out
