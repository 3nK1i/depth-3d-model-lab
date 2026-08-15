from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
CLIENT = ROOT / "app/widgets/reconstruction_nerf_worker_client.py"


def test_5h_ui_keeps_interactive_preview_hide_controls_resources_and_authority_text() -> None:
    page = PAGE.read_text(encoding="utf-8")
    for text in (
        'QPushButton("Interactive")',
        'QPushButton("Preview")',
        'QPushButton("Hide")',
        '"Render detail"',
        '"Opacity"',
        '"Auto refresh"',
        '"Resources"',
        '"Performance"',
        '"Lifecycle"',
        '"NeRF appearance"',
    ):
        assert text in page
    assert "canonical geometry remains" in page.lower()
    assert "appearance only" in page.lower()


def test_5h_ui_shutdown_path_releases_external_worker_and_keeps_process_group_cleanup() -> None:
    page = PAGE.read_text(encoding="utf-8")
    client = CLIENT.read_text(encoding="utf-8")
    assert "self._nerf_worker_client.stop_blocking()" in page
    assert "os.killpg" in client
    assert "signal.SIGTERM" in client
    assert "signal.SIGKILL" in client
    assert "shell=True" not in client
