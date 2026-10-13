from __future__ import annotations

import importlib
from pathlib import Path

import pytest


CHECKOUT_WORKERS = (
    "scripts.nerf_workers.nerfstudio_worker",
    "scripts.nerf_workers.nerfstudio_e3_worker",
    "scripts.nerf_workers.nerfstudio_e4_worker",
)

ENVIRONMENT_PAIRS = (
    ("DEPTH3D_NERFSTUDIO_ROOT", "EIA_NERFSTUDIO_ROOT"),
    ("DEPTH3D_TCNN_ROOT", "EIA_TCNN_ROOT"),
)


@pytest.mark.parametrize("module_name", CHECKOUT_WORKERS)
@pytest.mark.parametrize(("public_name", "legacy_name"), ENVIRONMENT_PAIRS)
def test_checkout_workers_accept_legacy_environment_fallback(
    monkeypatch,
    tmp_path: Path,
    module_name: str,
    public_name: str,
    legacy_name: str,
) -> None:
    module = importlib.import_module(module_name)
    legacy_root = tmp_path / "legacy-checkout"
    (legacy_root / ".git").mkdir(parents=True)

    monkeypatch.delenv(public_name, raising=False)
    monkeypatch.setenv(legacy_name, str(legacy_root))
    monkeypatch.setattr(module, "_git_revision", lambda _path: "abc123")

    resolved, revision = module._require_checkout(
        public_name,
        "abc",
        legacy_env_name=legacy_name,
    )

    assert resolved == legacy_root.resolve()
    assert revision == "abc123"


@pytest.mark.parametrize("module_name", CHECKOUT_WORKERS)
@pytest.mark.parametrize(("public_name", "legacy_name"), ENVIRONMENT_PAIRS)
def test_checkout_workers_prefer_public_environment_namespace(
    monkeypatch,
    tmp_path: Path,
    module_name: str,
    public_name: str,
    legacy_name: str,
) -> None:
    module = importlib.import_module(module_name)

    public_root = tmp_path / "public-checkout"
    legacy_root = tmp_path / "legacy-checkout"
    (public_root / ".git").mkdir(parents=True)
    (legacy_root / ".git").mkdir(parents=True)

    monkeypatch.setenv(public_name, str(public_root))
    monkeypatch.setenv(legacy_name, str(legacy_root))
    monkeypatch.setattr(module, "_git_revision", lambda _path: "abc123")

    resolved, revision = module._require_checkout(
        public_name,
        "abc",
        legacy_env_name=legacy_name,
    )

    assert resolved == public_root.resolve()
    assert revision == "abc123"


@pytest.mark.parametrize(("public_name", "legacy_name"), ENVIRONMENT_PAIRS)
def test_interactive_worker_accepts_legacy_environment_fallback(
    monkeypatch,
    public_name: str,
    legacy_name: str,
) -> None:
    module = importlib.import_module(
        "scripts.nerf_workers.nerfstudio_interactive_worker"
    )

    monkeypatch.delenv(public_name, raising=False)
    monkeypatch.setenv(legacy_name, "/legacy/backend")

    assert module._environment_value(
        public_name,
        legacy_name=legacy_name,
    ) == "/legacy/backend"


@pytest.mark.parametrize(("public_name", "legacy_name"), ENVIRONMENT_PAIRS)
def test_interactive_worker_prefers_public_environment_namespace(
    monkeypatch,
    public_name: str,
    legacy_name: str,
) -> None:
    module = importlib.import_module(
        "scripts.nerf_workers.nerfstudio_interactive_worker"
    )

    monkeypatch.setenv(public_name, "/public/backend")
    monkeypatch.setenv(legacy_name, "/legacy/backend")

    assert module._environment_value(
        public_name,
        legacy_name=legacy_name,
    ) == "/public/backend"


def test_worker_call_sites_use_public_namespace_with_legacy_fallback() -> None:
    root = Path(__file__).resolve().parents[2]

    for relative in (
        "scripts/nerf_workers/nerfstudio_worker.py",
        "scripts/nerf_workers/nerfstudio_e3_worker.py",
        "scripts/nerf_workers/nerfstudio_e4_worker.py",
        "scripts/nerf_workers/nerfstudio_interactive_worker.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")

        assert "DEPTH3D_NERFSTUDIO_ROOT" in text
        assert "DEPTH3D_TCNN_ROOT" in text
        assert "EIA_NERFSTUDIO_ROOT" in text
        assert "EIA_TCNN_ROOT" in text
