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

def test_nerf_build_producer_emits_public_environment_namespace(
    tmp_path: Path,
) -> None:
    from app.reconstruction_nerf_build import _isolated_environment_overrides

    prefix = tmp_path / "nerf-runtime"
    python = prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")

    nerfstudio_root = tmp_path / "nerfstudio"
    tcnn_root = tmp_path / "tiny-cuda-nn"
    nerfstudio_root.mkdir()
    tcnn_root.mkdir()

    overrides = dict(
        _isolated_environment_overrides(
            python,
            nerfstudio_root=nerfstudio_root,
            tcnn_root=tcnn_root,
        )
    )

    assert overrides["DEPTH3D_NERFSTUDIO_ROOT"] == str(nerfstudio_root)
    assert overrides["DEPTH3D_TCNN_ROOT"] == str(tcnn_root)
    assert "EIA_NERFSTUDIO_ROOT" not in overrides
    assert "EIA_TCNN_ROOT" not in overrides


def test_interactive_client_scrubs_legacy_and_emits_public_namespace() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (
        root / "app/widgets/reconstruction_nerf_worker_client.py"
    ).read_text(encoding="utf-8")

    assert 'env.pop("EIA_NERFSTUDIO_ROOT", None)' in text
    assert 'env.pop("EIA_TCNN_ROOT", None)' in text
    assert 'env["DEPTH3D_NERFSTUDIO_ROOT"]' in text
    assert 'env["DEPTH3D_TCNN_ROOT"]' in text

    assert 'env["EIA_NERFSTUDIO_ROOT"] =' not in text
    assert 'env["EIA_TCNN_ROOT"] =' not in text

def test_qualification_helpers_use_public_nerf_environment_namespace() -> None:
    root = Path(__file__).resolve().parents[2]

    helpers = (
        "scripts/nerf_e4_support.py",
        "scripts/reconstruction_engine_checkpoint_worker.py",
        "scripts/run_nerf_backend_qualification.py",
        "scripts/run_nerf_real_scene_validation.py",
        "scripts/run_phase9_5e_real_interactive_render.py",
        "scripts/run_phase9_5g_real_nerf_resource_check.py",
        "scripts/run_reconstruction_f4_real_backends.py",
    )

    for relative in helpers:
        source = (root / relative).read_text(encoding="utf-8")

        assert "DEPTH3D_NERFSTUDIO_ROOT" in source
        assert "DEPTH3D_TCNN_ROOT" in source
        assert "EIA_NERFSTUDIO_ROOT" not in source
        assert "EIA_TCNN_ROOT" not in source


def test_e4_worker_environment_emits_public_nerf_namespace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from scripts.nerf_e4_support import worker_environment

    nerfstudio_root = tmp_path / "nerfstudio"
    tcnn_root = tmp_path / "tiny-cuda-nn"

    monkeypatch.setenv("EIA_NERFSTUDIO_ROOT", "/legacy/nerfstudio")
    monkeypatch.setenv("EIA_TCNN_ROOT", "/legacy/tcnn")

    env = worker_environment(
        nerfstudio_root=nerfstudio_root,
        tcnn_root=tcnn_root,
    )

    assert env["DEPTH3D_NERFSTUDIO_ROOT"] == str(nerfstudio_root)
    assert env["DEPTH3D_TCNN_ROOT"] == str(tcnn_root)
